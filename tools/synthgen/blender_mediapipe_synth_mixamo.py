"""Synthetic water-polo dataset generator for the CapAI MediaPipe pose pipeline
— Mixamo edition.

Same output contract as ``blender_mediapipe_synth.py`` (per-frame `.jpg` +
`.json` mirroring the backend `/analyze` schema: 33 MediaPipe landmarks
`[x, y, z, visibility]`, joint angles, visibility tiers, person bbox), but
driven by **Mixamo-rigged human bodies** (the standard `mixamorig:` skeleton)
instead of a Rigify metarig.

Design choices are grounded in water-polo biomechanics for accuracy, not looks:
  * Eggbeater legs — hip flexion 30-75°, hip abduction 10-30° (compact, not a
    split), knee flexion 60-110°, ankles sculling, left/right out of phase.
  * Shooting arm — overhead high-elbow cock (hand by/behind the head); the
    other arm extends for counter-balance.
  * Camera orbits the athlete (front + both sides, slight-above to elevated)
    so the set covers the viewpoints real footage is shot from.
  * Per-frame randomized skin/suit/cap colours, lighting and waterline.

It discovers every Mixamo armature in the scene (one collection each) and
cycles through them, so a male + female body yield a mixed dataset. Each frame
renders only the active athlete (others hidden).

───────────────────────────────────────────────────────────────
HOW TO RUN
───────────────────────────────────────────────────────────────
The scene must already contain the imported Mixamo bodies (one collection per
athlete, each with an armature whose bones are named `mixamorig:*` and its
skinned surface mesh) and, optionally, a cleaned cap mesh to fit per head.
Set the CONFIG below and run from Blender's Scripting tab, or headless:

  blender scene.blend --background --python blender_mediapipe_synth_mixamo.py

The script reads/writes data only — it never saves your .blend.
"""

import json
import math
import os
import random

import bpy
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
ATHLETE_COLLECTIONS = ["XBot", "YBot"]   # one Mixamo body per collection
CAP_SOURCE = "geometry_0"                # cleaned, watertight cap to clone per head (or None)
WATER_PLANE_NAME = "Waterline"

OUTPUT_DIR = "/tmp/capai_mp_mixamo"
NUM_FRAMES = 200                         # total, split evenly across athletes
RES_X, RES_Y = 1280, 720
SEED = None

# Camera orbit around the athlete's torso (degrees / metres).
CAM_AZIMUTH = (-120, 120)                # 0 = front; covers front + both sides
CAM_ELEVATION = (-5, 30)                 # pool-deck to elevated/broadcast
CAM_DIST_FACTOR = (1.1, 1.5)             # framing margin on the bounding sphere (1.0 = tight)

LIGHT_ENERGY = (2.0, 7.0)
LIGHT_ANGLE = (-45, 45)

WATER_OFFSET = (-0.15, 0.20)             # waterline height around the torso centre

VIS_VISIBLE = (0.85, 1.00)
VIS_OCCLUDED = (0.30, 0.60)

# Skin tone palette (linear-ish RGB) the body base colour is sampled from.
SKIN_TONES = [
    (0.80, 0.62, 0.47), (0.66, 0.47, 0.34), (0.50, 0.34, 0.23),
    (0.36, 0.24, 0.16), (0.90, 0.74, 0.62), (0.27, 0.18, 0.12),
]

# ═══════════════════════════════════════════════════════════════
# Pose ranges (degrees, Mixamo local bone axes — validated by render QA).
# Sign conventions discovered empirically:
#   UpLeg +X = hip flexion forward; UpLeg ±Z = abduction (L=+, R=-).
#   Leg  -X = knee flexion.  Arm raise = Z, sign flips per side (R=-, L=+).
# ═══════════════════════════════════════════════════════════════
HIP_FLEX = (30, 75)
HIP_ABDUCT = (10, 30)
HIP_DRIFT = (5, 20)          # out-of-phase alternation between legs
KNEE_FLEX = (60, 110)
ANKLE_X = (-30, 15)
ANKLE_Z = (-20, 20)

SHOOT_ARM_RAISE = (80, 120)  # |Z| on the shooting upper arm
SHOOT_ARM_TILT = (-40, 0)    # X on the shooting upper arm
SHOOT_ELBOW = (50, 90)       # |Y| on the shooting forearm (high-elbow cock)
OFF_ARM_RAISE = (20, 50)     # |Z| on the relaxed arm (extended for balance)
OFF_ELBOW = (10, 40)         # |Y| on the relaxed forearm

# ═══════════════════════════════════════════════════════════════
# MediaPipe BlazePose 33 landmarks → Mixamo bone points.
# Body joints map to real bone heads/tails; the 11 face points are synthesized
# from a head reference frame (no facial bones on these mannequins — flagged for
# later refinement, and caps cover the head region anyway).
# ═══════════════════════════════════════════════════════════════
MP_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]
FACE_SET = set(MP_NAMES[0:11])

# Body landmark -> (bone, head|tail). 'L'/'R' here is the athlete's own side,
# which is what MediaPipe's left_/right_ refer to.
BODY_BONES = {
    "left_shoulder":   ("mixamorig:LeftArm", "head"),
    "right_shoulder":  ("mixamorig:RightArm", "head"),
    "left_elbow":      ("mixamorig:LeftForeArm", "head"),
    "right_elbow":     ("mixamorig:RightForeArm", "head"),
    "left_wrist":      ("mixamorig:LeftHand", "head"),
    "right_wrist":     ("mixamorig:RightHand", "head"),
    "left_pinky":      ("mixamorig:LeftHandPinky1", "head"),
    "right_pinky":     ("mixamorig:RightHandPinky1", "head"),
    "left_index":      ("mixamorig:LeftHandIndex1", "head"),
    "right_index":     ("mixamorig:RightHandIndex1", "head"),
    "left_thumb":      ("mixamorig:LeftHandThumb1", "head"),
    "right_thumb":     ("mixamorig:RightHandThumb1", "head"),
    "left_hip":        ("mixamorig:LeftUpLeg", "head"),
    "right_hip":       ("mixamorig:RightUpLeg", "head"),
    "left_knee":       ("mixamorig:LeftLeg", "head"),
    "right_knee":      ("mixamorig:RightLeg", "head"),
    "left_ankle":      ("mixamorig:LeftFoot", "head"),
    "right_ankle":     ("mixamorig:RightFoot", "head"),
    "left_foot_index": ("mixamorig:LeftToeBase", "tail"),
    "right_foot_index":("mixamorig:RightToeBase", "tail"),
    # heels handled specially (extrapolated behind the ankle)
}


# ═══════════════════════════════════════════════════════════════
# Pose helpers
# ═══════════════════════════════════════════════════════════════
def deg(a, b):
    return math.radians(random.uniform(a, b))


def _set(arm, name, x=0.0, y=0.0, z=0.0):
    pb = arm.pose.bones.get(name)
    if pb is None:
        return
    pb.rotation_mode = "XYZ"
    pb.rotation_euler = (x, y, z)


def reset_pose(arm):
    for pb in arm.pose.bones:
        pb.rotation_mode = "XYZ"
        pb.rotation_euler = (0.0, 0.0, 0.0)


def randomize_pose(arm):
    """Eggbeater legs + shooting wind-up, in Mixamo local axes."""
    reset_pose(arm)
    drift = deg(*HIP_DRIFT)
    for side, sgn in (("Left", 1.0), ("Right", -1.0)):
        _set(arm, f"mixamorig:{side}UpLeg",
             x=deg(*HIP_FLEX) + sgn * drift, z=sgn * deg(*HIP_ABDUCT))
        _set(arm, f"mixamorig:{side}Leg", x=-deg(*KNEE_FLEX))
        _set(arm, f"mixamorig:{side}Foot", x=deg(*ANKLE_X), z=deg(*ANKLE_Z))

    shoot, off = random.choice([("Right", "Left"), ("Left", "Right")])
    ssgn = -1.0 if shoot == "Right" else 1.0
    osgn = -1.0 if off == "Right" else 1.0
    # Elbow flexion is about the forearm's X axis (Y is the twist axis — rotating
    # it leaves the elbow straight). +X flexes the hand up/back toward the head.
    _set(arm, f"mixamorig:{shoot}Arm", x=deg(*SHOOT_ARM_TILT), z=ssgn * deg(*SHOOT_ARM_RAISE))
    _set(arm, f"mixamorig:{shoot}ForeArm", x=deg(*SHOOT_ELBOW))
    _set(arm, f"mixamorig:{off}Arm", z=osgn * deg(*OFF_ARM_RAISE))
    _set(arm, f"mixamorig:{off}ForeArm", x=deg(*OFF_ELBOW))


# ═══════════════════════════════════════════════════════════════
# Geometry helpers
# ═══════════════════════════════════════════════════════════════
def bone_pt(arm, name, end="head"):
    pb = arm.pose.bones.get(name)
    if pb is None:
        return None
    return arm.matrix_world @ (pb.head if end == "head" else pb.tail)


def torso_center(arm):
    pts = [bone_pt(arm, b) for b in ("mixamorig:LeftArm", "mixamorig:RightArm",
                                     "mixamorig:LeftUpLeg", "mixamorig:RightUpLeg")]
    pts = [p for p in pts if p]
    return sum(pts, Vector((0, 0, 0))) / len(pts) if pts else arm.matrix_world.translation.copy()


def head_frame(arm):
    """(centre, up_dir, left_dir, radius) for synthesizing face landmarks."""
    h = bone_pt(arm, "mixamorig:Head", "head")
    top = bone_pt(arm, "mixamorig:HeadTop_End", "head") or bone_pt(arm, "mixamorig:Head", "tail")
    l = bone_pt(arm, "mixamorig:LeftArm", "head")
    r = bone_pt(arm, "mixamorig:RightArm", "head")
    centre = (h + top) * 0.5
    up = (top - h).normalized()
    left = (l - r).normalized()
    radius = (top - h).length * 0.5
    return centre, up, left, radius


def face_point(name, fr):
    c, up, left, rad = fr
    table = {
        "nose":            c + up * (-0.05 * rad),
        "left_eye_inner":  c + left * (0.20 * rad) + up * (0.20 * rad),
        "left_eye":        c + left * (0.40 * rad) + up * (0.20 * rad),
        "left_eye_outer":  c + left * (0.55 * rad) + up * (0.20 * rad),
        "right_eye_inner": c - left * (0.20 * rad) + up * (0.20 * rad),
        "right_eye":       c - left * (0.40 * rad) + up * (0.20 * rad),
        "right_eye_outer": c - left * (0.55 * rad) + up * (0.20 * rad),
        "left_ear":        c + left * (1.00 * rad),
        "right_ear":       c - left * (1.00 * rad),
        "mouth_left":      c + left * (0.15 * rad) - up * (0.30 * rad),
        "mouth_right":     c - left * (0.15 * rad) - up * (0.30 * rad),
    }
    return table[name]


def landmark_world(arm, name, face_fr):
    if name in FACE_SET:
        return face_point(name, face_fr)
    if name == "left_heel" or name == "right_heel":
        side = "Left" if name.startswith("left") else "Right"
        ankle = bone_pt(arm, f"mixamorig:{side}Foot", "head")
        toe = bone_pt(arm, f"mixamorig:{side}ToeBase", "head")
        if ankle and toe:
            return ankle + (ankle - toe) * 0.25      # behind the ankle
        return ankle
    bone, end = BODY_BONES[name]
    return bone_pt(arm, bone, end)


# ═══════════════════════════════════════════════════════════════
# Projection / labels (shared with the Rigify variant)
# ═══════════════════════════════════════════════════════════════
def project(scene, cam, co):
    v = world_to_camera_view(scene, cam, co)
    return v.x, 1.0 - v.y, v.z, v.z > 0.0


def vis_score(x, y, in_front, world_z, water_z):
    if not in_front or not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        return 0.0
    return random.uniform(*(VIS_VISIBLE if world_z > water_z else VIS_OCCLUDED))


def vis_tiers(scores):
    return {
        "tracked": sum(1 for s in scores if s >= 0.75),
        "partial": sum(1 for s in scores if 0.30 <= s < 0.75),
        "estimated": sum(1 for s in scores if s < 0.30),
    }


def angle_2d(a, b, c, aspect):
    ba = ((a[0] - b[0]) * aspect, a[1] - b[1])
    bc = ((c[0] - b[0]) * aspect, c[1] - b[1])
    na, nc = math.hypot(*ba), math.hypot(*bc)
    if na == 0 or nc == 0:
        return None
    cosv = (ba[0] * bc[0] + ba[1] * bc[1]) / (na * nc)
    return round(math.degrees(math.acos(max(-1.0, min(1.0, cosv)))), 1)


def compute_angles(lms, aspect):
    p = [(l[0], l[1]) for l in lms]
    ls, rs = p[11], p[12]
    tilt = math.degrees(math.atan2(rs[1] - ls[1], (rs[0] - ls[0]) * aspect))
    if tilt > 90:
        tilt -= 180
    elif tilt < -90:
        tilt += 180
    return {
        "elbow_l": angle_2d(p[11], p[13], p[15], aspect),
        "elbow_r": angle_2d(p[12], p[14], p[16], aspect),
        "knee_l": angle_2d(p[23], p[25], p[27], aspect),
        "knee_r": angle_2d(p[24], p[26], p[28], aspect),
        "shoulder_tilt": round(tilt, 1),
    }


def landmark_row(scene, cam, arm, water_z):
    face_fr = head_frame(arm)
    world = [landmark_world(arm, n, face_fr) for n in MP_NAMES]
    hip_mid = (world[23] + world[24]) * 0.5
    sh_mid = (world[11] + world[12]) * 0.5
    scale = (sh_mid - hip_mid).length or 1.0
    _, _, hip_depth, _ = project(scene, cam, hip_mid)
    rows = []
    for wco in world:
        x, y, depth, in_front = project(scene, cam, wco)
        v = vis_score(x, y, in_front, wco.z, water_z)
        rows.append([round(min(max(x, 0), 1), 6), round(min(max(y, 0), 1), 6),
                     round((depth - hip_depth) / scale, 6), round(v, 4)])
    return rows


def surface_bbox(scene, cam, surf):
    dg = bpy.context.evaluated_depsgraph_get()
    ev = surf.evaluated_get(dg)
    mesh = ev.to_mesh()
    M = ev.matrix_world
    xs, ys = [], []
    try:
        for v in mesh.vertices:
            co = world_to_camera_view(scene, cam, M @ v.co)
            if co.z <= 0:
                continue
            xs.append(co.x); ys.append(1.0 - co.y)
    finally:
        ev.to_mesh_clear()
    if not xs:
        return None
    x0, x1 = max(0, min(xs)), min(1, max(xs))
    y0, y1 = max(0, min(ys)), min(1, max(ys))
    if x1 <= x0 or y1 <= y0:
        return None
    return [round((x0 + x1) / 2, 6), round((y0 + y1) / 2, 6),
            round(x1 - x0, 6), round(y1 - y0, 6)]


# ═══════════════════════════════════════════════════════════════
# Scene wiring
# ═══════════════════════════════════════════════════════════════
def srgb(c):
    return (c[0], c[1], c[2], 1.0)


def ensure_principled(mat_name, color):
    mat = bpy.data.materials.get(mat_name) or bpy.data.materials.new(mat_name)
    mat.use_nodes = True
    for n in mat.node_tree.nodes:
        if n.type == "BSDF_PRINCIPLED":
            n.inputs["Base Color"].default_value = srgb(color)
    return mat


def set_principled_color(mat, color):
    if not mat or not mat.use_nodes:
        return
    for n in mat.node_tree.nodes:
        if n.type == "BSDF_PRINCIPLED":
            n.inputs["Base Color"].default_value = srgb(color)


def discover_athletes():
    """One athlete per configured collection: its rig, surface, and cap clone."""
    out = []
    for cname in ATHLETE_COLLECTIONS:
        coll = bpy.data.collections.get(cname)
        if not coll:
            continue
        rig = next((o for o in coll.objects if o.type == "ARMATURE"), None)
        surf = next((o for o in coll.objects if o.type == "MESH"
                     and any(m.type == "ARMATURE" for m in o.modifiers)), None)
        if surf is None:  # fall back to largest mesh
            meshes = [o for o in coll.objects if o.type == "MESH"]
            surf = max(meshes, key=lambda m: len(m.data.vertices)) if meshes else None
        if rig and surf:
            out.append({"coll": coll, "rig": rig, "surface": surf,
                        "skin_mat": ensure_principled(f"Skin_{cname}", SKIN_TONES[0])})
    return out


def fit_cap(athlete, cname):
    """Clone the cap, scale it to the head, and lock it to the head bone.

    Uses a Child-Of constraint (not bone-parenting) so there's no bone-length
    tail offset — the cap stays seated on the head through every pose.
    """
    src = bpy.data.objects.get(CAP_SOURCE) if CAP_SOURCE else None
    if src is None:
        return None
    rig = athlete["rig"]
    head = bone_pt(rig, "mixamorig:Head", "head")
    top = bone_pt(rig, "mixamorig:HeadTop_End", "head")
    head_h = (top - head).length
    cap = src.copy(); cap.data = src.data.copy(); cap.name = f"Cap_{cname}"
    athlete["coll"].objects.link(cap)
    cap.rotation_euler = (0.0, 0.0, 0.0); cap.scale = (1.0, 1.0, 1.0)
    bpy.context.view_layer.update()
    # Scale so the cap is ~1.7x the head-bone length (covers crown down past the ears).
    s = (head_h * 1.7) / (cap.dimensions.z or 1.0)
    cap.scale = (s, s, s)
    cap_mat = ensure_principled("Cap", (0.1, 0.1, 0.4))
    cap.data.materials.clear(); cap.data.materials.append(cap_mat)
    bpy.context.view_layer.update()
    # Seat the cap's bbox centre on the head centre (nudged up so it sits as a cap).
    centre = (head + top) * 0.5
    cur = sum((cap.matrix_world @ Vector(c) for c in cap.bound_box), Vector((0, 0, 0))) / 8.0
    cap.location = cap.location + (centre - cur) + Vector((0, 0, head_h * 0.1))
    bpy.context.view_layer.update()
    # Follow the head bone, freezing the current world transform.
    bone_world = rig.matrix_world @ rig.pose.bones["mixamorig:Head"].matrix
    con = cap.constraints.new("CHILD_OF")
    con.target = rig; con.subtarget = "mixamorig:Head"
    con.inverse_matrix = bone_world.inverted()
    return cap


def isolate_render(active, athletes):
    """Render only the active athlete's surface + cap; hide the rest."""
    for a in athletes:
        on = a is active
        for o in a["coll"].objects:
            if o.type in {"MESH"}:
                o.hide_render = not on


def main():
    if SEED is not None:
        random.seed(SEED)
    scene = bpy.context.scene
    aspect = RES_X / RES_Y

    athletes = discover_athletes()
    if not athletes:
        raise RuntimeError("No Mixamo athletes found in the configured collections.")
    for i, a in enumerate(athletes):
        cname = ATHLETE_COLLECTIONS[i] if i < len(ATHLETE_COLLECTIONS) else f"A{i}"
        # skin material on the surface
        a["surface"].data.materials.clear()
        a["surface"].data.materials.append(a["skin_mat"])
        a["cap"] = fit_cap(a, cname)

    cam = bpy.data.objects.get("Camera")
    light = next((o for o in bpy.data.objects if o.type == "LIGHT"), None)
    water = bpy.data.objects.get(WATER_PLANE_NAME)
    if water is None:
        me = bpy.data.meshes.new(WATER_PLANE_NAME)
        me.from_pydata([(-50, -50, 0), (50, -50, 0), (50, 50, 0), (-50, 50, 0)], [], [(0, 1, 2, 3)])
        me.update()
        water = bpy.data.objects.new(WATER_PLANE_NAME, me)
        scene.collection.objects.link(water)

    # World ambient so no frame renders near-black (real footage is never pitch dark).
    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (0.55, 0.68, 0.82, 1.0)  # soft pool-blue fill
        bg.inputs["Strength"].default_value = 0.6

    scene.camera = cam
    scene.render.resolution_x = RES_X
    scene.render.resolution_y = RES_Y
    scene.render.image_settings.file_format = "JPEG"
    scene.render.image_settings.quality = 90

    img_dir = os.path.join(OUTPUT_DIR, "images")
    lbl_dir = os.path.join(OUTPUT_DIR, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    for i in range(NUM_FRAMES):
        athlete = athletes[i % len(athletes)]
        arm, surf = athlete["rig"], athlete["surface"]
        cname = ATHLETE_COLLECTIONS[i % len(athletes)]
        stem = f"frame_{i:05d}"

        isolate_render(athlete, athletes)
        randomize_pose(arm)
        bpy.context.view_layer.update()

        target = torso_center(arm)
        # Fit the WHOLE body in frame: aim at the surface bbox centre and set the
        # distance from the body's bounding sphere + the camera's vertical FOV, so
        # no joints get cropped at any orbit angle.
        corners = [surf.matrix_world @ Vector(c) for c in surf.bound_box]
        bb_center = sum(corners, Vector((0, 0, 0))) / 8.0
        radius = max((c - bb_center).length for c in corners)
        az = math.radians(random.uniform(*CAM_AZIMUTH))
        el = math.radians(random.uniform(*CAM_ELEVATION))
        direction = Vector((math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), math.sin(el)))
        vfov = 2.0 * math.atan(math.tan(cam.data.angle / 2.0) * (RES_Y / RES_X))
        dist = radius / math.sin(vfov / 2.0) * random.uniform(*CAM_DIST_FACTOR)
        cam.location = bb_center + direction * dist
        cam.rotation_euler = (bb_center - cam.location).to_track_quat("-Z", "Y").to_euler()

        if light:
            if hasattr(light.data, "energy"):
                light.data.energy = random.uniform(*LIGHT_ENERGY)
            light.rotation_euler = (math.radians(random.uniform(*LIGHT_ANGLE)),
                                    math.radians(random.uniform(*LIGHT_ANGLE)),
                                    math.radians(random.uniform(0, 360)))

        set_principled_color(athlete["skin_mat"], random.choice(SKIN_TONES))
        set_principled_color(bpy.data.materials.get("Cap"),
                             (random.random(), random.random(), random.random()))

        water_z = target.z + random.uniform(*WATER_OFFSET)
        water.location.z = water_z
        bpy.context.view_layer.update()

        scene.render.filepath = os.path.join(img_dir, stem + ".jpg")
        bpy.ops.render.render(write_still=True)

        lms = landmark_row(scene, cam, arm, water_z)
        record = {
            "index": i, "image": f"images/{stem}.jpg", "athlete": cname,
            "landmarks": lms, "angles": compute_angles(lms, aspect),
            "visibility": vis_tiers([l[3] for l in lms]),
            "bbox": surface_bbox(scene, cam, surf),
        }
        with open(os.path.join(lbl_dir, stem + ".json"), "w") as fh:
            json.dump(record, fh)
        print(f"[{i + 1}/{NUM_FRAMES}] {stem} ({cname}) written")

    meta = {
        "format": "capai-mediapipe-pose-synthetic", "rig": "Mixamo (mixamorig)",
        "model": "MediaPipe BlazePose (33 landmarks)",
        "landmark_names": MP_NAMES, "landmark_fields": ["x", "y", "z", "visibility"],
        "athletes": ATHLETE_COLLECTIONS,
        "angles": ["elbow_l", "elbow_r", "knee_l", "knee_r", "shoulder_tilt"],
        "resolution": [RES_X, RES_Y],
        "notes": "Face landmarks (0-10) synthesized from a head frame; refine later.",
    }
    with open(os.path.join(OUTPUT_DIR, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"Done. {NUM_FRAMES} frames -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
