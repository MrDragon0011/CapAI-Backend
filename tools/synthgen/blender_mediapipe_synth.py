"""Synthetic water-polo dataset generator for the CapAI MediaPipe pose pipeline.

Renders a rigged athlete in randomized shooting / eggbeater poses under
randomized cameras, lighting, and kit colors, then writes a paired
`.jpg` + `.json` for every frame. Each JSON mirrors the CapAI backend
`/analyze` per-frame schema, so the set drops straight in as ground truth
for the 33-point MediaPipe (BlazePose) landmarks **and** the joint
kinematics (elbow / knee / shoulder-tilt) the service computes:

    {
      "index": 0,
      "image": "images/frame_00000.jpg",
      "landmarks": [[x, y, z, visibility], ...33 entries...],
      "angles": {"elbow_l": .., "elbow_r": .., "knee_l": .., "knee_r": ..,
                 "shoulder_tilt": ..},
      "visibility": {"tracked": .., "partial": .., "estimated": ..},
      "bbox": [cx, cy, w, h]            # normalized person box, may be null
    }

Landmark x / y are normalized to the image (top-left origin, matching
MediaPipe and the image itself); z is depth relative to the hip midpoint
(negative = closer to camera) on roughly the same scale as x, as MediaPipe
reports it; visibility is in [0, 1]. Angles follow the backend exactly:
shoulder-elbow-wrist and hip-knee-ankle triplets plus the shoulder-line
tilt from horizontal, with normalized x scaled by the image aspect ratio so
the angles reflect true geometry. Visibility tiers match the service:
tracked (>= 0.75), partial (0.30-0.74), estimated (< 0.30).

───────────────────────────────────────────────────────────────
HOW TO RUN
───────────────────────────────────────────────────────────────
Inside Blender (GUI):
  1. Open the .blend that contains your rigged athlete: an armature whose
     deform bones follow the standard Rigify metarig naming (upper_arm.L,
     forearm.L, hand.L, thigh.L, shin.L, foot.L, heel.02.L, toe.L, eye.L,
     ear.L, nose, lip.*, f_index.01.L, ...), the body mesh, a camera, and at
     least one light.
  2. Edit the CONFIG block below so the names + OUTPUT_DIR match your scene.
  3. Open the 'Scripting' workspace tab, then Text ▸ Open this file.
  4. Press 'Run Script' (▶). Frames land in OUTPUT_DIR/images and
     OUTPUT_DIR/labels, plus a meta.json describing the schema.

Headless (no UI, much faster for big batches):
  blender yourscene.blend --background --python blender_mediapipe_synth.py

The script only reads/writes data — it never saves your .blend, so the
randomized pose left in the scene afterwards is harmless.
"""

import json
import math
import os
import random

import bpy
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view

# ═══════════════════════════════════════════════════════════════
# CONFIG — edit these to match your scene
# ═══════════════════════════════════════════════════════════════
ARMATURE_NAME = "Metarig"          # armature driving the deform / holding the bones below
BODY_MESH_NAME = "Body"            # mesh used to compute the 2D bounding box
CAMERA_NAME = "Camera"             # camera to render through
LIGHT_NAME = "Sun"                 # light to jitter (falls back to first light if absent)
CAP_MATERIAL = "Cap"              # material whose Base Color = swim-cap color
SUIT_MATERIAL = "Suit"            # material whose Base Color = swimsuit color
WATER_PLANE_NAME = "Waterline"    # horizontal plane; auto-created at origin if missing

OUTPUT_DIR = "/tmp/capai_mp_synth"  # dataset root (images/ + labels/ created inside)
NUM_FRAMES = 200                   # how many frames to generate
RES_X, RES_Y = 1280, 720           # render resolution
SEED = None                        # set an int for reproducible runs, or None

# Camera: a base spot in front of the swimmer (metarig faces -Y) + per-frame jitter.
CAM_BASE_LOC = (0.0, -6.0, 1.2)
CAM_JITTER = (1.5, 1.0, 0.8)       # +/- metres of deck-side wander per axis

# Lighting jitter.
LIGHT_ENERGY = (2.0, 7.0)          # sun strength range
LIGHT_ANGLE = (-45, 45)            # +/- degrees of pitch & yaw

# Waterline height relative to the torso centre (metres). Hips sit ~0 here, so a
# small negative-to-positive band keeps shoulders dry and legs submerged.
WATER_OFFSET = (-0.15, 0.20)

# Visibility values written for each tier (MediaPipe reports a [0,1] score). A
# little jitter keeps the ground truth from being a step function.
VIS_VISIBLE = (0.85, 1.00)         # above the waterline, in frame
VIS_OCCLUDED = (0.30, 0.60)        # below the waterline (submerged) but in frame

# ═══════════════════════════════════════════════════════════════
# Pose ranges (degrees). Applied as local euler rotation off the rest pose, so
# (0,0,0) == rest. One arm per frame is the "shooting" arm (big wind-up); the
# other gets the calmer OFF_ARM range. Tune axis ranges to your rig's bone roll.
# ═══════════════════════════════════════════════════════════════
SHOOT_ARM = {
    "upper_arm": {"x": (-130, -40), "y": (-30, 30), "z": (-45, 45)},  # raise high & back
    "forearm":   {"x": (35, 120),  "y": (-25, 25), "z": (-25, 25)},   # high-elbow catch
    "hand":      {"x": (-45, 45),  "y": (-35, 35), "z": (-35, 35)},   # wrist snap
}
OFF_ARM = {
    "upper_arm": {"x": (-40, 20),  "y": (-20, 20), "z": (-25, 25)},
    "forearm":   {"x": (10, 70),   "y": (-15, 15), "z": (-15, 15)},
    "hand":      {"x": (-25, 25),  "y": (-25, 25), "z": (-25, 25)},
}
# Eggbeater legs: knees stay bent forward, thighs + ankles churn. L/R drift out of
# phase to mimic the alternating sweep.
LEG = {
    "thigh": {"x": (-50, 50),  "y": (-25, 25), "z": (-20, 20)},
    "shin":  {"x": (15, 100),  "y": (-15, 15), "z": (-15, 15)},
    "foot":  {"x": (-35, 35),  "y": (-30, 30), "z": (-30, 30)},
}

# ═══════════════════════════════════════════════════════════════
# MediaPipe BlazePose 33 landmarks, in order. Each maps to one or more
# Rigify metarig bone points (bone, head|tail, weight); the landmark is the
# weighted average of whichever of those bones exist. Blends let us synthesize
# the face sub-points (eye corners, mouth corners) MediaPipe wants but a single
# bone doesn't name. Missing bones are skipped; if a whole landmark resolves to
# nothing it falls back to HEAD_BONE so the row stays 33 long.
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

MP_BONES = {
    "nose":            [("nose", "head", 1.0)],
    "left_eye_inner":  [("eye.L", "head", 0.7), ("nose", "head", 0.3)],
    "left_eye":        [("eye.L", "head", 1.0)],
    "left_eye_outer":  [("eye.L", "head", 0.7), ("ear.L", "head", 0.3)],
    "right_eye_inner": [("eye.R", "head", 0.7), ("nose", "head", 0.3)],
    "right_eye":       [("eye.R", "head", 1.0)],
    "right_eye_outer": [("eye.R", "head", 0.7), ("ear.R", "head", 0.3)],
    "left_ear":        [("ear.L", "head", 1.0)],
    "right_ear":       [("ear.R", "head", 1.0)],
    "mouth_left":      [("lip.T.L.001", "tail", 0.5), ("lip.B.L.001", "tail", 0.5)],
    "mouth_right":     [("lip.T.R.001", "tail", 0.5), ("lip.B.R.001", "tail", 0.5)],
    "left_shoulder":   [("upper_arm.L", "head", 1.0)],
    "right_shoulder":  [("upper_arm.R", "head", 1.0)],
    "left_elbow":      [("forearm.L", "head", 1.0)],
    "right_elbow":     [("forearm.R", "head", 1.0)],
    "left_wrist":      [("hand.L", "head", 1.0)],
    "right_wrist":     [("hand.R", "head", 1.0)],
    "left_pinky":      [("f_pinky.01.L", "head", 1.0)],
    "right_pinky":     [("f_pinky.01.R", "head", 1.0)],
    "left_index":      [("f_index.01.L", "head", 1.0)],
    "right_index":     [("f_index.01.R", "head", 1.0)],
    "left_thumb":      [("thumb.01.L", "head", 1.0)],
    "right_thumb":     [("thumb.01.R", "head", 1.0)],
    "left_hip":        [("thigh.L", "head", 1.0)],
    "right_hip":       [("thigh.R", "head", 1.0)],
    "left_knee":       [("shin.L", "head", 1.0)],
    "right_knee":      [("shin.R", "head", 1.0)],
    "left_ankle":      [("foot.L", "head", 1.0)],
    "right_ankle":     [("foot.R", "head", 1.0)],
    "left_heel":       [("heel.02.L", "head", 1.0)],
    "right_heel":      [("heel.02.R", "head", 1.0)],
    "left_foot_index": [("toe.L", "tail", 1.0)],
    "right_foot_index":[("toe.R", "tail", 1.0)],
}
HEAD_BONE = "spine.006"  # fallback for face points on a rig without face bones

# Landmark indices used by the kinematics (must match MP_NAMES order above).
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════
def require_object(name):
    obj = bpy.data.objects.get(name)
    if obj is None:
        raise RuntimeError(f"Object '{name}' not found — fix the CONFIG block.")
    return obj


def deg(rng):
    """Random radians from a (min_deg, max_deg) range."""
    return math.radians(random.uniform(*rng))


def set_bone_euler(arm, bone_name, ranges, phase=0.0):
    """Apply a randomized local rotation to a pose bone (no-op if it's missing)."""
    pb = arm.pose.bones.get(bone_name)
    if pb is None:
        return
    pb.rotation_mode = "XYZ"
    pb.rotation_euler = (
        deg(ranges["x"]) + phase,
        deg(ranges["y"]),
        deg(ranges["z"]),
    )


def randomize_pose(arm):
    """Shooting wind-up on one arm + relaxed other arm + eggbeater legs."""
    shoot_side, off_side = random.choice([("L", "R"), ("R", "L")])
    for seg in ("upper_arm", "forearm", "hand"):
        set_bone_euler(arm, f"{seg}.{shoot_side}", SHOOT_ARM[seg])
        set_bone_euler(arm, f"{seg}.{off_side}", OFF_ARM[seg])

    # Legs churn out of phase: push one thigh forward while the other drives back.
    drift = math.radians(random.uniform(15, 40))
    for side, sign in (("L", 1.0), ("R", -1.0)):
        set_bone_euler(arm, f"thigh.{side}", LEG["thigh"], phase=sign * drift)
        set_bone_euler(arm, f"shin.{side}", LEG["shin"])
        set_bone_euler(arm, f"foot.{side}", LEG["foot"])


def srgb_to_linear(c):
    """Blender Base Color is linear; convert an 8-bit sRGB channel into it."""
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def random_color_rgba():
    """A random hex colour, returned as a linear RGBA tuple for a Principled BSDF."""
    rgb = [srgb_to_linear(random.randint(0, 255) / 255.0) for _ in range(3)]
    return (rgb[0], rgb[1], rgb[2], 1.0)


def set_material_color(mat_name, rgba):
    mat = bpy.data.materials.get(mat_name)
    if not mat or not mat.use_nodes:
        return
    for node in mat.node_tree.nodes:
        if node.type == "BSDF_PRINCIPLED":
            node.inputs["Base Color"].default_value = rgba
            return


def bone_point_world(arm, bone_name, end):
    """World-space position of a pose bone's head/tail, or None if absent."""
    pb = arm.pose.bones.get(bone_name)
    if pb is None:
        return None
    local = pb.head if end == "head" else pb.tail
    return arm.matrix_world @ local


def landmark_world(arm, spec, head_fallback):
    """Weighted blend of the bone points in `spec`, over whichever bones exist."""
    acc = Vector((0.0, 0.0, 0.0))
    total = 0.0
    for bone, end, weight in spec:
        p = bone_point_world(arm, bone, end)
        if p is not None:
            acc += p * weight
            total += weight
    if total > 0.0:
        return acc / total
    return head_fallback


def torso_center(arm):
    """Midpoint of the four shoulder/hip joints — the look-at + waterline anchor."""
    pts = [
        bone_point_world(arm, "upper_arm.L", "head"),
        bone_point_world(arm, "upper_arm.R", "head"),
        bone_point_world(arm, "thigh.L", "head"),
        bone_point_world(arm, "thigh.R", "head"),
    ]
    pts = [p for p in pts if p is not None]
    if not pts:
        return arm.matrix_world.translation.copy()
    return sum(pts, Vector((0, 0, 0))) / len(pts)


def get_or_make_waterline():
    """Reuse the configured plane, or drop a large horizontal one at the origin."""
    plane = bpy.data.objects.get(WATER_PLANE_NAME)
    if plane is None:
        mesh = bpy.data.meshes.new(WATER_PLANE_NAME)
        size = 50.0
        verts = [(-size, -size, 0), (size, -size, 0), (size, size, 0), (-size, size, 0)]
        mesh.from_pydata(verts, [], [(0, 1, 2, 3)])
        mesh.update()
        plane = bpy.data.objects.new(WATER_PLANE_NAME, mesh)
        bpy.context.scene.collection.objects.link(plane)
    return plane


def randomize_environment(scene, cam, arm, light, water):
    """Jitter camera, lighting, kit colours and the waterline. Returns waterline Z."""
    target = torso_center(arm)

    # Camera: base spot + deck-side wander, always re-aimed at the torso.
    cam.location = Vector(CAM_BASE_LOC) + Vector(
        (random.uniform(-CAM_JITTER[0], CAM_JITTER[0]),
         random.uniform(-CAM_JITTER[1], CAM_JITTER[1]),
         random.uniform(-CAM_JITTER[2], CAM_JITTER[2]))
    )
    direction = target - cam.location
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

    # Lighting: strength + pitch/yaw.
    if light is not None:
        if hasattr(light.data, "energy"):
            light.data.energy = random.uniform(*LIGHT_ENERGY)
        light.rotation_euler = (
            math.radians(random.uniform(*LIGHT_ANGLE)),
            math.radians(random.uniform(*LIGHT_ANGLE)),
            math.radians(random.uniform(0, 360)),
        )

    # Kit: fresh random cap + suit colour.
    set_material_color(CAP_MATERIAL, random_color_rgba())
    set_material_color(SUIT_MATERIAL, random_color_rgba())

    # Waterline height around the torso.
    water_z = target.z + random.uniform(*WATER_OFFSET)
    water.location.z = water_z
    return water_z


def project(scene, cam, world_co):
    """Project a world point to (x, y, depth, in_front).

    world_to_camera_view returns (0,0) at the bottom-left and a positive .z for
    points in front of the camera; images / MediaPipe use a top-left origin, so
    y is flipped. `depth` is that camera-space distance, used for the relative z.
    """
    co = world_to_camera_view(scene, cam, world_co)
    return co.x, 1.0 - co.y, co.z, co.z > 0.0


def visibility_score(x, y, in_front, world_z, water_z):
    """MediaPipe-style [0,1] score: 0 off-frame, low if submerged, high if dry."""
    if not in_front or not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        return 0.0
    return random.uniform(*(VIS_VISIBLE if world_z > water_z else VIS_OCCLUDED))


def vis_tiers(scores):
    """Bucket visibility scores into the backend's tracked/partial/estimated tiers."""
    tracked = sum(1 for s in scores if s >= 0.75)
    partial = sum(1 for s in scores if 0.30 <= s < 0.75)
    estimated = sum(1 for s in scores if s < 0.30)
    return {"tracked": tracked, "partial": partial, "estimated": estimated}


def angle_2d(a, b, c, aspect):
    """Angle at vertex b (degrees) for points a-b-c, x scaled by aspect ratio."""
    ba = ((a[0] - b[0]) * aspect, a[1] - b[1])
    bc = ((c[0] - b[0]) * aspect, c[1] - b[1])
    na = math.hypot(*ba)
    nc = math.hypot(*bc)
    if na == 0.0 or nc == 0.0:
        return None
    cosv = (ba[0] * bc[0] + ba[1] * bc[1]) / (na * nc)
    return round(math.degrees(math.acos(max(-1.0, min(1.0, cosv)))), 1)


def compute_angles(landmarks, aspect):
    """Elbow / knee / shoulder-tilt, matching the backend's conventions.

    Each landmark is [x, y, z, visibility]; angles use the x/y plane only.
    """
    p = [(lm[0], lm[1]) for lm in landmarks]
    ls, rs = p[L_SHOULDER], p[R_SHOULDER]
    # Tilt of the shoulder line from horizontal, folded to [-90, 90] so a level
    # line reads ~0 regardless of which shoulder is image-left (MediaPipe's
    # left_shoulder sits on image-right for a front-facing athlete).
    tilt = math.degrees(math.atan2(rs[1] - ls[1], (rs[0] - ls[0]) * aspect))
    if tilt > 90.0:
        tilt -= 180.0
    elif tilt < -90.0:
        tilt += 180.0
    tilt = round(tilt, 1)
    return {
        "elbow_l": angle_2d(p[L_SHOULDER], p[L_ELBOW], p[L_WRIST], aspect),
        "elbow_r": angle_2d(p[R_SHOULDER], p[R_ELBOW], p[R_WRIST], aspect),
        "knee_l": angle_2d(p[L_HIP], p[L_KNEE], p[L_ANKLE], aspect),
        "knee_r": angle_2d(p[R_HIP], p[R_KNEE], p[R_ANKLE], aspect),
        "shoulder_tilt": tilt,
    }


def compute_bbox(scene, cam, body):
    """Normalized (cx, cy, w, h) around the deformed body mesh, or None if off-frame.

    Reads the *evaluated* mesh so the armature deformation is baked in — using
    the rest-pose vertices would box the T-pose, not the shot.
    """
    dg = bpy.context.evaluated_depsgraph_get()
    body_eval = body.evaluated_get(dg)
    mesh = body_eval.to_mesh()
    M = body_eval.matrix_world
    xs, ys = [], []
    try:
        for v in mesh.vertices:
            co = world_to_camera_view(scene, cam, M @ v.co)
            if co.z <= 0.0:
                continue  # behind the camera
            xs.append(co.x)
            ys.append(1.0 - co.y)
    finally:
        body_eval.to_mesh_clear()

    if not xs:
        return None
    x0, x1 = max(0.0, min(xs)), min(1.0, max(xs))
    y0, y1 = max(0.0, min(ys)), min(1.0, max(ys))
    if x1 <= x0 or y1 <= y0:
        return None
    return [round((x0 + x1) / 2.0, 6), round((y0 + y1) / 2.0, 6),
            round(x1 - x0, 6), round(y1 - y0, 6)]


def landmark_row(scene, cam, arm, water_z):
    """33 [x, y, z, visibility] landmarks in MediaPipe order.

    z is depth relative to the hip midpoint (negative = closer to the camera),
    scaled by the shoulder→hip torso length so it tracks MediaPipe's units.
    """
    head_fallback = bone_point_world(arm, HEAD_BONE, "tail")
    world = [landmark_world(arm, MP_BONES[name], head_fallback) for name in MP_NAMES]

    # Reference frame for z: hip midpoint depth + a torso-length scale.
    hip_mid = (world[L_HIP] + world[R_HIP]) * 0.5
    sh_mid = (world[L_SHOULDER] + world[R_SHOULDER]) * 0.5
    scale = (sh_mid - hip_mid).length or 1.0
    _, _, hip_depth, _ = project(scene, cam, hip_mid)

    rows = []
    for wco in world:
        x, y, depth, in_front = project(scene, cam, wco)
        v = visibility_score(x, y, in_front, wco.z, water_z)
        z = (depth - hip_depth) / scale  # negative = nearer the camera than the hips
        rows.append([
            round(min(max(x, 0.0), 1.0), 6),
            round(min(max(y, 0.0), 1.0), 6),
            round(z, 6),
            round(v, 4),
        ])
    return rows


def write_meta(root):
    """Schema sidecar so a consumer knows the landmark order and conventions."""
    meta = {
        "format": "capai-mediapipe-pose-synthetic",
        "model": "MediaPipe BlazePose (33 landmarks)",
        "landmark_names": MP_NAMES,
        "landmark_fields": ["x", "y", "z", "visibility"],
        "coords": "x,y normalized to image (top-left origin); "
                  "z = depth relative to hip midpoint (negative = closer)",
        "visibility_tiers": {"tracked": ">=0.75", "partial": "0.30-0.74",
                             "estimated": "<0.30"},
        "angles": ["elbow_l", "elbow_r", "knee_l", "knee_r", "shoulder_tilt"],
        "resolution": [RES_X, RES_Y],
    }
    with open(os.path.join(root, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
def main():
    if SEED is not None:
        random.seed(SEED)

    scene = bpy.context.scene
    arm = require_object(ARMATURE_NAME)
    body = require_object(BODY_MESH_NAME)
    cam = require_object(CAMERA_NAME)
    light = bpy.data.objects.get(LIGHT_NAME) or next(
        (o for o in bpy.data.objects if o.type == "LIGHT"), None)
    water = get_or_make_waterline()

    # Render settings.
    scene.camera = cam
    scene.render.resolution_x = RES_X
    scene.render.resolution_y = RES_Y
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "JPEG"
    scene.render.image_settings.quality = 90
    aspect = RES_X / RES_Y

    img_dir = os.path.join(OUTPUT_DIR, "images")
    lbl_dir = os.path.join(OUTPUT_DIR, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    for i in range(NUM_FRAMES):
        stem = f"frame_{i:05d}"

        randomize_pose(arm)
        bpy.context.view_layer.update()                 # posed bone positions
        water_z = randomize_environment(scene, cam, arm, light, water)
        bpy.context.view_layer.update()                 # camera re-aim + waterline

        # Render the image.
        scene.render.filepath = os.path.join(img_dir, stem + ".jpg")
        bpy.ops.render.render(write_still=True)

        landmarks = landmark_row(scene, cam, arm, water_z)
        record = {
            "index": i,
            "image": f"images/{stem}.jpg",
            "landmarks": landmarks,
            "angles": compute_angles(landmarks, aspect),
            "visibility": vis_tiers([lm[3] for lm in landmarks]),
            "bbox": compute_bbox(scene, cam, body),
        }
        with open(os.path.join(lbl_dir, stem + ".json"), "w") as fh:
            json.dump(record, fh)

        print(f"[{i + 1}/{NUM_FRAMES}] {stem} written")

    write_meta(OUTPUT_DIR)
    print(f"Done. {NUM_FRAMES} frames -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
