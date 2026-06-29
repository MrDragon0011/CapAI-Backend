"""Synthetic water-polo dataset generator for YOLO26-Pose (Blender).

Renders a rigged athlete in randomized shooting / eggbeater poses under
randomized cameras, lighting, and kit colors, then writes a paired
`.jpg` + YOLO-pose `.txt` for every frame (17 COCO keypoints with
waterline-aware visibility flags).

───────────────────────────────────────────────────────────────
HOW TO RUN
───────────────────────────────────────────────────────────────
Inside Blender (GUI):
  1. Open the .blend that contains your rigged athlete: an armature whose
     deform bones follow the standard Rigify metarig naming (upper_arm.L,
     forearm.L, thigh.L, shin.L, foot.L, eye.L, ear.L, nose, ...), the body
     mesh, a camera, and at least one light.
  2. Edit the CONFIG block below so the names + OUTPUT_DIR match your scene.
  3. Open the 'Scripting' workspace tab, then Text ▸ Open this file
     (or paste it into a new text block).
  4. Press 'Run Script' (▶). Frames land in OUTPUT_DIR/images and
     OUTPUT_DIR/labels, plus a ready-to-train data.yaml.

Headless (no UI, much faster for big batches):
  blender yourscene.blend --background --python blender_pose_synth.py

The script only reads/writes data — it never saves your .blend, so the
randomized pose left in the scene afterwards is harmless.
"""

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

OUTPUT_DIR = "/tmp/capai_synth"    # dataset root (images/ + labels/ created inside)
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
# COCO-17 keypoint -> (deform bone, which end). Standard Rigify metarig names.
# COCO order is fixed; YOLO-pose reads them positionally.
# ═══════════════════════════════════════════════════════════════
KP_BONES = [
    ("nose",          "nose",        "head"),
    ("left_eye",      "eye.L",       "head"),
    ("right_eye",     "eye.R",       "head"),
    ("left_ear",      "ear.L",       "head"),
    ("right_ear",     "ear.R",       "head"),
    ("left_shoulder", "upper_arm.L", "head"),
    ("right_shoulder","upper_arm.R", "head"),
    ("left_elbow",    "forearm.L",   "head"),
    ("right_elbow",   "forearm.R",   "head"),
    ("left_wrist",    "hand.L",      "head"),
    ("right_wrist",   "hand.R",      "head"),
    ("left_hip",      "thigh.L",     "head"),
    ("right_hip",     "thigh.R",     "head"),
    ("left_knee",     "shin.L",      "head"),
    ("right_knee",    "shin.R",      "head"),
    ("left_ankle",    "foot.L",      "head"),
    ("right_ankle",   "foot.R",      "head"),
]
HEAD_BONE = "spine.006"  # fallback for face points if eye./ear./nose bones are absent

# Left/right keypoint swap for YOLO's horizontal-flip augmentation.
FLIP_IDX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


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
    """Project a world point to (x, y) image-normalized coords + in-front flag.

    world_to_camera_view returns (0,0) at the bottom-left of frame; YOLO/images
    use a top-left origin, so y is flipped.
    """
    co = world_to_camera_view(scene, cam, world_co)
    return co.x, 1.0 - co.y, co.z > 0.0


def visibility_flag(x, y, in_front, world_z, water_z):
    """0 off-frame, 1 occluded (below waterline), 2 visible (above waterline)."""
    if not in_front or not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        return 0
    return 2 if world_z > water_z else 1


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
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0, x1 - x0, y1 - y0)


def keypoint_row(scene, cam, arm, water_z):
    """17 (x, y, v) triples in COCO order, flattened for the YOLO row."""
    head_fallback = bone_point_world(arm, HEAD_BONE, "tail")  # top of skull
    out = []
    for _name, bone, end in KP_BONES:
        wco = bone_point_world(arm, bone, end)
        if wco is None:
            wco = head_fallback  # face point on a rig without eye./ear./nose bones
        if wco is None:
            out.extend([0.0, 0.0, 0])
            continue
        x, y, in_front = project(scene, cam, wco)
        v = visibility_flag(x, y, in_front, wco.z, water_z)
        if v == 0:
            out.extend([0.0, 0.0, 0])
        else:
            out.extend([min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0), v])
    return out


def write_data_yaml(root):
    """Minimal Ultralytics pose data.yaml so the set trains as-is."""
    flip = ", ".join(str(i) for i in FLIP_IDX)
    text = (
        f"path: {root}\n"
        "train: images\n"
        "val: images\n"
        "kpt_shape: [17, 3]\n"
        f"flip_idx: [{flip}]\n"
        "names:\n"
        "  0: athlete\n"
    )
    with open(os.path.join(root, "data.yaml"), "w") as fh:
        fh.write(text)


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

        # Label: skip frames where the athlete left the frame entirely.
        bbox = compute_bbox(scene, cam, body)
        if bbox is None:
            print(f"[{i + 1}/{NUM_FRAMES}] {stem}: athlete off-frame, skipped label")
            continue
        kps = keypoint_row(scene, cam, arm, water_z)

        row = ["0"] + [f"{c:.6f}" for c in bbox]
        for j in range(0, len(kps), 3):
            row += [f"{kps[j]:.6f}", f"{kps[j + 1]:.6f}", str(int(kps[j + 2]))]
        with open(os.path.join(lbl_dir, stem + ".txt"), "w") as fh:
            fh.write(" ".join(row) + "\n")

        print(f"[{i + 1}/{NUM_FRAMES}] {stem} written")

    write_data_yaml(OUTPUT_DIR)
    print(f"Done. {NUM_FRAMES} frames -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
