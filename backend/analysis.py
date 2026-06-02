import json
import math
import sys
from pathlib import Path

import numpy as np

from classifier import (
    POSE_KEYPOINTS,
    _angle,
    _lm_xyz,
    load_sequence,
)

OUTPUT_PATH = Path(__file__).parent / "analysis.json"
SEQUENCE_PATH = Path(__file__).parent / "landmarks_sequence.json"

ELITE_STANDARDS = {
    "shooting": {
        "label": "Power Shot",
        "metrics": {
            "elbow_angle_at_release": {
                "elite_min": math.radians(155),
                "elite_max": math.radians(180),
                "description": "Throwing elbow extension at release",
                "unit": "degrees",
            },
            "shoulder_abduction": {
                "elite_min": math.radians(85),
                "elite_max": math.radians(100),
                "description": "Shooting shoulder abduction angle",
                "unit": "degrees",
            },
            "torso_rotation_range": {
                "elite_min": math.radians(30),
                "elite_max": math.radians(60),
                "description": "Torso axial rotation range across shot motion",
                "unit": "degrees",
            },
            "wrist_height_above_shoulder": {
                "elite_min": -0.05,
                "elite_max": 0.20,
                "description": "Shooting wrist y-offset above ipsilateral shoulder (normalised)",
                "unit": "normalised",
            },
            "hip_drive_velocity": {
                "elite_min": 0.008,
                "elite_max": 1.0,
                "description": "Mean frame-to-frame hip displacement during wind-up",
                "unit": "normalised/frame",
            },
        },
        "tips": {
            "elbow_angle_at_release": {
                "low": "Your elbow is not fully extending through release. Drive your forearm forward and snap the wrist to maximise ball velocity.",
                "high": "Elbow is hyperextending — ensure your arm is relaxed through follow-through to protect the joint.",
                "good": "Elbow extension is within elite range. Focus on maintaining this consistently under fatigue.",
            },
            "shoulder_abduction": {
                "low": "Your throwing arm is too close to your body. Raise the elbow to shoulder height before initiating the throw for optimal leverage.",
                "high": "Excessive shoulder abduction reduces power transfer. Keep the elbow at 90° to the torso during the cocking phase.",
                "good": "Shoulder position is ideal. Pair this with trunk rotation to increase shot power.",
            },
            "torso_rotation_range": {
                "low": "Insufficient trunk rotation is limiting your shot power. Initiate the throw by rotating your hips first, then let the torso unwind into the arm.",
                "high": "Over-rotation may cause accuracy loss. Focus on controlled, targeted trunk rotation toward the goal.",
                "good": "Torso rotation is in the elite range. Ensure hip-to-shoulder sequencing for maximum kinetic chain efficiency.",
            },
            "wrist_height_above_shoulder": {
                "low": "Release point is too low. A higher release angle increases shot arc and makes it harder for the goalkeeper to read.",
                "high": "Release point is too high, reducing power direction. Aim for the wrist to be just above shoulder level at release.",
                "good": "Release height is optimal for a water polo power shot.",
            },
            "hip_drive_velocity": {
                "low": "Hips are not generating enough drive. Explode your hips toward the goal during the wind-up to load the kinetic chain.",
                "high": "Hip motion is within range.",
                "good": "Hip drive velocity is strong. Coordinate this with shoulder rotation for peak power.",
            },
        },
    },

    "passing": {
        "label": "Pass",
        "metrics": {
            "elbow_angle_at_release": {
                "elite_min": math.radians(140),
                "elite_max": math.radians(175),
                "description": "Throwing elbow extension at release",
                "unit": "degrees",
            },
            "wrist_separation_at_catch": {
                "elite_min": 0.25,
                "elite_max": 0.70,
                "description": "Normalised distance between wrists at pass initiation",
                "unit": "normalised",
            },
            "torso_lean": {
                "elite_min": -0.30,
                "elite_max": 0.30,
                "description": "Mean lateral torso lean during pass",
                "unit": "radians",
            },
            "follow_through_wrist_drop": {
                "elite_min": 0.02,
                "elite_max": 1.0,
                "description": "Wrist y-displacement downward after release (follow-through)",
                "unit": "normalised",
            },
        },
        "tips": {
            "elbow_angle_at_release": {
                "low": "Elbow is not extending fully. Snap the arm through for a crisp, accurate pass.",
                "high": "Arm is overextending — keep a slight bend to maintain control and reduce injury risk.",
                "good": "Good arm extension. Vary your release speed to disguise passes from defenders.",
            },
            "wrist_separation_at_catch": {
                "low": "Hands are too close together at initiation. Widen your base to improve passing options and disguise direction.",
                "high": "Wrists are too far apart, telegraphing the pass. Bring them closer for a quicker, less readable release.",
                "good": "Hand position at initiation is correct. Use eye fakes to complement your passing mechanics.",
            },
            "torso_lean": {
                "low": "Minimal torso lean detected — you may be passing flat. A slight lean into the pass direction adds power and disguise.",
                "high": "Excessive lateral lean reduces balance and pass accuracy. Stay tall in the water.",
                "good": "Torso position during the pass is balanced and efficient.",
            },
            "follow_through_wrist_drop": {
                "low": "Incomplete follow-through detected. Let the wrist drop naturally after release for accuracy and arm health.",
                "high": "Follow-through looks good.",
                "good": "Follow-through is within elite range. Consistent follow-through builds muscle memory for accuracy.",
            },
        },
    },

    "swimming": {
        "label": "Swimming Stroke",
        "metrics": {
            "stroke_symmetry": {
                "elite_min": 0.85,
                "elite_max": 1.0,
                "description": "Left/right wrist velocity symmetry ratio",
                "unit": "ratio",
            },
            "elbow_high_catch_r": {
                "elite_min": math.radians(100),
                "elite_max": math.radians(140),
                "description": "Right elbow angle at catch phase (high-elbow pull)",
                "unit": "degrees",
            },
            "elbow_high_catch_l": {
                "elite_min": math.radians(100),
                "elite_max": math.radians(140),
                "description": "Left elbow angle at catch phase (high-elbow pull)",
                "unit": "degrees",
            },
            "hip_roll_range": {
                "elite_min": math.radians(25),
                "elite_max": math.radians(50),
                "description": "Hip axial roll range per stroke cycle",
                "unit": "degrees",
            },
            "head_position": {
                "elite_min": -0.10,
                "elite_max": 0.05,
                "description": "Nose y-offset relative to mid-shoulder (head position)",
                "unit": "normalised",
            },
        },
        "tips": {
            "stroke_symmetry": {
                "low": "Stroke asymmetry detected — one arm is doing more work. Focus on matching entry angle and pull depth on both sides.",
                "high": "Symmetry is within range.",
                "good": "Bilateral stroke symmetry is excellent. Maintain this under sprint conditions.",
            },
            "elbow_high_catch_r": {
                "low": "Right elbow is dropping at the catch. Keep the elbow high and fingertips down to maximise water grip.",
                "high": "Right elbow angle is too wide at catch — bring it in to improve pull efficiency.",
                "good": "Right arm catch position is in the elite range. Focus on early vertical forearm entry.",
            },
            "elbow_high_catch_l": {
                "low": "Left elbow is dropping at the catch. Drill single-arm catch focus sets with a pull buoy.",
                "high": "Left elbow angle is too wide — tighten the catch position.",
                "good": "Left arm catch is strong. Match this consistency to your right side.",
            },
            "hip_roll_range": {
                "low": "Insufficient hip roll is reducing stroke length and power. Let the hips rotate fully with each arm entry.",
                "high": "Excessive hip roll is creating drag. Moderate your rotation to stay streamlined.",
                "good": "Hip roll range is optimal. Coordinate rotation timing with the catch for maximum propulsion.",
            },
            "head_position": {
                "low": "Head is too high — this lifts the hips and creates drag. Keep the waterline at mid-forehead.",
                "high": "Head is too low, over-rotating the neck. Look slightly forward-down to maintain a neutral spine.",
                "good": "Head position is neutral and hydrodynamic.",
            },
        },
    },

    "goalie": {
        "label": "Goalkeeper Technique",
        "metrics": {
            "eggbeater_knee_range": {
                "elite_min": math.radians(70),
                "elite_max": math.radians(110),
                "description": "Mean knee flexion angle during eggbeater kick",
                "unit": "degrees",
            },
            "body_height_stability": {
                "elite_min": 0.0,
                "elite_max": 0.04,
                "description": "Std dev of nose y-coordinate (vertical stability in water)",
                "unit": "normalised",
            },
            "arm_width_at_block": {
                "elite_min": 0.35,
                "elite_max": 0.80,
                "description": "Wrist separation during block stance",
                "unit": "normalised",
            },
            "lateral_reach": {
                "elite_min": 0.30,
                "elite_max": 1.0,
                "description": "Max wrist lateral displacement from body midline",
                "unit": "normalised",
            },
            "shoulder_symmetry": {
                "elite_min": 0.80,
                "elite_max": 1.0,
                "description": "Left/right shoulder height symmetry ratio",
                "unit": "ratio",
            },
        },
        "tips": {
            "eggbeater_knee_range": {
                "low": "Knee flexion is insufficient for effective eggbeater kick. Aim for ~90° knee bend to generate maximum lift force.",
                "high": "Over-bending at the knee reduces kick efficiency. Aim for a 70–110° range.",
                "good": "Knee angle is in the optimal range for eggbeater lift. Work on kick cadence to improve explosive height.",
            },
            "body_height_stability": {
                "low": "Body height is within range.",
                "high": "Significant vertical bobbing detected. Increase eggbeater kick frequency and synchronise arm position to reduce oscillation.",
                "good": "Excellent vertical stability in the water. This is a key marker for elite goalkeeping.",
            },
            "arm_width_at_block": {
                "low": "Arms are too narrow during block stance. Spread your arms wider to cover more of the goal frame.",
                "high": "Arms are over-extended, reducing explosive reach speed. Bring them to a ready position closer to mid-width.",
                "good": "Arm width in block stance is optimal. Focus on reading the shooter's shoulder for early reaction.",
            },
            "lateral_reach": {
                "low": "Lateral reach is limited. Practise explosive side-lunge drills from eggbeater to improve corner coverage.",
                "high": "Lateral reach looks good.",
                "good": "Lateral reach is in the elite range. Combine with body rotation for full-extension blocks.",
            },
            "shoulder_symmetry": {
                "low": "Shoulder asymmetry detected — one side is dropping. Keep both shoulders level to maintain equal reaction speed to either post.",
                "high": "Symmetry is within range.",
                "good": "Shoulder symmetry is excellent. Maintain this ready position between shots.",
            },
        },
    },
}


def _compute_metrics(frames, action):
    pose_kp = POSE_KEYPOINTS

    def kp(frame, name):
        return _lm_xyz(frame.get("pose_landmarks", []), pose_kp[name])

    def kp_w(frame, name):
        return _lm_xyz(frame.get("pose_world_landmarks", []), pose_kp[name])

    results = {}

    if action == "shooting":
        elbow_angles = [
            _angle(kp(f, "right_shoulder"), kp(f, "right_elbow"), kp(f, "right_wrist"))
            for f in frames
        ]
        results["elbow_angle_at_release"] = float(np.max(elbow_angles))

        shoulder_angles = [
            _angle(kp(f, "right_hip"), kp(f, "right_shoulder"), kp(f, "right_elbow"))
            for f in frames
        ]
        results["shoulder_abduction"] = float(np.mean(shoulder_angles))

        torso_angles = []
        for f in frames:
            ls = kp(f, "left_shoulder")
            rs = kp(f, "right_shoulder")
            lh = kp(f, "left_hip")
            rh = kp(f, "right_hip")
            shoulder_vec = rs - ls
            hip_vec = rh - lh
            cross = float(np.cross(shoulder_vec[:2], hip_vec[:2]))
            torso_angles.append(math.atan2(abs(cross), max(np.dot(shoulder_vec[:2], hip_vec[:2]), 1e-8)))
        results["torso_rotation_range"] = float(np.max(torso_angles) - np.min(torso_angles))

        wrist_above_shoulder = [
            kp(f, "right_shoulder")[1] - kp(f, "right_wrist")[1]
            for f in frames
        ]
        results["wrist_height_above_shoulder"] = float(np.max(wrist_above_shoulder))

        hip_positions = [((kp(f, "left_hip") + kp(f, "right_hip")) / 2) for f in frames]
        hip_deltas = [float(np.linalg.norm(hip_positions[i+1] - hip_positions[i])) for i in range(len(hip_positions)-1)]
        results["hip_drive_velocity"] = float(np.mean(hip_deltas)) if hip_deltas else 0.0

    elif action == "passing":
        elbow_angles = [
            _angle(kp(f, "right_shoulder"), kp(f, "right_elbow"), kp(f, "right_wrist"))
            for f in frames
        ]
        results["elbow_angle_at_release"] = float(np.max(elbow_angles))

        wrist_seps = [
            float(np.linalg.norm(kp(f, "right_wrist") - kp(f, "left_wrist")))
            for f in frames
        ]
        results["wrist_separation_at_catch"] = float(np.mean(wrist_seps))

        torso_leans = []
        for f in frames:
            ms = (kp(f, "left_shoulder") + kp(f, "right_shoulder")) / 2
            mh = (kp(f, "left_hip") + kp(f, "right_hip")) / 2
            vec = ms - mh
            torso_leans.append(math.atan2(vec[0], vec[1] + 1e-8))
        results["torso_lean"] = float(np.mean(torso_leans))

        rw_y = [kp(f, "right_wrist")[1] for f in frames]
        follow = max(0.0, float(np.max(rw_y) - rw_y[0]))
        results["follow_through_wrist_drop"] = follow

    elif action == "swimming":
        rw_pos = [kp(f, "right_wrist") for f in frames]
        lw_pos = [kp(f, "left_wrist") for f in frames]
        rw_vel = [float(np.linalg.norm(rw_pos[i+1] - rw_pos[i])) for i in range(len(rw_pos)-1)]
        lw_vel = [float(np.linalg.norm(lw_pos[i+1] - lw_pos[i])) for i in range(len(lw_pos)-1)]
        mean_r = float(np.mean(rw_vel)) if rw_vel else 1e-8
        mean_l = float(np.mean(lw_vel)) if lw_vel else 1e-8
        results["stroke_symmetry"] = float(min(mean_r, mean_l) / (max(mean_r, mean_l) + 1e-8))

        r_catches = [
            _angle(kp(f, "right_shoulder"), kp(f, "right_elbow"), kp(f, "right_wrist"))
            for f in frames
        ]
        l_catches = [
            _angle(kp(f, "left_shoulder"), kp(f, "left_elbow"), kp(f, "left_wrist"))
            for f in frames
        ]
        results["elbow_high_catch_r"] = float(np.mean(r_catches))
        results["elbow_high_catch_l"] = float(np.mean(l_catches))

        hip_rolls = []
        for f in frames:
            lh = kp(f, "left_hip")
            rh = kp(f, "right_hip")
            hip_vec = rh - lh
            hip_rolls.append(math.atan2(hip_vec[2], hip_vec[0] + 1e-8))
        results["hip_roll_range"] = float(np.max(hip_rolls) - np.min(hip_rolls))

        nose_y = [kp(f, "nose")[1] for f in frames]
        shoulder_y = [
            ((kp(f, "left_shoulder")[1] + kp(f, "right_shoulder")[1]) / 2)
            for f in frames
        ]
        results["head_position"] = float(np.mean([n - s for n, s in zip(nose_y, shoulder_y)]))

    elif action == "goalie":
        knee_angles = [
            (_angle(kp(f, "right_hip"), kp(f, "right_knee"), kp(f, "right_ankle")) +
             _angle(kp(f, "left_hip"), kp(f, "left_knee"), kp(f, "left_ankle"))) / 2
            for f in frames
        ]
        results["eggbeater_knee_range"] = float(np.mean(knee_angles))

        nose_y = [kp(f, "nose")[1] for f in frames]
        results["body_height_stability"] = float(np.std(nose_y))

        wrist_seps = [
            float(np.linalg.norm(kp(f, "right_wrist") - kp(f, "left_wrist")))
            for f in frames
        ]
        results["arm_width_at_block"] = float(np.mean(wrist_seps))

        mid_x = [
            ((kp(f, "left_shoulder")[0] + kp(f, "right_shoulder")[0]) / 2)
            for f in frames
        ]
        rw_x = [kp(f, "right_wrist")[0] for f in frames]
        lw_x = [kp(f, "left_wrist")[0] for f in frames]
        r_reach = [abs(rw_x[i] - mid_x[i]) for i in range(len(frames))]
        l_reach = [abs(lw_x[i] - mid_x[i]) for i in range(len(frames))]
        results["lateral_reach"] = float(max(np.max(r_reach), np.max(l_reach)))

        ls_y = [kp(f, "left_shoulder")[1] for f in frames]
        rs_y = [kp(f, "right_shoulder")[1] for f in frames]
        sym = [min(ls_y[i], rs_y[i]) / (max(ls_y[i], rs_y[i]) + 1e-8) for i in range(len(frames))]
        results["shoulder_symmetry"] = float(np.mean(sym))

    return results


def _evaluate_metric(value, standards, metric_key):
    spec = standards["metrics"][metric_key]
    tips = standards["tips"][metric_key]
    elite_min = spec["elite_min"]
    elite_max = spec["elite_max"]

    if value < elite_min:
        status = "below_elite"
        feedback = tips["low"]
    elif value > elite_max:
        status = "above_elite"
        feedback = tips["high"]
    else:
        status = "elite"
        feedback = tips["good"]

    display_value = math.degrees(value) if spec["unit"] == "degrees" else value

    return {
        "metric": metric_key,
        "description": spec["description"],
        "value": round(display_value, 4),
        "unit": spec["unit"],
        "elite_min": round(math.degrees(elite_min) if spec["unit"] == "degrees" else elite_min, 4),
        "elite_max": round(math.degrees(elite_max) if spec["unit"] == "degrees" else elite_max, 4),
        "status": status,
        "feedback": feedback,
    }


def _overall_score(evaluations):
    elite_count = sum(1 for e in evaluations if e["status"] == "elite")
    return round(elite_count / len(evaluations) * 100, 1) if evaluations else 0.0


def _priority_focus(evaluations):
    non_elite = [e for e in evaluations if e["status"] != "elite"]
    if not non_elite:
        return "All metrics are within elite range. Focus on consistency and performance under fatigue."
    primary = non_elite[0]
    return f"Priority: {primary['description']} — {primary['feedback']}"


def analyse(action: str, sequence_path: str = None, output_path: str = None) -> dict:
    if action not in ELITE_STANDARDS:
        raise ValueError(f"Unknown action '{action}'. Valid actions: {list(ELITE_STANDARDS.keys())}")

    frames = load_sequence(sequence_path)
    standards = ELITE_STANDARDS[action]
    raw_metrics = _compute_metrics(frames, action)

    evaluations = [
        _evaluate_metric(raw_metrics[key], standards, key)
        for key in standards["metrics"]
        if key in raw_metrics
    ]

    score = _overall_score(evaluations)
    priority = _priority_focus(evaluations)

    output = {
        "action": action,
        "label": standards["label"],
        "overall_elite_score_pct": score,
        "priority_focus": priority,
        "total_frames_analysed": len(frames),
        "metrics": evaluations,
    }

    dest = Path(output_path) if output_path else OUTPUT_PATH
    with open(dest, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Analysis saved to {dest}")
    return output


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analysis.py <action> [sequence.json] [output.json]")
        print("Actions: shooting, passing, swimming, goalie")
        sys.exit(1)

    action_arg = sys.argv[1]
    seq_arg = sys.argv[2] if len(sys.argv) > 2 else None
    out_arg = sys.argv[3] if len(sys.argv) > 3 else None
    result = analyse(action_arg, seq_arg, out_arg)
    print(json.dumps(result, indent=2))
