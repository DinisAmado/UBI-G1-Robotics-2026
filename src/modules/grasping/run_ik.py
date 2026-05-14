# run_ik.py  — runs inside g1_ik env
import argparse, json, numpy as np
from g1_ik import G1_29_ArmIK

parser = argparse.ArgumentParser()
parser.add_argument("--left",  type=str)
parser.add_argument("--right", type=str)
parser.add_argument("--current_q", type=str, default=None)
args = parser.parse_args()

left_wrist  = np.array(json.loads(args.left))
right_wrist = np.array(json.loads(args.right))
current_q   = np.array(json.loads(args.current_q)) if args.current_q else None

ik_solver = G1_29_ArmIK()
sol_q, sol_tau = ik_solver.solve_ik(left_wrist, right_wrist, current_q)

arm_joint_names = [
    "left_shoulder_pitch_joint",  "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",    "left_elbow_joint",
    "left_wrist_roll_joint",      "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",       "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",  "right_shoulder_yaw_joint",
    "right_elbow_joint",          "right_wrist_roll_joint",
    "right_wrist_pitch_joint",    "right_wrist_yaw_joint",
]
# mapeamento nome → índice G1JointIndex
joint_index_map = {
    "left_shoulder_pitch_joint":  15,
    "left_shoulder_roll_joint":   16,
    "left_shoulder_yaw_joint":    17,
    "left_elbow_joint":           18,
    "left_wrist_roll_joint":      19,
    "left_wrist_pitch_joint":     20,
    "left_wrist_yaw_joint":       21,
    "right_shoulder_pitch_joint": 22,
    "right_shoulder_roll_joint":  23,
    "right_shoulder_yaw_joint":   24,
    "right_elbow_joint":          25,
    "right_wrist_roll_joint":     26,
    "right_wrist_pitch_joint":    27,
    "right_wrist_yaw_joint":      28,
}

result = [
    [joint_index_map[name], float(sol_q[i])]
    for i, name in enumerate(arm_joint_names)
]
print(json.dumps(result))
