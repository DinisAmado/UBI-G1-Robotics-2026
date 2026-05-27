# run_fk.py — corre no ambiente g1_ik
import argparse, json
import numpy as np
from g1_ik import G1_29_ArmIK
import pinocchio as pin

parser = argparse.ArgumentParser()
parser.add_argument("--current_q", type=str, required=True)
args = parser.parse_args()

current_q = np.array(json.loads(args.current_q))

ik = G1_29_ArmIK()
q_pin = current_q[ik._arm_reorder_g1_to_pin]
data = ik.reduced_robot.model.createData()
pin.forwardKinematics(ik.reduced_robot.model, data, q_pin)
pin.updateFramePlacements(ik.reduced_robot.model, data)

L_id = ik.reduced_robot.model.getFrameId("L_ee")
R_id = ik.reduced_robot.model.getFrameId("R_ee")

result = {
    "left":  data.oMf[L_id].homogeneous.tolist(),
    "right": data.oMf[R_id].homogeneous.tolist(),
}
print(json.dumps(result))