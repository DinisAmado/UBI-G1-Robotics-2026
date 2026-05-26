"""

versão 2b: usa estado para controlar as ações
versão 2c: replica posições gravadas com o código
movimentar_joints_ler_valores.py
que ficaram gravadas no file
posições_braços_verter_água.txt


2026-04-20
"""
import time
import sys
import json
import subprocess
from pathlib import Path
import numpy as np
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.idl.default import unitree_go_msg_dds__SportModeState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
import numpy as np
from hand_control import HandControl
#from wav import read_wav, play_pcm_stream
from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient
from PIL import Image
#from Image_now import get_image
from scipy.spatial.transform import Rotation as R



kPi = 3.141592654
kPi_2 = 1.57079632


class MovementConfigs:

    # Posição padrão dos dois braços (joints 15-28)
    ArmStandardPosition = [
        (15,  0.1019), (16,  0.2136), (17,  0.1771), (18,  0.1842),
        (19, -0.0336), (20, -0.0532), (21,  0.0868),
        (22,  0.1035), (23, -0.3205), (24, -0.0719), (25,  0.0792),
        (26, -0.0667), (27,  0.0186), (28, -0.0127),
    ]

    # Braço direito em cima (pose 1 — intermédia)
    RightArmUP1Position = [
        (22, -0.2416), (23, -0.7700), (24, -0.0620),
        (25,  0.2747), (26,  1.1240), (27, -0.2075), (28,  0.9425),
    ]

    # Braço direito em cima (pose 2 — final após grasp / lift)
    RightArmUP2Position = [
        (22, -0.3068), (23, -1.1202), (24, -0.0598),
        (25,  0.2346), (26,  1.4680), (27, -0.2060), (28,  1.2231),
    ]

    # Braço direito estendido para frente (entregar) — alterar conforme calibração
    ArmGivingPosition = [
        (22, -0.8576), (23, -0.101), (24, 0.0354), (25, 0.411), (26, 0.2811), (27, 0.1713), (28, 0.0899)
    ]

    # Braço direito em posição de transporte (carry) — neutro ao lado do corpo
    ArmCarryPosition = [
        (22, -0.2214), (23, -0.0227), (24, -0.2982), (25, 0.04), (26, 0.171), (27, -1.0556), (28, -0.0411)
    ]



class G1JointIndex:
    # Left leg
    LeftHipPitch = 0
    LeftHipRoll = 1
    LeftHipYaw = 2
    LeftKnee = 3
    LeftAnklePitch = 4
    LeftAnkleB = 4
    LeftAnkleRoll = 5
    LeftAnkleA = 5

    # Right leg
    RightHipPitch = 6
    RightHipRoll = 7
    RightHipYaw = 8
    RightKnee = 9
    RightAnklePitch = 10
    RightAnkleB = 10
    RightAnkleRoll = 11
    RightAnkleA = 11

    WaistYaw = 12
    WaistRoll = 13        # NOTE: INVALID for g1 23dof/29dof with waist locked
    WaistA = 13           # NOTE: INVALID for g1 23dof/29dof with waist locked
    WaistPitch = 14       # NOTE: INVALID for g1 23dof/29dof with waist locked
    WaistB = 14           # NOTE: INVALID for g1 23dof/29dof with waist locked

    # Left arm
    LeftShoulderPitch = 15
    LeftShoulderRoll = 16
    LeftShoulderYaw = 17
    LeftElbow = 18
    LeftWristRoll = 19
    LeftWristPitch = 20   # NOTE: INVALID for g1 23dof
    LeftWristYaw = 21     # NOTE: INVALID for g1 23dof

    # Right arm
    RightShoulderPitch = 22
    RightShoulderRoll = 23
    RightShoulderYaw = 24
    RightElbow = 25
    RightWristRoll = 26
    RightWristPitch = 27  # NOTE: INVALID for g1 23dof
    RightWristYaw = 28    # NOTE: INVALID for g1 23dof

    kNotUsedJoint = 29 # NOTE: Weight

class Custom:
    def __init__(self):
        self.time_ = 0.0
        self.control_dt_ = 0.02
        self.counter_ = 0
        self.weight = 0.
        self.weight_rate = 0.2
        self.kp = 100 # 60.
        self.kd = 5 # 1.5
        self.dq = 0.
        self.tau_ff = 0.
        self.mode_machine_ = 0
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = None
        self.first_update_low_state = False
        self.crc = CRC()
        self.done = False
        # 0= inicio, 1=levantou braço, 2=abriu mão, 3=rodou tronco, 4=fechou mão,
        # 5=rodou para posição incial, 6=fim
        self.estado = 0
        # Configuração de ações
        self.action_configs = {}

        self.target_pos = [
            0., kPi_2,  0., kPi_2, 0., 0., 0.,
            0., -kPi_2, 0., kPi_2, 0., 0., 0.,
            0, 0, 0
        ]

        self.arm_joints = [
          G1JointIndex.LeftShoulderPitch,  G1JointIndex.LeftShoulderRoll,
          G1JointIndex.LeftShoulderYaw,    G1JointIndex.LeftElbow,
          G1JointIndex.LeftWristRoll,      G1JointIndex.LeftWristPitch,
          G1JointIndex.LeftWristYaw,
          G1JointIndex.RightShoulderPitch, G1JointIndex.RightShoulderRoll,
          G1JointIndex.RightShoulderYaw,   G1JointIndex.RightElbow,
          G1JointIndex.RightWristRoll,     G1JointIndex.RightWristPitch,
          G1JointIndex.RightWristYaw,
        ]
        self.waist_init = None
        self.right_init = None
        self.left_init = None

    def Init(self):
        # create publisher #
        self.arm_sdk_publisher = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self.arm_sdk_publisher.Init()

        # create subscriber #
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.LowStateHandler, 10)

        # Iniciar controlo das maos -----------------------------------------------------------------------------
        hand_side = "R"
        self.hand_r = HandControl(hand_side)

    def Start(self):
        self.lowCmdWriteThreadPtr = RecurrentThread(
            interval=self.control_dt_, target=self.My_Control, name="control"
        )
        while self.first_update_low_state == False:
            time.sleep(1)

        if self.first_update_low_state == True:
            self.lowCmdWriteThreadPtr.Start()

    def LowStateHandler(self, msg: LowState_):
        self.low_state = msg

        if self.first_update_low_state == False:
            self.first_update_low_state = True


    def _pose_to_SE3(self,action):

        """
        Convert absolute [x, y, z, roll, pitch, yaw] from Octo to a 4×4 SE3 matrix.
        """
        x, y, z = action[:3]
        roll, pitch, yaw = action[3:6]
        se3 = np.eye(4)
        se3[:3, :3] = R.from_euler("xyz", [roll, pitch, yaw]).as_matrix()
        se3[:3, 3] = [x, y, z]
        return se3

    def _camera_to_base(self, cam_xyz, cam_rpy_deg):
        """
        Converte posição E orientação do referencial câmara D435i para referencial base do robot.
        cam_xyz: [x, y, z] no referencial câmara
        cam_rpy_deg: [roll, pitch, yaw] em graus no referencial câmara
        """
        # transformação câmara→base (do URDF)
        cam_pos_in_base = np.array([0.0576235, 0.01753, 0.42987])
        R_cam_to_base = R.from_euler("xyz", [0, 0.8307767239493009, 0]).as_matrix()

        # --- posição ---
        obj_pos_in_base = R_cam_to_base @ np.array(cam_xyz) + cam_pos_in_base

        # --- orientação ---
        R_obj_in_cam = R.from_euler("xyz", cam_rpy_deg, degrees=True).as_matrix()
        R_obj_in_base = R_cam_to_base @ R_obj_in_cam  # composição de rotações
        rpy_in_base = R.from_matrix(R_obj_in_base).as_euler("xyz", degrees=True)

        return obj_pos_in_base, rpy_in_base

    def _solve_ik_run(self, left_wrist: np.ndarray, right_wrist: np.ndarray, current_q=None) -> list:
        cmd = [
            "conda", "run", "-n", "g1_ik",
            "python", "run_ik.py",
            "--left",  json.dumps(left_wrist.tolist()),
            "--right", json.dumps(right_wrist.tolist()),
        ]
        if current_q is not None:
            cmd += ["--current_q", json.dumps(current_q.tolist())]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"IK subprocess failed:\n{result.stderr}")
        # Find the JSON line — it's the one starting with '['
        json_line = None
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("["):
                json_line = line
                break
        if json_line is None:
            raise RuntimeError(f"No JSON found in stdout:\n{result.stdout}\nSTDERR:\n{result.stderr}")
        return [tuple(pair) for pair in json.loads(json_line)]

    def move_joints(self, targets, duration=2.0):
        """
        Smoothly move selected joints to target positions.
        targets: list of (joint_index, target_q)
        """
        if self.low_state is None:
            print("No state available yet.")
            return
        # capture start pose ONCE
        start_q = {
            j: self.low_state.motor_state[j].q
            for j, _ in targets
        } 
        start_time = time.time()
        print("Starting smooth joint motion...")
        while True:
            # real elapsed time
            t = (time.time() - start_time) / duration
            t = np.clip(t, 0.0, 1.0)
            # smoothstep (cubic easing)
            ratio = t * t * (3 - 2 * t)
            for joint, target_q in targets:
                q0 = start_q[joint]
                q = (1 - ratio) * q0 + ratio * target_q
                cmd = self.low_cmd.motor_cmd[joint]
                cmd.tau = 0
                cmd.q = q
                cmd.dq = 0
                cmd.kp = self.kp
                cmd.kd = self.kd
            self.low_cmd.crc = self.crc.Crc(self.low_cmd)
            self.arm_sdk_publisher.Write(self.low_cmd)
            if t >= 1.0:
                break
            time.sleep(self.control_dt_)
        print("Motion complete")

    def fixa_waist(self):
        for idx, j in enumerate([12, 13, 14]):
            self.low_cmd.motor_cmd[j].q  = self.waist_init[idx]
            self.low_cmd.motor_cmd[j].dq = 0.0
            self.low_cmd.motor_cmd[j].kp = 200
            self.low_cmd.motor_cmd[j].kd = 10
            self.low_cmd.motor_cmd[j].tau = 0.0


    def next_state(self):
        self.estado += 1
        print(self.time_)
        print(f"self.estado={self.estado }" )

    def My_Control(self):
        self.time_ += self.control_dt_
        K = 18
        #print(self.time_)


        # ---------------------------------------------------------------------
        # isto só corre da primeira vez
        # gravar pose da waist ANTES de ativar o modo arm_sdk
        if self.waist_init is None:
            self.waist_init = [self.low_state.motor_state[j].q for j in [12,13,14]]
            #print("Captured waist init:", self.waist_init)

            self.left_init = [self.low_state.motor_state[j].q for j in [15,16,17,18,19,20,21]]
            #print("Captured left arm init:", self.left_init)

            self.right_init = [self.low_state.motor_state[j].q for j in [22,23,24,25,26,27,28]]
            #print("Captured right arm init:", self.right_init)

            # aqui está o ponto importante: ativar o arm_sdk
            self.low_cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q =  1 # 1:Enable arm_sdk, 0:Disable arm_sdk

            # manter braço direito rígido na pose inicial
            # copiar braço right
            for idx, j in enumerate([22,23,24,25,26,27,28]):
                  self.low_cmd.motor_cmd[j].q  = self.right_init[idx]
                  self.low_cmd.motor_cmd[j].dq = 0.0
                  self.low_cmd.motor_cmd[j].kp = 200
                  self.low_cmd.motor_cmd[j].kd = 10
                  self.low_cmd.motor_cmd[j].tau = 0.0
            # mostra a pose direito arm inicial
            #for j in [22,23,24,25,26,27,28]:
            #        s = self.low_state.motor_state[j]
            #        print(f"right arm Joint {j}: q={s.q:.4f}, dq={s.dq:.4f}, tau_est={s.tau_est:.4f}")

            # manter braço esquerdo rígido na pose inicial
            # copiar braço esquerdo
            for idx, j in enumerate([15,16,17,18,19,20,21]):
                  self.low_cmd.motor_cmd[j].q  = self.left_init[idx]
                  self.low_cmd.motor_cmd[j].dq = 0.0
                  self.low_cmd.motor_cmd[j].kp = 200
                  self.low_cmd.motor_cmd[j].kd = 10
                  self.low_cmd.motor_cmd[j].tau = 0.0
            # mostra a pose left arm inicial
            #for j in [15,16,17,18,19,20,21]:
            #        s = self.low_state.motor_state[j]
            #        print(f"left arm Joint {j}: q={s.q:.4f}, dq={s.dq:.4f}, tau_est={s.tau_est:.4f}")

            # manter as 3 waist joints iguais à posição inicial
            self.fixa_waist()
        # ---------------------------------------------------------------------



        # mostra a pose da waist
        #for j in [12, 13, 14]:
        #        s = self.low_state.motor_state[j]
        #        print(f"waist Joint {j}: q={s.q:.4f}, dq={s.dq:.4f}, tau_est={s.tau_est:.4f}")


        #print(f"self.estado={self.estado }" )

        if self.estado==0:
            # Mover os braços para posição inicial
            Move = MovementConfigs.ArmStandardPosition
            self.move_joints(Move)
            self.next_state()
            time.sleep(2)

        elif self.estado == 1:
            # ─── 1. Pose do objeto (câmara → base) ────────────────────────────
            obj_pos, obj_rpy = self._camera_to_base(
                [0.083, -0.066, 0.468],
                [7.15, -14.5, -25.14]
            )
            self.target_object_pose = self._pose_to_SE3([*obj_pos, *obj_rpy])
            self.T_base_obj = self.target_object_pose

            # ─── 2. Calcula pré-grasp (sobe no Z do mundo) ────────────────────
            T_pregrasp = self.T_base_obj.copy()
            T_pregrasp[:3, 3] += np.array([0.0, 0.0, 0.15])  # 15cm acima

            # ─── 3. IK para pré-grasp ─────────────────────────────────────────
            self.left_wrist = np.eye(4)
            q_current = [self.low_state.motor_state[j].q for j in range(15, 29)]
            pregrasp_joints = self._solve_ik_run(
                self.left_wrist,
                T_pregrasp,
                current_q=np.array(q_current)
            )
            right_pregrasp = pregrasp_joints[7:]
            self.pregrasp = right_pregrasp
            sla = input("Clica enter se ta tudo bem, agora vem o pregrasp prepara para o desastre")
            self.next_state()

        elif self.estado == 2:
            print('AQUI VAI!!!')
            self.hand_r.grip(0.1)  # abre mão

            # primeiro move para o pregrasp
            self.move_joints(self.pregrasp, duration=6.0)
            time.sleep(1)

            # só depois vira a palma para baixo
            self.move_joints([
                (G1JointIndex.RightWristRoll,  0.0),
                (G1JointIndex.RightWristPitch, -1.57),
                (G1JointIndex.RightWristYaw,   0.0),
            ], duration=1.0)

            sla = input('Clica enter para ver se ta tudo bem')
            self.next_state()
        
        elif self.estado == 3:
            print('Calma agora matemática brinca')
            q_at_pregrasp = [self.low_state.motor_state[j].q for j in range(15, 29)]

            T_grasp = self.T_base_obj.copy()
            T_grasp[:3, 3] += np.array([0.0, 0.0, 0.06])

            grasp_joints = self._solve_ik_run(
                self.left_wrist,
                T_grasp,
                current_q=np.array(q_at_pregrasp)
            )
            right_grasp = grasp_joints[7:]  # [(22, q), (23, q), ...]

            # força o pulso para palma para baixo, ignorando o que o IK calculou
            wrist_override = {
                26: 0.0,    # RightWristRoll  — ajusta conforme calibração
                27: -1.57,  # RightWristPitch
                28: 0.0,    # RightWristYaw
            }
            self.grasp = [
                (idx, wrist_override[idx]) if idx in wrist_override else (idx, q)
                for idx, q in right_grasp
            ]

            sla = input('Clica enter para ver se ta tudo bem, agora vai o grasp')
            self.next_state()

        elif  self.estado==4:
            print('AQUI VAI!!! AGAINNNNN!!!')
            # ─── 6. Desce devagar para o objeto ────────────────────────────
            self.move_joints(self.grasp, duration=6.0)  # mais lento
            time.sleep(0.5)

            # ─── 7. Fecha mão ──────────────────────────────────────────────
            self.hand_r.grip(0.6)
            time.sleep(0.8)  # deixa estabilizar
            sla = input('Clica enter para ver se ta tudo bem,')
            self.next_state()

        elif self.estado==5:
            Move = MovementConfigs.RightArmUP2Position
            self.move_joints(Move)
            sla = input('Clica enter para ver se ta tudo bem,')
            self.next_state()

        elif self.estado==6:
            Move2 = MovementConfigs.ArmCarryPosition
            self.move_joints(Move2)
            sla = input('Clica enter para ver se ta tudo bem,')
            self.next_state()

        elif self.estado==7:
            Move3 = MovementConfigs.ArmGivingPosition
            self.move_joints(Move3)
            sla = input('Clica enter para ver se ta tudo bem,')
            self.next_state()

        elif self.estado==8:
            # Levantar o braço
            
            Move4 = MovementConfigs.ArmStandardPosition
            self.move_joints(Move4)
            self.hand_r.stop()
            sla = input('Clica enter para ver se ta tudo bem,')
            self.next_state()
            time.sleep(2)

            print("Done!!")



        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.arm_sdk_publisher.Write(self.low_cmd)


if __name__ == '__main__':

    print("WARNING: Please ensure there are no obstacles around the robot while running this example.")
    input("Press Enter to continue...")

    ChannelFactoryInitialize(0, "enp117s0")

    custom = Custom()
    custom.Init()
    custom.Start()


    while True:
        time.sleep(1)
        if custom.done:
           print("Done!")
           sys.exit(-1)


'''
Captured waist init: [0.0016873788554221392, -0.00022736095706932247, -0.0013243765570223331]
Captured left arm init: [0.2964298129081726, 0.21361801028251648, -0.026976490393280983, 0.9903164505958557, 0.12208329886198044, 0.05655355751514435, 0.002672482281923294]
Captured right arm init: [0.29491978883743286, -0.21790838241577148, 0.01973801851272583, 0.9893936514854431, -0.14285196363925934, 0.056589510291814804, 0.004877579864114523]
waist Joint 12: q=0.0017, dq=-0.0034, tau_est=-0.2797
waist Joint 13: q=-0.0002, dq=-0.0029, tau_est=0.0000
waist Joint 14: q=-0.0013, dq=-0.0043, tau_est=1.0847
right arm Joint 22: q=0.2949, dq=0.0000, tau_est=0.5625
right arm Joint 23: q=-0.2179, dq=0.0123, tau_est=-2.5000
right arm Joint 24: q=0.0197, dq=0.0000, tau_est=-0.2500
right arm Joint 25: q=0.9894, dq=0.0077, tau_est=-1.0000
right arm Joint 26: q=-0.1429, dq=0.0031, tau_est=-0.1875
right arm Joint 27: q=0.0566, dq=0.0061, tau_est=-0.1937
right arm Joint 28: q=0.0049, dq=0.0000, tau_est=0.0188
left arm Joint 15: q=0.2964, dq=0.0077, tau_est=0.1875
left arm Joint 16: q=0.2136, dq=0.0000, tau_est=2.5625
left arm Joint 17: q=-0.0270, dq=0.0015, tau_est=0.6250
left arm Joint 18: q=0.9903, dq=0.0031, tau_est=-1.3750
left arm Joint 19: q=0.1221, dq=-0.0077, tau_est=0.0000
left arm Joint 20: q=0.0566, dq=0.0000, tau_est=-0.2562
left arm Joint 21: q=0.0027, dq=0.0000, tau_est=-0.0312

'''
