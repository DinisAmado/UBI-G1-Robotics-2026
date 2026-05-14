"""
grasp_control.py
================
Integração completa: CycloneDDS + controlo de braços G1

Comandos suportados (via tópico rt/grasp/command):
  - postura=EXTEND_ARM_FORWARD, objeto_id=''        → "apanhar objeto"  (estados 0→1→2)
  - postura=EXTEND_ARM_FORWARD, objeto_id='deliver' → "entregar objeto" (estado 3)
  - postura=NEUTRAL                                  → "largar objeto"   (estado 4)

Sequência de estados:
  IDLE (−1)
    │
    ▼  EXTEND_ARM_FORWARD + objeto_id==''
  INIT_POSITION (0)  → move para ArmStandardPosition + RightArmUP2Position
    │
    ▼
  GRASPING (1)       → IK pré-grasp → desce → fecha mão
    │
    ▼
  LIFT (2)           → levanta braço para RightArmUP2Position   [apanhar concluído]
    │
    ▼  EXTEND_ARM_FORWARD + objeto_id=='deliver'
  DELIVERING (3)     → estica braço para ArmGivingPosition
    │
    ▼  NEUTRAL
  RELEASING (4)      → abre mão → retrai braço para posição inicial

2026-04-20 / versão 3 — integração completa
"""

import time
import sys
import json
import logging
import subprocess
import threading
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

# ── Unitree SDK ───────────────────────────────────────────────────────────────
from unitree_sdk2py.core.channel import (
    ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
)
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient

# ── Hand control ──────────────────────────────────────────────────────────────
from hand_control import HandControl

# ── CycloneDDS ───────────────────────────────────────────────────────────────
from cyclonedds.domain import DomainParticipant
from cyclonedds.topic import Topic
from cyclonedds.pub import Publisher, DataWriter
from cyclonedds.sub import Subscriber, DataReader

from qos_profiles import QOS_GRASP
from idl_ri import (
    Header,
    Status,
    GraspCommand,
    GraspStatusMsg,
    Posture,
    Image, #alterar para pose
)

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
log = logging.getLogger('grasp_control')

# ─────────────────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────────────────
kPi   = 3.141592654
kPi_2 = 1.57079632

DOMAIN_ID = 0

# Estados internos do robô
STATE_IDLE         = -1
STATE_INIT_POS     =  0   # mover para posição inicial + braço cima
STATE_GRASPING     =  1   # IK pré-grasp → desce → fecha mão
STATE_LIFT         =  2   # levanta braço (apanhar concluído)
STATE_DELIVERING   =  3   # estica braço para frente (entregar)
STATE_RELEASING    =  4   # abre mão + retrai braço (largar)


# ─────────────────────────────────────────────────────────────────────────────
# Configurações de movimento
# ─────────────────────────────────────────────────────────────────────────────
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

    # Braço direito em cima (pose 2 — final após grasp)
    RightArmUP2Position = [
        (22, -0.3068), (23, -1.1202), (24, -0.0598),
        (25,  0.2346), (26,  1.4680), (27, -0.2060), (28,  1.2231),
    ]

    # Braço direito estendido para frente (entregar) alterar
    ArmGivingPosition = [
        (22, -0.5741), (23, -0.1391), (24, -0.2563),
        (25,  0.5985), (26, -0.0296), (27, -0.1613), (28,  0.1152),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Mapeamento de joints
# ─────────────────────────────────────────────────────────────────────────────
class G1JointIndex:
    LeftHipPitch    =  0;  LeftHipRoll     =  1;  LeftHipYaw      =  2
    LeftKnee        =  3;  LeftAnklePitch  =  4;  LeftAnkleRoll   =  5

    RightHipPitch   =  6;  RightHipRoll    =  7;  RightHipYaw     =  8
    RightKnee       =  9;  RightAnklePitch = 10;  RightAnkleRoll  = 11

    WaistYaw        = 12;  WaistRoll       = 13;  WaistPitch      = 14

    LeftShoulderPitch  = 15;  LeftShoulderRoll  = 16;  LeftShoulderYaw  = 17
    LeftElbow          = 18;  LeftWristRoll     = 19;  LeftWristPitch   = 20
    LeftWristYaw       = 21

    RightShoulderPitch = 22;  RightShoulderRoll = 23;  RightShoulderYaw = 24
    RightElbow         = 25;  RightWristRoll    = 26;  RightWristPitch  = 27
    RightWristYaw      = 28

    kNotUsedJoint = 29   # activar arm_sdk com q=1


# ─────────────────────────────────────────────────────────────────────────────
# Camada de comunicação CycloneDDS (thread-safe)
# ─────────────────────────────────────────────────────────────────────────────
class GraspComms:
    def __init__(self):
        self._dp  = DomainParticipant(DOMAIN_ID)
        self._pub = Publisher(self._dp)
        self._sub = Subscriber(self._dp)

        # Tópico de comando (já existe)
        self._t_cmd    = Topic(self._dp, 'rt/grasp/command', GraspCommand,   qos=QOS_GRASP)
        self._t_status = Topic(self._dp, 'rt/grasp/status',  GraspStatusMsg, qos=QOS_GRASP)
        self._w_status = DataWriter(self._pub, self._t_status)
        self._r_cmd    = DataReader(self._sub, self._t_cmd)

        # Tópico de pose do objeto  ← NOVO
        self._t_pose = Topic(self._dp, 'rt/object/pose', ObjectPoseMsg, qos=QOS_GRASP)
        self._r_pose = DataReader(self._sub, self._t_pose)

        self._lock        = threading.Lock()
        self._seq         = 0
        self.pending_cmd: GraspCommand    | None = None
        self._latest_pose: ObjectPoseMsg  | None = None   # ← NOVO

        self._poll_thread = threading.Thread(
            target=self._poll_loop, daemon=True, name='dds_poll'
        )
        self._poll_thread.start()

    def _poll_loop(self):
        while True:
            # comando (já existe)
            samples = self._r_cmd.take(1)
            if samples:
                with self._lock:
                    self.pending_cmd = samples[0]

            # pose do objeto  ← NOVO
            pose_samples = self._r_pose.take(1)
            if pose_samples:
                with self._lock:
                    self._latest_pose = pose_samples[0]
                log.info('Pose recebida: %s', pose_samples[0].pose_6dof)

            time.sleep(0.02)

    def get_latest_pose(self) -> ObjectPoseMsg | None:   # ← NOVO
        """Devolve a pose mais recente (thread-safe)."""
        with self._lock:
            return self._latest_pose

    def take_cmd(self) -> GraspCommand | None:
        with self._lock:
            cmd = self.pending_cmd
            self.pending_cmd = None
        return cmd

    def report_status(self, status: Status, reason: str = '', progress: float = 0.0):
        with self._lock:
            self._seq += 1
            seq = self._seq
        self._w_status.write(GraspStatusMsg(
            header=Header(timestamp_ns=time.time_ns(), frame_id='grasp', seq=seq),
            status=status,
            reason=reason,
            progress=progress,
        ))

# ─────────────────────────────────────────────────────────────────────────────
# Controlador principal
# ─────────────────────────────────────────────────────────────────────────────
class Custom:

    def __init__(self):
        self.time_       = 0.0
        self.control_dt_ = 0.02          # 50 Hz
        self.kp          = 100.
        self.kd          = 5.

        self.low_cmd   = unitree_hg_msg_dds__LowCmd_()
        self.low_state = None
        self.first_update_low_state = False
        self.crc  = CRC()
        self.done = False

        # Estado interno
        self.estado: int = STATE_IDLE

        # Pose do objeto gravada quando recebemos o comando de grasp
        self.target_object_pose: np.ndarray | None = None

        # Poses iniciais capturadas no arranque
        self.waist_init = None
        self.left_init  = None
        self.right_init = None

        # Flag: arm_sdk já activado?
        self._arm_sdk_enabled = False

        # Comunicações DDS
        self.comms = GraspComms()

    # ─────────────────────────────────────────────────────────────────────────
    # Inicialização SDK
    # ─────────────────────────────────────────────────────────────────────────
    def Init(self):
        self.arm_sdk_publisher = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self.arm_sdk_publisher.Init()

        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.LowStateHandler, 10)

        self.hand_r = HandControl("R")
        log.info('Custom.Init() concluído')

    def Start(self):
        self.lowCmdWriteThreadPtr = RecurrentThread(
            interval=self.control_dt_, target=self.My_Control, name="control"
        )
        log.info('À espera do primeiro LowState...')
        while not self.first_update_low_state:
            time.sleep(0.1)
        self.lowCmdWriteThreadPtr.Start()
        log.info('Thread de controlo iniciada')

    def LowStateHandler(self, msg: LowState_):
        self.low_state = msg
        if not self.first_update_low_state:
            self.first_update_low_state = True

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers de movimento
    # ─────────────────────────────────────────────────────────────────────────
    def _set_joint(self, j: int, q: float, kp: float = None, kd: float = None):
        cmd = self.low_cmd.motor_cmd[j]
        cmd.q   = q
        cmd.dq  = 0.0
        cmd.tau = 0.0
        cmd.kp  = kp if kp is not None else self.kp
        cmd.kd  = kd if kd is not None else self.kd

    def _fix_waist(self):
        """Mantém a cintura na pose inicial com ganhos altos."""
        for idx, j in enumerate([12, 13, 14]):
            self._set_joint(j, self.waist_init[idx], kp=200, kd=10)

    def _fix_left_arm(self):
        """Mantém o braço esquerdo na pose inicial."""
        for idx, j in enumerate(range(15, 22)):
            self._set_joint(j, self.left_init[idx], kp=200, kd=10)

    def _flush(self):
        """Calcula CRC e publica o low_cmd actual."""
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.arm_sdk_publisher.Write(self.low_cmd)

    def move_joints(self, targets: list, duration: float = 2.0):
        """
        Movimento suave (smoothstep) para os joints indicados.
        targets: [(joint_index, target_q), ...]
        Bloqueia até ao fim da interpolação.
        A cintura e o braço esquerdo são mantidos fixos durante o movimento.
        """
        if self.low_state is None:
            log.warning('move_joints: low_state ainda não disponível')
            return

        start_q  = {j: self.low_state.motor_state[j].q for j, _ in targets}
        t0       = time.time()

        while True:
            elapsed = time.time() - t0
            t       = min(elapsed / duration, 1.0)
            ratio   = t * t * (3.0 - 2.0 * t)       # smoothstep cúbico

            for j, q_target in targets:
                q = (1.0 - ratio) * start_q[j] + ratio * q_target
                self._set_joint(j, q)

            self._fix_waist()
            self._fix_left_arm()
            self._flush()

            if t >= 1.0:
                break
            time.sleep(self.control_dt_)

        log.info('move_joints concluído (duration=%.1fs)', duration)

    # ─────────────────────────────────────────────────────────────────────────
    # Cinemática inversa (sub-processo conda g1_ik)
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _pose_to_SE3(action: list) -> np.ndarray:
        """[x, y, z, roll, pitch, yaw] → 4×4 SE3."""
        se3 = np.eye(4)
        se3[:3, :3] = R.from_euler('xyz', action[3:6]).as_matrix()
        se3[:3,  3] = action[:3]
        return se3

    def _solve_ik_run(
        self,
        left_wrist:  np.ndarray,
        right_wrist: np.ndarray,
        current_q:   np.ndarray | None = None,
    ) -> list:
        cmd = [
            'conda', 'run', '-n', 'g1_ik',
            'python', 'run_ik.py',
            '--left',  json.dumps(left_wrist.tolist()),
            '--right', json.dumps(right_wrist.tolist()),
        ]
        if current_q is not None:
            cmd += ['--current_q', json.dumps(current_q.tolist())]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f'IK falhou:\n{result.stderr}')

        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith('['):
                return [tuple(pair) for pair in json.loads(line)]

        raise RuntimeError(f'Sem JSON no stdout:\n{result.stdout}')

    # ─────────────────────────────────────────────────────────────────────────
    # Sequências de acção
    # ─────────────────────────────────────────────────────────────────────────
    def _do_init_position(self):
        """
        Estado 0 — INIT_POSITION
        Move ambos os braços para a posição padrão e depois
        levanta o braço direito para a pose UP2.
        """
        log.info('[Estado 0] Posição inicial...')
        self.comms.report_status(Status.RUNNING, reason='init_position', progress=0.1)

        self.move_joints(MovementConfigs.ArmStandardPosition, duration=2.0)
        self.move_joints(MovementConfigs.RightArmUP2Position,  duration=2.0)

        log.info('[Estado 0] Concluído → GRASPING')
        self.comms.report_status(Status.RUNNING, reason='init_position_done', progress=0.2)
        self.estado = STATE_GRASPING

    def _do_grasping(self):
        """
        Estado 1 — GRASPING
        Usa IK para pré-grasp → desce → fecha mão.
        Requer self.target_object_pose preenchida.
        """
        if self.target_object_pose is None:
            log.error('[Estado 1] target_object_pose não definida — a abortar')
            self.comms.report_status(Status.ERROR, reason='no_object_pose')
            self.estado = STATE_IDLE
            return

        log.info('[Estado 1] GRASPING...')
        self.comms.report_status(Status.RUNNING, reason='grasping', progress=0.3)

        T_obj      = self.target_object_pose
        left_wrist = np.eye(4)            # braço esquerdo parado

        # ── pré-grasp: 15 cm à frente no eixo Y do mundo ──────────────────
        T_pregrasp         = T_obj.copy()
        T_pregrasp[:3, 3] += np.array([0.0, 0.15, 0.0])

        q_current = np.array(
            [self.low_state.motor_state[j].q for j in range(15, 29)]
        )
        pregrasp_joints = self._solve_ik_run(left_wrist, T_pregrasp, q_current)
        right_pregrasp  = pregrasp_joints[7:]        # só braço direito

        self.hand_r.grip(0.1)                        # abre mão antes de mover
        self.move_joints(right_pregrasp, duration=3.0)
        time.sleep(1.0)
        self.comms.report_status(Status.RUNNING, reason='pregrasp_done', progress=0.5)

        # ── grasp: parte do pré-grasp para reduzir saltos de IK ───────────
        q_at_pregrasp = np.array(
            [self.low_state.motor_state[j].q for j in range(15, 29)]
        )
        grasp_joints = self._solve_ik_run(left_wrist, T_obj, q_at_pregrasp)
        right_grasp  = grasp_joints[7:]

        self.move_joints(right_grasp, duration=6.0)  # lento para precisão
        time.sleep(0.5)

        # ── fecha mão ─────────────────────────────────────────────────────
        self.hand_r.grip(0.8)
        time.sleep(0.8)                              # deixa estabilizar

        log.info('[Estado 1] Grasp concluído → LIFT')
        self.comms.report_status(Status.RUNNING, reason='grasp_done', progress=0.7)
        self.estado = STATE_LIFT

    def _do_lift(self):
        """
        Estado 2 — LIFT
        Levanta o braço com o objecto para a pose UP2.
        Mão permanece fechada.
        """
        log.info('[Estado 2] LIFT...')
        self.comms.report_status(Status.RUNNING, reason='lifting', progress=0.8)

        # NÃO chamar hand_r.stop() aqui — mantém a pega
        self.move_joints(MovementConfigs.RightArmUP2Position, duration=2.5)

        log.info('[Estado 2] Levantamento concluído → IDLE (à espera de entregar/largar)')
        self.comms.report_status(Status.DONE, reason='lift_done', progress=1.0)
        # Fica em IDLE à espera do próximo comando (DELIVERING ou RELEASING)
        self.estado = STATE_IDLE

    def _do_delivering(self):
        """
        Estado 3 — DELIVERING
        Estica o braço direito para a frente para entregar o objecto.
        Mão permanece fechada até o operador enviar NEUTRAL/largar.
        """
        log.info('[Estado 3] DELIVERING...')
        self.comms.report_status(Status.RUNNING, reason='delivering', progress=0.1)

        self.move_joints(MovementConfigs.ArmGivingPosition, duration=3.0)

        log.info('[Estado 3] Braço estendido → IDLE (à espera de largar)')
        self.comms.report_status(Status.DONE, reason='arm_extended', progress=1.0)
        self.estado = STATE_IDLE

    def _do_releasing(self):
        """
        Estado 4 — RELEASING
        Abre a mão → retrai o braço para a posição padrão.
        """
        log.info('[Estado 4] RELEASING...')
        self.comms.report_status(Status.RUNNING, reason='releasing', progress=0.1)

        # 1. Abre mão devagar para não atirar o objecto
        self.hand_r.grip(0.1)
        time.sleep(1.2)
        self.comms.report_status(Status.RUNNING, reason='hand_open', progress=0.5)

        # 2. Retrai braço para posição padrão
        self.move_joints(MovementConfigs.ArmStandardPosition, duration=3.0)

        log.info('[Estado 4] Largar concluído → IDLE')
        self.comms.report_status(Status.DONE, reason='release_done', progress=1.0)
        self.estado = STATE_IDLE

    # ─────────────────────────────────────────────────────────────────────────
    # Loop de controlo principal (50 Hz)
    # ─────────────────────────────────────────────────────────────────────────
    def My_Control(self):
        self.time_ += self.control_dt_

        # ── Inicialização única ───────────────────────────────────────────
        if self.waist_init is None:
            self.waist_init  = [0,0,0]
            self.left_init   = [self.low_state.motor_state[j].q for j in range(15, 22)]
            self.right_init  = [self.low_state.motor_state[j].q for j in range(22, 29)]

            # Activar arm_sdk
            self.low_cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = 1

            # Fixar ambos os braços e cintura na pose inicial
            for idx, j in enumerate(range(22, 29)):
                self._set_joint(j, self.right_init[idx], kp=200, kd=10)
            self._fix_left_arm()
            self._fix_waist()
            self._flush()
            self._arm_sdk_enabled = True
            log.info('arm_sdk activado; poses iniciais gravadas')
            return


        # ── Processar comando DDS (só quando IDLE) ────────────────────────
        if self.estado == STATE_IDLE:
            cmd = self.comms.take_cmd()
            if cmd is not None:
                self._handle_command(cmd)

        # ── Máquina de estados ────────────────────────────────────────────
        if   self.estado == STATE_INIT_POS:
            self._do_init_position()

        elif self.estado == STATE_GRASPING:
            self._do_grasping()

        elif self.estado == STATE_LIFT:
            self._do_lift()

        elif self.estado == STATE_DELIVERING:
            self._do_delivering()

        elif self.estado == STATE_RELEASING:
            self._do_releasing()

        # ── Manter cintura e braço esquerdo fixos entre estados ───────────
        self._fix_waist()
        self._fix_left_arm()
        self._flush()

    # ─────────────────────────────────────────────────────────────────────────
    # Despacho de comandos DDS
    # ─────────────────────────────────────────────────────────────────────────
    def _handle_command(self, cmd: GraspCommand):
        """
        Interpreta um GraspCommand e transita para o estado correcto.

        Comandos reconhecidos:
          EXTEND_ARM_FORWARD + objeto_id==''         → apanhar objeto
          EXTEND_ARM_FORWARD + objeto_id=='deliver'  → entregar objeto
          NEUTRAL                                    → largar objeto
        """
        log.info('A processar comando: postura=%s objeto=%s id=%s',
                 cmd.postura, cmd.objeto, cmd.objeto_id)

        if cmd.postura == Posture.EXTEND_ARM_FORWARD:

            if cmd.objeto_id == '':
                # ── Apanhar objeto ─────────────────────────────────────────
                # A pose do objeto vem no campo cmd.image como lista [x,y,z,r,p,y]
                # Ajusta conforme o teu IDL (pode ser cmd.objeto, cmd.pose, etc.)
                try:
                    # Tenta interpretar cmd.image como pose 6-DOF serializada
                    # Se o teu IDL transportar a pose num campo diferente,
                    # substitui cmd.image pelo campo correcto.
                    raw_pose = json.loads(cmd.image) if isinstance(cmd.image, str) else list(cmd.image)
                    if len(raw_pose) < 6:
                        raise ValueError(f'Pose incompleta: {raw_pose}')
                    self.target_object_pose = self._pose_to_SE3(raw_pose)
                    log.info('Pose do objeto recebida: %s', raw_pose)
                except Exception as e:
                    log.error('Não foi possível ler a pose do objeto: %s', e)
                    self.comms.report_status(Status.ERROR, reason=f'bad_pose: {e}')
                    return

                self.comms.report_status(Status.RUNNING, reason='grasp_started', progress=0.0)
                self.estado = STATE_INIT_POS

            elif cmd.objeto_id == 'deliver':
                # ── Entregar objeto ────────────────────────────────────────
                self.comms.report_status(Status.RUNNING, reason='deliver_started', progress=0.0)
                self.estado = STATE_DELIVERING

            else:
                log.warning('objeto_id desconhecido: %s', cmd.objeto_id)
                self.comms.report_status(Status.ERROR, reason=f'unknown_id:{cmd.objeto_id}')

        elif cmd.postura == Posture.NEUTRAL:
                if cmd.objeto_id == 'shutdown':          # <── novo
                    log.info('Shutdown command recebido — a terminar')
                    self.comms.report_status(Status.DONE, reason='shutdown')
                    self.done = True                     # o while True do main vê isto e faz sys.exit
                    return

            # ── Largar objeto ──────────────────────────────────────────────
            self.comms.report_status(Status.RUNNING, reason='release_started', progress=0.0)
            self.estado = STATE_RELEASING

        else:
            log.warning('Postura desconhecida: %s', cmd.postura)
            self.comms.report_status(Status.ERROR, reason=f'unknown_posture:{cmd.postura}')


# ─────────────────────────────────────────────────────────────────────────────
# Ponto de entrada
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('AVISO: Certifica-te de que não há obstáculos à volta do robô.')
    input('Pressiona Enter para continuar...\n')

    ChannelFactoryInitialize(0, 'enp117s0')

    custom = Custom()
    custom.Init()
    custom.Start()

    log.info('Controlador em execução. À espera de comandos DDS...')

    try:
        while True:
            time.sleep(1.0)
            if custom.done:
                log.info('Done flag activa — a sair')
                sys.exit(0)
    except KeyboardInterrupt:
        log.info('Interrompido pelo utilizador')
        sys.exit(0)