"""
grasp_control.py
================
Integração completa: CycloneDDS + controlo de braços G1

Comandos suportados (via tópico rt/grasp/command):
  - postura=EXTEND_ARM_FORWARD, objeto_id=''         → "apanhar objeto"   (estados 0→1→2)
  - postura=EXTEND_ARM_FORWARD, objeto_id='carry'    → "modo transporte"  (estado 3)
  - postura=EXTEND_ARM_FORWARD, objeto_id='deliver'  → "entregar objeto"  (estado 4)
  - postura=NEUTRAL,            objeto_id='drop'     → "largar imediato"  (estado 5)
  - postura=NEUTRAL,            objeto_id='shutdown' → desligar script

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
    ▼  EXTEND_ARM_FORWARD + objeto_id=='carry'
  CARRY (3)          → braço neutro/transporte
    │
    ▼  EXTEND_ARM_FORWARD + objeto_id=='deliver'
  DELIVERING (4)     → estica braço para ArmGivingPosition + abre garra
    │   (ou directamente do IDLE/LIFT)
    ▼  NEUTRAL + objeto_id=='drop'
  DROPPING (5)       → abre mão imediatamente (recuperação de erro)
    │
    ▼
  IDLE

2026-05 — versão 4: semântica objeto_id actualizada, pose via Pose6DOF, tópico separado removido
"""

import time
import sys
import json
import logging
import subprocess
import threading

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
    Pose6DOF,       # ← substituiu Image
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
STATE_IDLE       = -1
STATE_INIT_POS   =  0   # mover para posição inicial + braço cima
STATE_GRASPING   =  1   # IK pré-grasp → desce → fecha mão
STATE_LIFT       =  2   # levanta braço (apanhar concluído)
STATE_CARRY      =  3   # braço neutro para transporte
STATE_DELIVERING =  4   # estica braço + abre garra (entregar)
STATE_DROPPING   =  5   # abre mão imediatamente (drop/erro)


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

    # Braço direito em cima (pose 2 — final após grasp / lift)
    RightArmUP2Position = [
        (22, -0.3068), (23, -1.1202), (24, -0.0598),
        (25,  0.2346), (26,  1.4680), (27, -0.2060), (28,  1.2231),
    ]

    # Braço direito estendido para frente (entregar) — alterar conforme calibração
    ArmGivingPosition = [
        (22, -0.5741), (23, -0.1391), (24, -0.2563),
        (25,  0.5985), (26, -0.0296), (27, -0.1613), (28,  0.1152),
    ]

    # Braço direito em posição de transporte (carry) — neutro ao lado do corpo
    ArmCarryPosition = [
        (22,  0.1035), (23, -0.3205), (24, -0.0719), (25,  0.0792),
        (26, -0.0667), (27,  0.0186), (28, -0.0127),
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
# Subscreve apenas: rt/grasp/command
# Publica apenas:   rt/grasp/status
# A pose do objeto vem dentro do GraspCommand (campo pose: Pose6DOF)
# ─────────────────────────────────────────────────────────────────────────────
class GraspComms:

    def __init__(self):
        self._dp  = DomainParticipant(DOMAIN_ID)
        self._pub = Publisher(self._dp)
        self._sub = Subscriber(self._dp)

        # Único tópico subscrito
        self._t_cmd    = Topic(self._dp, 'rt/grasp/command', GraspCommand,   qos=QOS_GRASP)
        self._r_cmd    = DataReader(self._sub, self._t_cmd)

        # Tópico de status publicado
        self._t_status = Topic(self._dp, 'rt/grasp/status',  GraspStatusMsg, qos=QOS_GRASP)
        self._w_status = DataWriter(self._pub, self._t_status)

        self._lock       = threading.Lock()
        self._seq        = 0
        self.pending_cmd: GraspCommand | None = None

        # Thread de polling DDS (daemon — termina com o processo principal)
        self._poll_thread = threading.Thread(
            target=self._poll_loop, daemon=True, name='dds_poll'
        )
        self._poll_thread.start()
        log.info('GraspComms iniciado no domínio %d', DOMAIN_ID)

    # ── Polling interno ───────────────────────────────────────────────────────
    def _poll_loop(self):
        while True:
            samples = self._r_cmd.take(1)
            if samples:
                with self._lock:
                    self.pending_cmd = samples[0]
                cmd = samples[0]
                log.info('Comando recebido: postura=%s objeto=%s objeto_id=%s',
                         cmd.postura, cmd.objeto, cmd.objeto_id)
            time.sleep(0.02)   # 50 Hz

    # ── API pública ───────────────────────────────────────────────────────────
    def take_cmd(self) -> GraspCommand | None:
        """Devolve e limpa o comando pendente (thread-safe)."""
        with self._lock:
            cmd = self.pending_cmd
            self.pending_cmd = None
        return cmd

    def report_status(self, status: Status, reason: str = '', progress: float = 0.0):
        with self._lock:
            self._seq += 1
            seq = self._seq
        self._w_status.write(GraspStatusMsg(
            header=Header(
                timestamp_ns=time.time_ns(),
                frame_id='grasp',
                seq=seq,
            ),
            status=status,
            reason=reason,
            progress=progress,
        ))
        log.info('Status: %s (%.0f%%) — %s', status, progress * 100, reason)


# ─────────────────────────────────────────────────────────────────────────────
# Controlador principal
# ─────────────────────────────────────────────────────────────────────────────
class Custom:

    def __init__(self):
        self.time_       = 0.0
        self.control_dt_ = 0.02   # 50 Hz
        self.kp          = 100.
        self.kd          = 5.

        self.low_cmd   = unitree_hg_msg_dds__LowCmd_()
        self.low_state = None
        self.first_update_low_state = False
        self.crc  = CRC()
        self.done = False

        # Estado interno da máquina de estados
        self.estado: int = STATE_IDLE

        # Pose do objeto (SE3 4×4) preenchida quando chega o comando de grasp
        self.target_object_pose: np.ndarray | None = None

        # Poses iniciais capturadas no primeiro ciclo
        self.waist_init = None
        self.left_init  = None
        self.right_init = None

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
        for idx, j in enumerate([12, 13, 14]):
            self._set_joint(j, self.waist_init[idx], kp=200, kd=10)

    def _fix_left_arm(self):
        for idx, j in enumerate(range(15, 22)):
            self._set_joint(j, self.left_init[idx], kp=200, kd=10)

    def _flush(self):
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.arm_sdk_publisher.Write(self.low_cmd)

    def move_joints(self, targets: list, duration: float = 2.0):
        """
        Movimento suave (smoothstep cúbico) para os joints indicados.
        Bloqueia até ao fim. Mantém cintura e braço esquerdo fixos.
        """
        if self.low_state is None:
            log.warning('move_joints: low_state não disponível')
            return

        start_q = {j: self.low_state.motor_state[j].q for j, _ in targets}
        t0      = time.time()

        while True:
            t     = min((time.time() - t0) / duration, 1.0)
            ratio = t * t * (3.0 - 2.0 * t)   # smoothstep

            for j, q_target in targets:
                q = (1.0 - ratio) * start_q[j] + ratio * q_target
                self._set_joint(j, q)

            self._fix_waist()
            self._fix_left_arm()
            self._flush()

            if t >= 1.0:
                break
            time.sleep(self.control_dt_)

        log.info('move_joints concluído (%.1fs)', duration)

    # ─────────────────────────────────────────────────────────────────────────
    # Cinemática inversa
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _pose6dof_to_SE3(pose: Pose6DOF) -> np.ndarray:
        """Pose6DOF (x,y,z,roll,pitch,yaw) → 4×4 SE3."""
        se3 = np.eye(4)
        se3[:3, :3] = R.from_euler('xyz', [pose.roll, pose.pitch, pose.yaw]).as_matrix()
        se3[:3,  3] = [pose.x, pose.y, pose.z]
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
        """Estado 0 — move para posição padrão + braço direito cima."""
        log.info('[Estado 0] INIT_POSITION')
        self.comms.report_status(Status.RUNNING, reason='init_position', progress=0.1)

        self.move_joints(MovementConfigs.ArmStandardPosition, duration=2.0)
        self.move_joints(MovementConfigs.RightArmUP2Position,  duration=2.0)

        self.comms.report_status(Status.RUNNING, reason='init_done', progress=0.2)
        self.estado = STATE_GRASPING

    def _do_grasping(self):
        """Estado 1 — IK pré-grasp → desce → fecha mão."""
        if self.target_object_pose is None:
            log.error('[Estado 1] target_object_pose não definida — abortar')
            self.comms.report_status(Status.FAILED, reason='no_object_pose')
            self.estado = STATE_IDLE
            return

        log.info('[Estado 1] GRASPING')
        self.comms.report_status(Status.RUNNING, reason='grasping', progress=0.3)

        T_obj      = self.target_object_pose
        left_wrist = np.eye(4)

        # pré-grasp: 15 cm acima
        T_pregrasp         = T_obj.copy()
        T_pregrasp[:3, 3] += np.array([0.0, 0.15, 0.0])

        q_current = np.array([self.low_state.motor_state[j].q for j in range(15, 29)])
        right_pregrasp = self._solve_ik_run(left_wrist, T_pregrasp, q_current)[7:]

        self.hand_r.grip(0.1)   # abre mão
        self.move_joints(right_pregrasp, duration=3.0)
        time.sleep(1.0)
        self.comms.report_status(Status.RUNNING, reason='pregrasp_done', progress=0.5)

        # grasp final — começa do pré-grasp
        q_at_pregrasp = np.array([self.low_state.motor_state[j].q for j in range(15, 29)])
        right_grasp   = self._solve_ik_run(left_wrist, T_obj, q_at_pregrasp)[7:]

        self.move_joints(right_grasp, duration=6.0)
        time.sleep(0.5)

        self.hand_r.grip(0.8)   # fecha mão
        time.sleep(0.8)

        self.comms.report_status(Status.RUNNING, reason='grasp_done', progress=0.7)
        self.estado = STATE_LIFT

    def _do_lift(self):
        """Estado 2 — levanta o braço. Mão permanece fechada."""
        log.info('[Estado 2] LIFT')
        self.comms.report_status(Status.RUNNING, reason='lifting', progress=0.8)

        self.move_joints(MovementConfigs.RightArmUP2Position, duration=2.5)

        self.comms.report_status(Status.DONE, reason='lift_done', progress=1.0)
        # Para aqui — aguarda comando 'carry' ou 'deliver'
        self.estado = STATE_IDLE

    def _do_carry(self):
        """Estado 3 — braço em posição neutra de transporte. Mão fechada."""
        log.info('[Estado 3] CARRY')
        self.comms.report_status(Status.RUNNING, reason='carry', progress=0.1)

        self.move_joints(MovementConfigs.ArmCarryPosition, duration=2.0)

        self.comms.report_status(Status.DONE, reason='carry_done', progress=1.0)
        # Para aqui — aguarda comando 'deliver' ou 'drop'
        self.estado = STATE_IDLE

    def _do_delivering(self):
        """Estado 4 — estica braço para frente e abre garra para entregar."""
        log.info('[Estado 4] DELIVERING')
        self.comms.report_status(Status.RUNNING, reason='delivering', progress=0.1)

        self.move_joints(MovementConfigs.ArmGivingPosition, duration=3.0)

        # Abre garra para a pessoa receber o objeto
        self.hand_r.grip(0.1)
        time.sleep(1.0)

        self.comms.report_status(Status.DONE, reason='deliver_done', progress=1.0)
        # Para aqui — aguarda comando 'drop' (largar de vez) ou novo grasp
        self.estado = STATE_IDLE

    def _do_dropping(self):
        """Estado 5 — abre mão imediatamente (drop de emergência/erro)."""
        log.info('[Estado 5] DROPPING')
        self.comms.report_status(Status.RUNNING, reason='dropping', progress=0.1)

        self.hand_r.grip(0.1)
        time.sleep(0.5)   # mais rápido que o releasing normal

        self.move_joints(MovementConfigs.ArmStandardPosition, duration=2.0)

        self.comms.report_status(Status.DONE, reason='drop_done', progress=1.0)
        self.estado = STATE_IDLE

    # ─────────────────────────────────────────────────────────────────────────
    # Loop de controlo principal (50 Hz)
    # ─────────────────────────────────────────────────────────────────────────
    def My_Control(self):
        self.time_ += self.control_dt_

        # ── Inicialização única (primeiro ciclo) ──────────────────────────
        if self.waist_init is None:
            self.waist_init = [0, 0, 0]   # cintura bloqueada a zero
            self.left_init  = [self.low_state.motor_state[j].q for j in range(15, 22)]
            self.right_init = [self.low_state.motor_state[j].q for j in range(22, 29)]

            self.low_cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = 1   # activa arm_sdk

            for idx, j in enumerate(range(22, 29)):
                self._set_joint(j, self.right_init[idx], kp=200, kd=10)
            self._fix_left_arm()
            self._fix_waist()
            self._flush()
            log.info('arm_sdk activado; poses iniciais gravadas')
            return

        # ── Aceitar comandos DDS apenas quando IDLE ───────────────────────
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
        elif self.estado == STATE_CARRY:
            self._do_carry()
        elif self.estado == STATE_DELIVERING:
            self._do_delivering()
        elif self.estado == STATE_DROPPING:
            self._do_dropping()

        # ── Manter cintura e braço esquerdo fixos em todos os estados ─────
        self._fix_waist()
        self._fix_left_arm()
        self._flush()

    # ─────────────────────────────────────────────────────────────────────────
    # Despacho de comandos DDS
    # ─────────────────────────────────────────────────────────────────────────
    def _handle_command(self, cmd: GraspCommand):
        """
        Semântica do objeto_id:
          ''         → apanhar o objeto indicado em cmd.objeto (usa cmd.pose)
          'carry'    → modo transporte, braço neutro
          'deliver'  → entregar à pessoa, abre garra
          'drop'     → largar imediatamente (recuperação de erro)
          'shutdown' → desligar o script
        """
        log.info('Comando: postura=%s objeto="%s" objeto_id="%s"',
                 cmd.postura, cmd.objeto, cmd.objeto_id)

        # ── shutdown — tratado antes de qualquer outra coisa ──────────────
        if cmd.objeto_id == 'shutdown':
            log.info('Shutdown recebido — a terminar')
            self.comms.report_status(Status.DONE, reason='shutdown')
            self.done = True
            return

        # ── drop — funciona em qualquer postura, prioridade alta ──────────
        if cmd.objeto_id == 'drop':
            self.comms.report_status(Status.RUNNING, reason='drop_started', progress=0.0)
            self.estado = STATE_DROPPING
            return

        # ── comandos dependentes da postura ───────────────────────────────
        if cmd.postura == Posture.EXTEND_ARM_FORWARD:

            if cmd.objeto_id == '':
                # apanhar objeto — pose vem em cmd.pose (Pose6DOF)
                try:
                    self.target_object_pose = self._pose6dof_to_SE3(cmd.pose)
                    log.info('Pose do objeto: x=%.3f y=%.3f z=%.3f r=%.3f p=%.3f y=%.3f',
                             cmd.pose.x, cmd.pose.y, cmd.pose.z,
                             cmd.pose.roll, cmd.pose.pitch, cmd.pose.yaw)
                except Exception as e:
                    log.error('Erro ao ler pose: %s', e)
                    self.comms.report_status(Status.FAILED, reason=f'bad_pose:{e}')
                    return
                self.comms.report_status(Status.RUNNING, reason='grasp_started', progress=0.0)
                self.estado = STATE_INIT_POS

            elif cmd.objeto_id == 'carry':
                self.comms.report_status(Status.RUNNING, reason='carry_started', progress=0.0)
                self.estado = STATE_CARRY

            elif cmd.objeto_id == 'deliver':
                self.comms.report_status(Status.RUNNING, reason='deliver_started', progress=0.0)
                self.estado = STATE_DELIVERING

            else:
                log.warning('objeto_id desconhecido: "%s"', cmd.objeto_id)
                self.comms.report_status(Status.FAILED, reason=f'unknown_id:{cmd.objeto_id}')

        elif cmd.postura == Posture.NEUTRAL:
            # NEUTRAL sem objeto_id especial — não faz nada (drop já tratado acima)
            log.warning('NEUTRAL recebido sem objeto_id reconhecido (usa drop para largar)')
            self.comms.report_status(Status.FAILED, reason='neutral_no_action')

        else:
            log.warning('Postura desconhecida: %s', cmd.postura)
            self.comms.report_status(Status.FAILED, reason=f'unknown_posture:{cmd.postura}')


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

    log.info('Controlador em execução. À espera de comandos DDS em rt/grasp/command ...')

    try:
        while True:
            time.sleep(1.0)
            if custom.done:
                log.info('Done flag activa — a sair')
                sys.exit(0)
    except KeyboardInterrupt:
        log.info('Interrompido pelo utilizador')
        sys.exit(0)