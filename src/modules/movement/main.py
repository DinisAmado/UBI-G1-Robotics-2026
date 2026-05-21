import sys
import os
import time
import logging
import math

pasta_atual = os.path.dirname(os.path.abspath(__file__))
pasta_src = os.path.abspath(os.path.join(pasta_atual, '../..'))
if pasta_src not in sys.path:
    sys.path.append(pasta_src)

from cyclonedds.domain import DomainParticipant
from cyclonedds.topic import Topic
from cyclonedds.pub import Publisher, DataWriter
from cyclonedds.sub import Subscriber, DataReader

from idl_ri import (
    Header, OrchestratorState, Heartbeat,
    CmdVel, OdometryMsg, Pose, Vector3, Quaternion
)
from qos_profiles import (
    QOS_ORCHESTRATION, QOS_HEARTBEAT, 
    QOS_MOTION, QOS_ODOMETRY
)

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] MOTION: %(message)s")
log = logging.getLogger("motion")

class MotionModule:
    def __init__(self, network_interface="enp117s0"):   
        self.seq = 0
        self.is_active = True  
        self.last_cmd_time = time.time()
        
        log.info(f"A inicializar LocoClient na interface: {network_interface}")
        try:
            ChannelFactoryInitialize(0, network_interface)
            self.loco_client = LocoClient()
            self.loco_client.SetTimeout(10.0)
            self.loco_client.Init()
            log.info("CONEXÃO HARDWARE: Sucesso ao ligar ao LocoClient do G1.")
        except Exception as e:
            log.error(f"ERRO HARDWARE: Falha crítica ao ligar ao SDK: {e}")
            sys.exit(1)

        log.info("A inicializar barramento CycloneDDS...")
        self.dp = DomainParticipant(0)
        pub = Publisher(self.dp)
        sub = Subscriber(self.dp)

        self.r_state = DataReader(sub, Topic(self.dp, "rt/orchestration/state", OrchestratorState, qos=QOS_ORCHESTRATION))
        self.r_cmd_vel = DataReader(sub, Topic(self.dp, "rt/motion/cmd_vel", CmdVel, qos=QOS_MOTION))

        self.w_odom = DataWriter(pub, Topic(self.dp, "rt/motion/odometry", OdometryMsg, qos=QOS_ODOMETRY))
        self.w_heartbeat = DataWriter(pub, Topic(self.dp, "rt/orchestration/heartbeat", Heartbeat, qos=QOS_HEARTBEAT))
        log.info("DDS: Tópicos, Readers e Writers configurados.")

    def _get_header(self) -> Header:
        self.seq += 1
        return Header(timestamp_ns=time.time_ns(), frame_id="base_link", seq=self.seq)

    def update_global_state(self):
        samples = self.r_state.take()
        for s in samples:
            if self.is_active != s.active_modules.motion:
                self.is_active = s.active_modules.motion
                log.info(f"MUDANÇA DE ESTADO: Estado ativo alterado para: {self.is_active} (Fase: {s.phase.name})")

    def process_locomotion(self):
        if not self.is_active:
            self.loco_client.Move(0.0, 0.0, 0.0)
            return

        samples = self.r_cmd_vel.take()
        if samples:
            cmd = samples[-1] 
            self.last_cmd_time = time.time()
            log.info(f"COMANDO RECEBIDO (DDS) -> [vx: {cmd.vx:.2f}, vy: {cmd.vy:.2f}, wz: {cmd.wz:.2f}] (Seq: {cmd.header.seq})")
            
            try:
                self.loco_client.Move(cmd.vx, cmd.vy, cmd.wz)
                log.info(f"EXECUÇÃO HARDWARE (G1) -> Comando enviado com sucesso.")
            except Exception as e:
                log.error(f"ERRO EXECUÇÃO: Falha ao enviar movimento para o robô: {e}")
        else:
            tempo_sem_comandos = time.time() - self.last_cmd_time
            if tempo_sem_comandos > 0.5:
                self.loco_client.Move(0.0, 0.0, 0.0)
                if not hasattr(self, 'last_timeout_log') or (time.time() - self.last_timeout_log) >= 5.0:
                    log.warning(f"TIMEOUT SEGURANÇA: Sem comandos há {tempo_sem_comandos:.2f}s. Robô travado.")
                    self.last_timeout_log = time.time()

    def publish_odometry(self):
        try:
            estado_robo = self.loco_client.GetState()
            
            vx = estado_robo.velocity[0]
            vy = estado_robo.velocity[1]
            wz = estado_robo.yaw_speed
            
            px = estado_robo.position[0]
            py = estado_robo.position[1]
            pz = estado_robo.position[2]
            
            yaw = estado_robo.imu.rpy[2]
            
            qw = math.cos(yaw / 2.0)
            qz = math.sin(yaw / 2.0)
            
        except Exception as e:
            log.warning(f"AVISO ODOMETRIA: Falha temporária ao ler encoders: {e}")
            vx, vy, wz, px, py, pz, qz, qw = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0

        odom_msg = OdometryMsg(
            header=self._get_header(),
            pose=Pose(
                position=Vector3(x=px, y=py, z=pz),
                orientation=Quaternion(x=0.0, y=0.0, z=qz, w=qw)
            ),
            vx=vx, vy=vy, wz=wz
        )
        self.w_odom.write(odom_msg)

    def publish_heartbeat(self):
        hb = Heartbeat(header=self._get_header(), module_name="motion", ready=True, error_msg="")
        self.w_heartbeat.write(hb)
        log.info("HEARTBEAT: Sinal de vida enviado ao Orquestrador.")

    def run(self):
        log.info("Módulo Motion em execução [Loop de 50Hz iniciado].")
        loop_counter = 0

        while True:
            self.update_global_state()
            self.process_locomotion()
            self.publish_odometry()
            
            if loop_counter % 50 == 0:
                self.publish_heartbeat()
            
            loop_counter += 1
            time.sleep(0.02)

if __name__ == "__main__":
    interface = sys.argv[1] if len(sys.argv) > 1 else "enp117s0"
    MotionModule(network_interface=interface).run()