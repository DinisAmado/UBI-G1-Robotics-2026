import sys
import os
import time
import logging
import math

# Configuração do caminho para importar os módulos do projeto
pasta_atual = os.path.dirname(os.path.abspath(__file__))
pasta_src = os.path.abspath(os.path.join(pasta_atual, '../..'))
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
from unitree_sdk2py.go2.sport.sport_client import SportClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] MOTION: %(message)s")
log = logging.getLogger("motion")

class MotionModule:
    def __init__(self, network_interface="enp117s0"):   
        # FORÇADO A TRUE PARA TESTE ISOLADO
        self.is_active = True  
        self.seq = 0
        
        log.info(f"A inicializar SDK na interface: {network_interface}")
        try:
            ChannelFactoryInitialize(0, network_interface)
            self.sport_client = SportClient()
            self.sport_client.Init()
        except Exception as e:
            log.error(f"Falha ao ligar ao robô: {e}")
            sys.exit(1)

        # Configuração DDS
        self.dp = DomainParticipant(0)
        pub = Publisher(self.dp)
        sub = Subscriber(self.dp)

        # Readers
        self.r_state = DataReader(sub, Topic(self.dp, "rt/orchestration/state", OrchestratorState, qos=QOS_ORCHESTRATION))
        self.r_cmd_vel = DataReader(sub, Topic(self.dp, "rt/motion/cmd_vel", CmdVel, qos=QOS_MOTION))

        # Writers
        self.w_odom = DataWriter(pub, Topic(self.dp, "rt/motion/odometry", OdometryMsg, qos=QOS_ODOMETRY))
        self.w_heartbeat = DataWriter(pub, Topic(self.dp, "rt/orchestration/heartbeat", Heartbeat, qos=QOS_HEARTBEAT))

    def _get_header(self):
        self.seq += 1
        return Header(timestamp_ns=time.time_ns(), frame_id="body", seq=self.seq)

    def update_global_state(self):
        """Verifica o estado do orquestrador mas não bloqueia o movimento neste modo de teste."""
        samples = self.r_state.take()
        for s in samples:
            # Apenas logamos a mudança para verificar se o tópico funciona
            log.info(f"Estado recebido do Orquestrador: Phase={s.phase}, MotionActive={s.active_modules.motion}")

    def process_locomotion(self):
        """Lê as velocidades do DDS e envia para o robô."""
        samples = self.r_cmd_vel.take()
        if samples:
            cmd = samples[-1] # Executa o comando mais recente
            log.info(f"A EXECUTAR -> Vx: {cmd.vx:.2f}, Vy: {cmd.vy:.2f}, Wz: {cmd.wz:.2f}")
            self.sport_client.Move(cmd.vx, cmd.vy, cmd.wz)
        # Se não houver amostras, não fazemos nada (mantém a última velocidade ou para conforme o SDK)

    def publish_odometry(self):
        """Simulação de publicação de odometria para teste."""
        odom_msg = OdometryMsg(
            header=self._get_header(),
            pose=Pose(position=Vector3(x=0.0, y=0.0, z=0.0)),
            vx=0.0, vy=0.0, wz=0.0
        )
        self.w_odom.write(odom_msg)

    def publish_heartbeat(self):
        hb_msg = Heartbeat(header=self._get_header(), module_name="motion", ready=True)
        self.w_heartbeat.write(hb_msg)

    def run(self):
        log.info("Módulo Motion em execução (MODO DE TESTE DIRETO).")
        sleep_s = 0.02  
        loop_counter = 0

        try:
            while True:
                self.update_global_state()
                self.process_locomotion()
                self.publish_odometry()
                
                if loop_counter % 50 == 0:
                    self.publish_heartbeat()
                
                loop_counter += 1
                time.sleep(sleep_s)

        except KeyboardInterrupt:
            log.info("Paragem manual. A imobilizar robô...")
            self.sport_client.Move(0.0, 0.0, 0.0)

if __name__ == "__main__":
    # Certifica-te que 'enp117s0' é o nome correto da tua interface de rede
    module = MotionModule(network_interface="enp117s0")
    module.run()