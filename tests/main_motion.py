import sys
import os
import time
import logging
from cyclonedds.domain import DomainParticipant
from cyclonedds.topic import Topic
from cyclonedds.sub import Subscriber, DataReader
from idl_ri import CmdVel
from qos_profiles import QOS_MOTION

# SDK Unitree
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.sport.sport_client import SportClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [MOTION] %(message)s")
log = logging.getLogger("motion")

class MotionModule:
    def __init__(self, network_interface="enp117s0"):
        log.info(f"A ligar ao G1 na interface: {network_interface}")
        try:
            ChannelFactoryInitialize(0, network_interface)  
            self.sport_client = SportClient()
            self.sport_client.Init()
            log.info("SDK Unitree Inicializado.")
        except Exception as e:
            log.error(f"Erro SDK: {e}")
            sys.exit(1)
        
        self.dp = DomainParticipant(0)
        sub = Subscriber(self.dp)
        self.r_cmd_vel = DataReader(sub, Topic(self.dp, "rt/motion/cmd_vel", CmdVel, qos=QOS_MOTION))

    def run(self):
        log.info("Modo de Teste Direto: A aceitar velocidades SEM verificação de estado.")
        try:
            while True:
                # 'take' retira as mensagens da fila
                samples = self.r_cmd_vel.take()
                if samples:
                    cmd = samples[-1] # Executa a última recebida
                    log.info(f"A EXECUTAR NO G1 -> Vx: {cmd.vx:.2f} Vy: {cmd.vy:.2f} Wz: {cmd.wz:.2f}")
                    self.sport_client.Move(cmd.vx, cmd.vy, cmd.wz)
                
                time.sleep(0.02) # Ciclo de 50Hz
        except KeyboardInterrupt:
            self.sport_client.Move(0.0, 0.0, 0.0)
            log.info("Paragem de emergência executada.")

if __name__ == "__main__":
    # Garante que a interface de rede (enp117s0) é a correta para o teu PC
    MotionModule(network_interface="enp117s0").run()