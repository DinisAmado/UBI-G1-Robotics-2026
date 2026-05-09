import sys
import os
import time
import logging
import math

# Configuração do caminho para importar os módulos do projeto (IDL e QoS)
pasta_atual = os.path.dirname(os.path.abspath(__file__))
pasta_src = os.path.abspath(os.path.join(pasta_atual, '../..'))
sys.path.append(pasta_src)

# Imports do CycloneDDS e dos tipos de mensagens definidos no IDL
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

# Imports do SDK da Unitree (Joystick Virtual) - Ter o SDK instalado e configurado
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.sport.sport_client import SportClient

# Configuração de Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] MOTION: %(message)s")
log = logging.getLogger("motion")

class MotionModule:
    def __init__(self, network_interface="enp117s0"):   # alterar para a placa de rede correta
        self.is_active = False  
        self.seq = 0
        
        # Inicialização do SDK da Unitree para controlar o G1
        log.info(f"A tentar ligar ao robô G1 na placa de rede: {network_interface}")
        try:
            # Ligar ao controlador nativo da Unitree e despertar o robô
            ChannelFactoryInitialize(0, network_interface)  
            self.sport_client = SportClient()   # Definir utilização do modelo nativo (G1)
            self.sport_client.SetTimeout(10.0)
            self.sport_client.Init()
            log.info("Ligação ao controlador nativo da Unitree (SportClient) estabelecida.")
        except Exception as e:
            log.error(f"Erro:Não foi possível ligar ao robô. Erro: {e}")
            sys.exit(1)
        
        # Inicialização do CycloneDDS (Comunicação com a Orquestração)
        log.info("A ligar à rede CycloneDDS...")
        self.dp = DomainParticipant(0)
        pub = Publisher(self.dp)
        sub = Subscriber(self.dp)

        # Configurar Tópicos com QoS Exatos
        t_orch_state = Topic(self.dp, "rt/orchestration/state", OrchestratorState, qos=QOS_ORCHESTRATION)
        t_cmd_vel    = Topic(self.dp, "rt/motion/cmd_vel", CmdVel, qos=QOS_MOTION)
        t_odom       = Topic(self.dp, "rt/motion/odometry", OdometryMsg, qos=QOS_ODOMETRY)
        t_heartbeat  = Topic(self.dp, "rt/orchestration/heartbeat", Heartbeat, qos=QOS_HEARTBEAT)

        # Configurar Readers e Writers
        self.r_orch_state = DataReader(sub, t_orch_state)
        self.r_cmd_vel    = DataReader(sub, t_cmd_vel)
        self.w_odom       = DataWriter(pub, t_odom)
        self.w_heartbeat  = DataWriter(pub, t_heartbeat)

    # Gera um identificador único para cada mensagem
    def _get_header(self) -> Header:
        self.seq += 1
        return Header(timestamp_ns=time.time_ns(), frame_id="base_link", seq=self.seq)

    # Lê o OrchestratorState e atua como Botão de Emergência.
    def update_global_state(self):
        samples = self.r_orch_state.take(1)
        if samples:
            estado_geral = samples[0]
            nova_atividade = estado_geral.active_modules.motion
            log.info(f"[DEBUG] OrchestratorState recebido: fase={estado_geral.phase.name}, motion={nova_atividade}")
            
            if nova_atividade != self.is_active:
                self.is_active = nova_atividade # Se self.is_active for True, o módulo está ativo e a escutar comandos. Se False, está em pausa.
                estado_str = "ATIVO (A escutar comandos)" if self.is_active else "EM PAUSA (Parado)"
                log.info(f"Fase mudou para {estado_geral.phase.name}. Motion está: {estado_str}")

    # Lê CmdVel (Navegação) e injeta no controlador da Unitree.
    def process_locomotion(self):
        vx, vy, wz = 0.0, 0.0, 0.0 

        if self.is_active:
            samples = self.r_cmd_vel.take(1)
            if samples:
                cmd = samples[-1]
                log.info(f"[DEBUG] CmdVel recebido: vx={cmd.vx} vy={cmd.vy} wz={cmd.wz}") 
                vx, vy, wz = cmd.vx, cmd.vy, cmd.wz
        
        # Envia a velocidade para os motores (Move() atua como o joystick analógico)
        self.sport_client.Move(vx, vy, wz)
        
        # Faz print apenas a cada segundo (50 ciclos) para não encher o terminal
        if self.is_active and self.seq % 50 == 0:
            log.debug(f"A executar no G1 -> Vx: {vx:.2f} | Vy: {vy:.2f} | Wz: {wz:.2f}")

    # Lê os encoders/estado nativo do robô para obter a odometria real pedida pelo SLAM.
    def read_encoders(self):
        try:
            # Acede ao estado de alto nível atualizado pelo controlador do G1
            estado_robo = self.sport_client.GetState()
            
            real_vx = estado_robo.velocity[0]
            real_vy = estado_robo.velocity[1]
            real_wz = estado_robo.yaw_speed
            
            pos_x = estado_robo.position[0]
            pos_y = estado_robo.position[1]
            pos_z = estado_robo.position[2]
            
            # O ângulo yaw (orientação Z) extraído do IMU
            yaw = estado_robo.imu.rpy[2] 
            
            return real_vx, real_vy, real_wz, pos_x, pos_y, pos_z, yaw
        except Exception:
            # Em caso de falha de comunicação momentânea, assume imobilidade
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    # Publica OdometryMsg de volta para a Navegação e SLAM com os valores reais.
    def publish_odometry(self):
        # Vai buscar os dados em tempo real, agora INCLUINDO o yaw!
        vx, vy, wz, px, py, pz, yaw = self.read_encoders()
        
        # Converte o yaw para Quaternião (exigência do contrato DDS)
        qw = math.cos(yaw / 2.0)
        qz = math.sin(yaw / 2.0)
        
        odom_msg = OdometryMsg(
            header=self._get_header(),
            pose=Pose(
                position=Vector3(x=px, y=py, z=pz),
                orientation=Quaternion(x=0.0, y=0.0, z=qz, w=qw)
            ),
            vx=vx, vy=vy, wz=wz
        )
        self.w_odom.write(odom_msg)

    # Publica o Heartbeat provando que o módulo está vivo e sem erros.
    def publish_heartbeat(self):
        hb_msg = Heartbeat(
            header=self._get_header(),
            module_name="motion",
            ready=True,
            error_msg="" 
        )
        self.w_heartbeat.write(hb_msg)

    # Ciclo Principal: 50Hz, compatível com a Deadline do QOS_MOTION (20ms).
    def run(self):
        sleep_s = 0.02  
        loop_counter = 0

        try:
            while True:
                self.update_global_state()
                self.process_locomotion()
                self.publish_odometry()
                
                # O Heartbeat bate a 1Hz (a cada 50 iterações do ciclo de 50Hz)
                if loop_counter % 50 == 0:
                    self.publish_heartbeat()
                
                loop_counter += 1
                time.sleep(sleep_s)

        except KeyboardInterrupt:
            log.info("Encerramento manual detetado. A travar o robô em segurança...")
            # Força o joystick virtual a zero para parar o G1
            self.sport_client.Move(0.0, 0.0, 0.0)

if __name__ == "__main__":
    # Confirma o nome da placa de rede ligada ao robô.
    MotionModule(network_interface="enp117s0").run()