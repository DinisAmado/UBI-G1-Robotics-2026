import sys
import os
import time

# Configuração de caminhos para os imports
pasta_atual = os.path.dirname(os.path.abspath(__file__))
pasta_src = os.path.abspath(os.path.join(pasta_atual, '../..'))
if pasta_src not in sys.path:
    sys.path.append(pasta_src)

from cyclonedds.domain import DomainParticipant
from cyclonedds.topic import Topic
from cyclonedds.pub import Publisher, DataWriter
from cyclonedds.sub import Subscriber, DataReader

from idl_ri import Header, CmdVel, OdometryMsg
from qos_profiles import QOS_MOTION, QOS_ODOMETRY

def run_mock_navigation():
    print("--- SIMULADOR DE NAVEGAÇÃO ---")
    
    dp = DomainParticipant(0)
    pub = Publisher(dp)
    sub = Subscriber(dp)

    # 1. A Navegação ESCREVE comandos de velocidade
    w_cmd_vel = DataWriter(pub, Topic(dp, "rt/motion/cmd_vel", CmdVel, qos=QOS_MOTION))
    
    # 2. A Navegação LÊ a odometria do robô
    r_odom = DataReader(sub, Topic(dp, "rt/motion/odometry", OdometryMsg, qos=QOS_ODOMETRY))

    print("A inicializar DDS... o robô vai arrancar em 3 segundos.")
    time.sleep(3)

    seq = 0
    try:
        while True:
            seq += 1
            header = Header(timestamp_ns=time.time_ns(), frame_id="nav_sim", seq=seq)

            # Publicar velocidade constante de 0.30 m/s para vencer a deadzone
            cmd = CmdVel(header=header, vx=0.30, vy=0.0, wz=0.0)
            w_cmd_vel.write(cmd)
            
            # Ler a odometria que vem do MotionModule e imprimir no ecrã
            samples = r_odom.take()
            if samples:
                odom = samples[-1]
                print(f"[ODOMETRIA] Pos: ({odom.pose.position.x:.2f}, {odom.pose.position.y:.2f}) | Vel atual: {odom.vx:.2f} m/s")
            else:
                print(f"[A enviar CmdVel: 0.30 m/s] ... a aguardar dados de odometria.")

            # Manter a cadência de 50Hz exigida pelo QoS e pelo robô
            time.sleep(0.02) 

    except KeyboardInterrupt:
        print("\n[SIMULADOR] Cancelamento manual detetado!")
        # Segurança: Mandar imediatamente ordem de paragem para o MotionModule
        header = Header(timestamp_ns=time.time_ns(), frame_id="nav_sim", seq=seq+1)
        w_cmd_vel.write(CmdVel(header=header, vx=0.0, vy=0.0, wz=0.0))
        print("Ordem de imobilização (0.0) enviada. Encerrando o simulador.")

if __name__ == "__main__":
    run_mock_navigation()