import sys
import os
import time
import math

# Ajustar path para importar os módulos da pasta src
pasta_atual = os.path.dirname(os.path.abspath(__file__))
pasta_src = os.path.abspath(os.path.join(pasta_atual, '..'))
if pasta_src not in sys.path:
    sys.path.append(pasta_src)

from cyclonedds.domain import DomainParticipant
from cyclonedds.topic import Topic
from cyclonedds.pub import Publisher
from idl_ri import OrchestratorState, CmdVel, Header, ModuleStates, PhaseEnum
from qos_profiles import QOS_ORCHESTRATION, QOS_MOTION

def main():
    dp = DomainParticipant(0)
    pub = Publisher(dp)

    # Tópicos
    t_state = Topic(dp, "rt/orchestration/state", OrchestratorState, qos=QOS_ORCHESTRATION)
    t_cmd = Topic(dp, "rt/motion/cmd_vel", CmdVel, qos=QOS_MOTION)

    w_state = Publisher(dp).create_writer(t_state)
    w_cmd = Publisher(dp).create_writer(t_cmd)

    print("--- Test Commander G1 Inicializado ---")
    print("1. Ativando módulo Motion via OrchestratorState...")
    
    # Simula o Orquestrador ativando o Motion
    state_msg = OrchestratorState(
        header=Header(timestamp_ns=time.time_ns(), frame_id="map", seq=1),
        phase=PhaseEnum.OPERATIONAL,
        active_modules=ModuleStates(motion=True, vision=False, grasping=False, hmi=False)
    )
    w_state.write(state_msg)

    try:
        print("2. Enviando comandos de velocidade (0.1 m/s) por 5 segundos...")
        start_time = time.time()
        
        while time.time() - start_time < 5:
            # Envia movimento para a frente suave
            cmd = CmdVel(vx=0.1, vy=0.0, wz=0.0)
            w_cmd.write(cmd)
            time.sleep(0.02) # Mantém os 50Hz esperados

        print("3. Teste finalizado. Parando o robô...")
        w_cmd.write(CmdVel(vx=0.0, vy=0.0, wz=0.0))
        
    except KeyboardInterrupt:
        w_cmd.write(CmdVel(vx=0.0, vy=0.0, wz=0.0))
        print("\nInterrompido pelo utilizador.")

if __name__ == "__main__":
    main()
