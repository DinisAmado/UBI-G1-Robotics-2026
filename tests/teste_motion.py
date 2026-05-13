import os
import sys
import time
import logging
import threading

# Ajuste de path para o teu projeto
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from cyclonedds.domain import DomainParticipant
from cyclonedds.topic import Topic
from cyclonedds.pub import Publisher, DataWriter
from idl_ri import (
    Header, OrchestratorState, Phase, CmdVel, ActiveModules
)
from qos_profiles import QOS_ORCHESTRATION, QOS_MOTION

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] TEST_MOTION: %(message)s")
log = logging.getLogger("test_motion")

class MotionTester:
    def __init__(self):
        self._seq = 0
        self._dp = DomainParticipant(0)
        pub = Publisher(self._dp)

        # Writers necessários para o teu módulo "acordar" e mover
        self._w_state = DataWriter(pub, Topic(self._dp, "rt/orchestration/state", OrchestratorState, qos=QOS_ORCHESTRATION))
        self._w_cmd_vel = DataWriter(pub, Topic(self._dp, "rt/motion/cmd_vel", CmdVel, qos=QOS_MOTION))

    def _header(self):
        self._seq += 1
        return Header(timestamp_ns=time.time_ns(), frame_id="test_trigger", seq=self._seq)

    def activate_motion(self, active=True):
        """
        Simula o Orquestrador ativando o bit de motion.
        Nota: Usamos Phase.IDLE porque OPERATIONAL não existe no teu IDL.
        O teu main_motion.py apenas valida se 'motion' é True em active_modules.
        """
        state = OrchestratorState(
            header=self._header(),
            phase=Phase.IDLE, # Alterado de OPERATIONAL para IDLE (que existe no IDL)
            active_modules=ActiveModules(
                motion=active, 
                vision_objects=False, 
                vision_persons=False, 
                navigation=False, 
                grasping=False
            )
        )
        self._w_state.write(state)
        status = "ATIVADO" if active else "DESATIVADO"
        log.info(f"Estado do Motion no Orquestrador: {status}")

    def send_velocity(self, vx, vy, wz, duration):
        """Envia uma velocidade constante durante um tempo determinado."""
        log.info(f"Enviando Vx={vx:.2f}, Vy={vy:.2f}, Wz={wz:.2f} por {duration}s")
        end_time = time.time() + duration
        
        while time.time() < end_time:
            # Importante: No teu IDL, CmdVel tem um Header.
            msg = CmdVel(header=self._header(), vx=vx, vy=vy, wz=wz)
            self._w_cmd_vel.write(msg)
            time.sleep(0.02) # 50Hz (20ms)

        # Forçar paragem após o movimento
        self._w_cmd_vel.write(CmdVel(header=self._header(), vx=0.0, vy=0.0, wz=0.0))

    def run_sequence(self):
        try:
            log.info("Iniciando sequência de teste de movimento...")
            
            # 1. Ativar o módulo (Envia o bit que o teu main_motion.py espera)
            self.activate_motion(True)
            time.sleep(1)

            # 2. Teste para a frente (1 metro a 0.2m/s)
            log.info("--- Teste: Frente ---")
            self.send_velocity(0.2, 0.0, 0.0, 5.0)
            
            time.sleep(1) # Pausa para estabilização

            # 3. Teste de rotação (Sentido anti-horário)
            log.info("--- Teste: Rotação ---")
            self.send_velocity(0.0, 0.0, 0.4, 3.0)

            # 4. Parar e Desativar
            log.info("Teste concluído. Parando robô e desativando módulo.")
            self.send_velocity(0.0, 0.0, 0.0, 0.5)
            self.activate_motion(False)

        except KeyboardInterrupt:
            log.warning("Interrupção manual! Enviando comando de paragem...")
            self._w_cmd_vel.write(CmdVel(header=self._header(), vx=0.0, vy=0.0, wz=0.0))
            self.activate_motion(False)

if __name__ == "__main__":
    tester = MotionTester()
    tester.run_sequence()