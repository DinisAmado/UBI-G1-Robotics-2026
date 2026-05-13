#!/usr/bin/env python3
"""
main_test.py — Simulador de SLAM + Navegação com função RELAY.
Atua como ponte: teste_motion -> este script -> main_motion.
"""

import os
import sys
import time
import logging
import threading

# Ajuste de path para importar do diretório raiz
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from cyclonedds.domain import DomainParticipant
from cyclonedds.topic import Topic
from cyclonedds.pub import Publisher, DataWriter
from cyclonedds.sub import Subscriber, DataReader

from idl_ri import (
    Header, Status, Image,
    Intent, Acao,
    OrchestratorState, Phase, ActiveModules,
    NavStatusMsg as NavStatus,
    GraspStatusMsg as GraspStatus,
    Objects as VisionObjects, ObjectDetection,
    Persons as VisionPersons, PersonDetection,
    Locations, Location, Pose, Vector3, Quaternion,
    CmdVel, OdometryMsg,
)
from qos_profiles import (
    QOS_HMI, QOS_ORCHESTRATION, QOS_NAV,
    QOS_GRASP, QOS_VISION, QOS_SLAM_MAP, QOS_MOTION, QOS_ODOMETRY,
)

# ─── Configuração ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] NAV_SIM: %(message)s",
)
log = logging.getLogger("nav_sim")

DOMAIN_ID     = 0
ALVO_OBJETO   = "bola de tenis"
ALVO_PESSOA   = "person_01"
PHASE_TIMEOUT = 30.0   

# ─── Simulador ────────────────────────────────────────────────────────────────

class NavSim:

    def __init__(self):
        self._seq  = 0
        self._dp   = DomainParticipant(DOMAIN_ID)
        pub        = Publisher(self._dp)
        sub        = Subscriber(self._dp)

        # ── Writers ────────────────────────────────────────────────────────
        self._w_intent      = DataWriter(pub, Topic(self._dp, "rt/hmi/intent",       Intent,        qos=QOS_HMI))
        self._w_nav_status  = DataWriter(pub, Topic(self._dp, "rt/nav/status",       NavStatus,     qos=QOS_NAV))
        self._w_grasp_st    = DataWriter(pub, Topic(self._dp, "rt/grasp/status",     GraspStatus,   qos=QOS_GRASP))
        self._w_vision_obj  = DataWriter(pub, Topic(self._dp, "rt/vision/objects",   VisionObjects, qos=QOS_VISION))
        self._w_vision_per  = DataWriter(pub, Topic(self._dp, "rt/vision/persons",   VisionPersons, qos=QOS_VISION))
        self._w_slam_locs   = DataWriter(pub, Topic(self._dp, "rt/slam/locations",   Locations,     qos=QOS_SLAM_MAP))
        self._w_state       = DataWriter(pub, Topic(self._dp, "rt/orchestration/state", OrchestratorState, qos=QOS_ORCHESTRATION))
        
        # PONTE (RELAY): Escuta o teste e escreve para o robô no mesmo tópico
        topic_cmd = Topic(self._dp, "rt/motion/cmd_vel", CmdVel, qos=QOS_MOTION)
        self._r_cmd_vel_in  = DataReader(sub, topic_cmd)  # Entrada do teste_motion
        self._w_cmd_vel_out = DataWriter(pub, topic_cmd) # Saída para o main_motion

        # ── Readers (monitorização) ────────────────────────────────────────
        self._r_orch_state  = DataReader(sub, Topic(self._dp, "rt/orchestration/state", OrchestratorState, qos=QOS_ORCHESTRATION))
        self._r_odometry    = DataReader(sub, Topic(self._dp, "rt/motion/odometry",     OdometryMsg,       qos=QOS_ODOMETRY))

        # ── Estado partilhado ─────────────────────────────────────────────
        self._current_phase = Phase.IDLE
        self._phase_lock    = threading.Lock()

        log.info("NavSim inicializado com função RELAY de comandos.")

    # ── Utilitários ───────────────────────────────────────────────────────────

    def _header(self) -> Header:
        self._seq += 1
        return Header(timestamp_ns=time.time_ns(), frame_id="nav_sim", seq=self._seq)

    def _publish_current_state(self):
        """Publica o estado da orquestração com os bits de ativação corretos."""
        # Define quais módulos estão ativos com base na fase atual
        motion_on = self._current_phase in [Phase.NAVIGATING_TO_TABLE, Phase.NAVIGATING_TO_PERSON, Phase.RECOVERING]
        
        msg = OrchestratorState(
            header=self._header(),
            phase=self._current_phase,
            active_modules=ActiveModules(
                motion=motion_on,
                navigation=True,
                vision_objects=(self._current_phase == Phase.LOCATING_OBJECT)
            )
        )
        self._w_state.write(msg)

    def _relay_cmd_vel(self):
        """Função que repassa a velocidade do teste para o robô se a fase permitir."""
        samples = self._r_cmd_vel_in.take()
        for s in samples:
            # Só republica para o robô se estivermos numa fase de movimento
            if self._current_phase in [Phase.NAVIGATING_TO_TABLE, Phase.NAVIGATING_TO_PERSON]:
                self._w_cmd_vel_out.write(s)

    def _wait_for_phase(self, phase: Phase, timeout: float = PHASE_TIMEOUT) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self._phase_lock:
                if self._current_phase == phase:
                    return True
            time.sleep(0.05)
        return False

    # ── Threads de fundo ──────────────────────────────────────────────────────

    def _monitor_thread(self):
        """Monitoriza o estado e executa a ponte de comandos."""
        while True:
            # 1. Atualiza fase interna a partir do DDS
            for s in self._r_orch_state.take():
                with self._phase_lock:
                    if s.phase != self._current_phase:
                        log.info(f"Nova Fase Detetada: {s.phase.name}")
                    self._current_phase = s.phase

            # 2. Executa a ponte de velocidade
            self._relay_cmd_vel()
            
            # 3. Publica o estado para garantir que o main_motion recebe a ativação
            self._publish_current_state()
            
            time.sleep(0.02) # 50Hz

    # ── Cenário ───────────────────────────────────────────────────────────────

    def run(self):
        # Iniciar thread de monitorização e relay
        threading.Thread(target=self._monitor_thread, daemon=True).start()

        log.info("\n[PASSO 1] A publicar Locations + Intent...")
        self._w_slam_locs.write(Locations(header=self._header(), locations=[
            Location(name="table", pose=Pose(position=Vector3(x=2.0))),
            Location(name=ALVO_PESSOA, pose=Pose(position=Vector3(y=3.0)))
        ]))
        time.sleep(0.5)
        self._w_intent.write(Intent(header=self._header(), acao=Acao.RECOLHER, alvo=ALVO_OBJETO))

        log.info("\n[PASSO 2] A aguardar LOCATING_OBJECT...")
        if self._wait_for_phase(Phase.LOCATING_OBJECT):
            time.sleep(1)
            self._w_vision_obj.write(VisionObjects(header=self._header(), detections=[
                ObjectDetection(name=ALVO_OBJETO, confidence=0.99)
            ]))

        log.info("\n[PASSO 3] A entrar em NAVIGATING_TO_TABLE. O teste_motion já pode atuar!")
        # A partir daqui, se correres o teste_motion.py, o robô irá mexer-se.
        
        while True:
            time.sleep(1)

if __name__ == "__main__":
    try:
        NavSim().run()
    except KeyboardInterrupt:
        log.info("Encerrando.")