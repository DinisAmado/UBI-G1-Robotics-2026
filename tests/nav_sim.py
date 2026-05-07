#!/usr/bin/env python3
"""
nav_sim.py — Simulador de SLAM + Navegação para teste isolado.

Simula o fluxo completo que o orquestrador precisa para avançar,
publicando dados fixos/pré-definidos nos tópicos corretos.

Fluxo simulado:
  1. Publica Intent (RECOLHER, alvo='bola de tenis')
  2. Publica VisionObjects (objeto encontrado)
  3. Publica Locations (mesa e pessoa no mapa)
  4. Publica VisionPersons (pessoa detetada)
  5. Fase NAVIGATING_TO_TABLE:
       → publica CmdVel durante NAV_DURATION segundos
       → publica NavStatus DONE
  6. Publica GraspStatus DONE (simula grasping)
  7. Fase NAVIGATING_TO_PERSON:
       → publica CmdVel durante NAV_DURATION segundos
       → publica NavStatus DONE
  8. Publica GraspStatus DONE (simula entrega)
  9. Monitoriza rt/motion/odometry e rt/orchestration/state em tempo real

Uso:
  python nav_sim.py
"""

import os
import sys
import time
import logging
import threading

# Caminho para idl_ri e qos_profiles
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from cyclonedds.domain import DomainParticipant
from cyclonedds.topic import Topic
from cyclonedds.pub import Publisher, DataWriter
from cyclonedds.sub import Subscriber, DataReader

from idl_ri import (
    Header, Status, Image,
    Intent, Acao,
    OrchestratorState, Phase,
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

DOMAIN_ID    = 0
ALVO_OBJETO  = "bola de tenis"
ALVO_PESSOA  = "person_01"
NAV_DURATION = 5.0     # segundos a publicar CmdVel antes de dizer DONE
CMD_VEL_HZ   = 50      # frequência de publicação de CmdVel

# Velocidades fixas de navegação para a mesa e para a pessoa
CMD_TO_TABLE  = (0.3, 0.0, 0.0)   # vx, vy, wz
CMD_TO_PERSON = (0.2, 0.0, 0.15)  # ligeiramente a virar


# ─── Simulador ────────────────────────────────────────────────────────────────

class NavSim:

    def __init__(self):
        self._seq  = 0
        self._dp   = DomainParticipant(DOMAIN_ID)
        pub        = Publisher(self._dp)
        sub        = Subscriber(self._dp)

        # ── Writers ────────────────────────────────────────────────────────
        self._w_intent      = DataWriter(pub, Topic(self._dp, "rt/hmi/intent",       Intent,       qos=QOS_HMI))
        self._w_nav_status  = DataWriter(pub, Topic(self._dp, "rt/nav/status",       NavStatus,    qos=QOS_NAV))
        self._w_grasp_st    = DataWriter(pub, Topic(self._dp, "rt/grasp/status",     GraspStatus,  qos=QOS_GRASP))
        self._w_vision_obj  = DataWriter(pub, Topic(self._dp, "rt/vision/objects",   VisionObjects, qos=QOS_VISION))
        self._w_vision_per  = DataWriter(pub, Topic(self._dp, "rt/vision/persons",   VisionPersons, qos=QOS_VISION))
        self._w_slam_locs   = DataWriter(pub, Topic(self._dp, "rt/slam/locations",   Locations,    qos=QOS_SLAM_MAP))
        self._w_cmd_vel     = DataWriter(pub, Topic(self._dp, "rt/motion/cmd_vel",   CmdVel,       qos=QOS_MOTION))

        # ── Readers (monitorização) ────────────────────────────────────────
        self._r_orch_state  = DataReader(sub, Topic(self._dp, "rt/orchestration/state",  OrchestratorState, qos=QOS_ORCHESTRATION))
        self._r_odometry    = DataReader(sub, Topic(self._dp, "rt/motion/odometry",      OdometryMsg,       qos=QOS_ODOMETRY))

        # Estado interno
        self._current_phase = Phase.IDLE
        self._stop_cmd_vel  = threading.Event()

        log.info("NavSim inicializado no domínio %d", DOMAIN_ID)

    def _header(self) -> Header:
        self._seq += 1
        return Header(timestamp_ns=time.time_ns(), frame_id="nav_sim", seq=self._seq)

    # ── Publicações de dados fixos ─────────────────────────────────────────────

    def _pub_locations(self) -> None:
        """Publica localizações fixas no mapa: mesa e pessoa."""
        locs = Locations(
            header=self._header(),
            locations=[
                Location(name="table",    pose=Pose(position=Vector3(x=2.0, y=0.0, z=0.0))),
                Location(name=ALVO_PESSOA, pose=Pose(position=Vector3(x=0.0, y=3.0, z=0.0))),
            ],
        )
        self._w_slam_locs.write(locs)
        log.info("Locations publicadas: table=(2,0) | %s=(0,3)", ALVO_PESSOA)

    def _pub_vision_objects(self) -> None:
        """Publica objeto detetado com alta confiança."""
        msg = VisionObjects(
            header=self._header(),
            detections=[
                ObjectDetection(name=ALVO_OBJETO, confidence=0.95, image=Image()),
            ],
        )
        self._w_vision_obj.write(msg)
        log.info("VisionObjects: '%s' detetado (conf=0.95)", ALVO_OBJETO)

    def _pub_vision_persons(self) -> None:
        """Publica pessoa detetada com movimento de lábios."""
        msg = VisionPersons(
            header=self._header(),
            detections=[
                PersonDetection(id=ALVO_PESSOA, lip_movement_confidence=0.85),
            ],
        )
        self._w_vision_per.write(msg)
        log.info("VisionPersons: '%s' detetado (lip_conf=0.85)", ALVO_PESSOA)

    def _pub_intent(self) -> None:
        """Publica intent de RECOLHER o objeto alvo."""
        msg = Intent(
            header=self._header(),
            acao=Acao.RECOLHER,
            alvo=ALVO_OBJETO,
            comando_grasping="pega",
        )
        self._w_intent.write(msg)
        log.info("Intent publicado: RECOLHER '%s'", ALVO_OBJETO)

    def _pub_nav_done(self) -> None:
        self._w_nav_status.write(NavStatus(
            header=self._header(), status=Status.DONE, reason="destino atingido", progress=1.0,
        ))
        log.info("NavStatus: DONE")

    def _pub_grasp_done(self) -> None:
        self._w_grasp_st.write(GraspStatus(
            header=self._header(), status=Status.DONE, reason="", progress=1.0,
        ))
        log.info("GraspStatus: DONE")

    # ── CmdVel em thread ──────────────────────────────────────────────────────

    def _loop_cmd_vel(self, vx: float, vy: float, wz: float) -> None:
        """Publica CmdVel a CMD_VEL_HZ Hz até _stop_cmd_vel ser ativado."""
        sleep_s = 1.0 / CMD_VEL_HZ
        count   = 0
        while not self._stop_cmd_vel.is_set():
            self._w_cmd_vel.write(CmdVel(
                header=self._header(), vx=vx, vy=vy, wz=wz,
            ))
            count += 1
            if count % CMD_VEL_HZ == 0:
                log.debug("CmdVel a publicar: vx=%.2f vy=%.2f wz=%.2f", vx, vy, wz)
            time.sleep(sleep_s)

        # Parar o robô ao terminar
        self._w_cmd_vel.write(CmdVel(header=self._header(), vx=0.0, vy=0.0, wz=0.0))
        log.info("CmdVel: parado (0, 0, 0)")

    def _navegar(self, destino: str, vx: float, vy: float, wz: float) -> None:
        """Inicia publicação de CmdVel, aguarda NAV_DURATION, publica DONE."""
        log.info("--- A navegar para '%s' durante %.1fs ---", destino, NAV_DURATION)
        self._stop_cmd_vel.clear()
        t = threading.Thread(target=self._loop_cmd_vel, args=(vx, vy, wz), daemon=True)
        t.start()
        time.sleep(NAV_DURATION)
        self._stop_cmd_vel.set()
        t.join(timeout=1.0)
        time.sleep(0.1)
        self._pub_nav_done()

    # ── Monitorização ─────────────────────────────────────────────────────────

    def _monitor(self) -> None:
        """Thread que imprime o estado do orquestrador e odometria recebida."""
        last_phase = None
        while True:
            for s in self._r_orch_state.take():
                if s and s.phase != last_phase:
                    log.info("[ORCH] Fase: %s | motion=%s | nav=%s",
                             s.phase.name,
                             "ON" if s.active_modules.motion else "off",
                             "ON" if s.active_modules.navigation else "off")
                    last_phase = s.phase

            for s in self._r_odometry.take():
                if s:
                    log.info("[ODOM] pos=(%.2f, %.2f) vx=%.2f vy=%.2f wz=%.2f",
                             s.pose.position.x, s.pose.position.y,
                             s.vx, s.vy, s.wz)
            time.sleep(0.2)

    # ── Cenário de teste ──────────────────────────────────────────────────────

    def run(self) -> None:
        log.info("=" * 55)
        log.info("  NAV SIM — Teste Orquestrador + Movimentação")
        log.info("=" * 55)

        # Iniciar thread de monitorização
        threading.Thread(target=self._monitor, daemon=True).start()

        # 1. Publicar dados de contexto imediatamente
        log.info("\n[PASSO 1] A publicar localizações SLAM...")
        self._pub_locations()
        time.sleep(0.5)

        # 2. Intent — arrancar a missão
        log.info("\n[PASSO 2] A publicar Intent (RECOLHER '%s')...", ALVO_OBJETO)
        self._pub_intent()
        time.sleep(1.0)

        # 3. Objeto detetado (fase LOCATING_OBJECT)
        log.info("\n[PASSO 3] A publicar VisionObjects...")
        for _ in range(5):   # publicar várias vezes para garantir que o orquestrador lê
            self._pub_vision_objects()
            time.sleep(0.2)

        # 4. Pessoa detetada — necessária durante GRASPING e depois
        log.info("\n[PASSO 4] A publicar VisionPersons (em background)...")
        def _loop_persons():
            while True:
                self._pub_vision_persons()
                time.sleep(1.0)
        threading.Thread(target=_loop_persons, daemon=True).start()

        # 5. Navegação para a mesa (fase NAVIGATING_TO_TABLE)
        log.info("\n[PASSO 5] Fase NAVIGATING_TO_TABLE — a publicar CmdVel...")
        time.sleep(1.0)   # aguardar que o orquestrador entre na fase
        self._navegar("mesa", *CMD_TO_TABLE)

        # 6. Grasping (fase GRASPING_OBJECT)
        log.info("\n[PASSO 6] A simular Grasping (DONE em 3s)...")
        time.sleep(3.0)
        self._pub_grasp_done()

        # 7. Navegação para a pessoa (fase NAVIGATING_TO_PERSON)
        log.info("\n[PASSO 7] Fase NAVIGATING_TO_PERSON — a publicar CmdVel...")
        time.sleep(1.0)
        self._navegar("pessoa", *CMD_TO_PERSON)

        # 8. Entrega (fase DELIVERING)
        log.info("\n[PASSO 8] A simular Entrega (DONE em 3s)...")
        time.sleep(3.0)
        self._pub_grasp_done()

        log.info("\n[FIM] Cenário completo. A monitorizar por mais 10s...")
        time.sleep(10.0)
        log.info("NavSim terminado.")


if __name__ == "__main__":
    try:
        NavSim().run()
    except KeyboardInterrupt:
        log.info("NavSim interrompido.")