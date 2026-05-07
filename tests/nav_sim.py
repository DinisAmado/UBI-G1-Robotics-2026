#!/usr/bin/env python3
"""
nav_sim.py — Simulador de SLAM + Navegação para teste isolado.

Reativo às fases reais do orquestrador: cada ação é disparada
quando o orquestrador entra na fase correspondente, não por
temporizadores fixos.

Fluxo simulado:
  1. Publica Locations + Intent
  2. Fase LOCATING_OBJECT      → publica VisionObjects
  3. Fase NAVIGATING_TO_TABLE  → publica CmdVel + NavStatus DONE
  4. Fase GRASPING_OBJECT      → publica GraspStatus DONE (após 3 s)
  5. Fase NAVIGATING_TO_PERSON → publica CmdVel + NavStatus DONE
  6. Fase DELIVERING           → publica GraspStatus DONE (após 3 s)

Uso:
  python nav_sim.py
"""

import os
import sys
import time
import logging
import threading

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

DOMAIN_ID     = 0
ALVO_OBJETO   = "bola de tenis"
ALVO_PESSOA   = "person_01"
NAV_DURATION  = 2.0    # segundos a publicar CmdVel antes de dizer DONE
CMD_VEL_HZ    = 50
PHASE_TIMEOUT = 30.0   # timeout máximo a aguardar cada fase (segundos)

CMD_TO_TABLE  = (0.3, 0.0, 0.0)
CMD_TO_PERSON = (0.2, 0.0, 0.15)


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
        self._w_cmd_vel     = DataWriter(pub, Topic(self._dp, "rt/motion/cmd_vel",   CmdVel,        qos=QOS_MOTION))

        # ── Readers (monitorização) ────────────────────────────────────────
        self._r_orch_state  = DataReader(sub, Topic(self._dp, "rt/orchestration/state", OrchestratorState, qos=QOS_ORCHESTRATION))
        self._r_odometry    = DataReader(sub, Topic(self._dp, "rt/motion/odometry",     OdometryMsg,       qos=QOS_ODOMETRY))

        # ── Estado partilhado entre threads ───────────────────────────────
        self._current_phase = Phase.IDLE
        self._phase_lock    = threading.Lock()
        self._stop_cmd_vel  = threading.Event()

        log.info("NavSim inicializado no domínio %d", DOMAIN_ID)

    # ── Utilitários ───────────────────────────────────────────────────────────

    def _header(self) -> Header:
        self._seq += 1
        return Header(timestamp_ns=time.time_ns(), frame_id="nav_sim", seq=self._seq)

    def _wait_for_phase(self, phase: Phase, timeout: float = PHASE_TIMEOUT) -> bool:
        """Bloqueia até o orquestrador entrar na fase indicada ou atingir timeout."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self._phase_lock:
                if self._current_phase == phase:
                    return True
            time.sleep(0.05)
        log.error("TIMEOUT a aguardar fase %s (%.0fs esgotados)", phase.name, timeout)
        return False

    # ── Publicações ───────────────────────────────────────────────────────────

    def _pub_locations(self) -> None:
        locs = Locations(
            header=self._header(),
            locations=[
                Location(name="table",     pose=Pose(position=Vector3(x=2.0, y=0.0, z=0.0))),
                Location(name=ALVO_PESSOA, pose=Pose(position=Vector3(x=0.0, y=3.0, z=0.0))),
            ],
        )
        self._w_slam_locs.write(locs)
        log.info("Locations publicadas: table=(2,0) | %s=(0,3)", ALVO_PESSOA)

    def _pub_vision_objects(self) -> None:
        msg = VisionObjects(
            header=self._header(),
            detections=[
                ObjectDetection(name=ALVO_OBJETO, confidence=0.95, image=Image()),
            ],
        )
        self._w_vision_obj.write(msg)
        log.info("VisionObjects: '%s' detetado (conf=0.95)", ALVO_OBJETO)

    def _pub_vision_persons(self) -> None:
        msg = VisionPersons(
            header=self._header(),
            detections=[
                PersonDetection(id=ALVO_PESSOA, lip_movement_confidence=0.85),
            ],
        )
        self._w_vision_per.write(msg)

    def _pub_intent(self) -> None:
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
            header=self._header(), status=Status.DONE,
            reason="destino atingido", progress=1.0,
        ))
        log.info("NavStatus: DONE")

    def _pub_grasp_done(self) -> None:
        self._w_grasp_st.write(GraspStatus(
            header=self._header(), status=Status.DONE,
            reason="", progress=1.0,
        ))
        log.info("GraspStatus: DONE")

    # ── CmdVel em thread ──────────────────────────────────────────────────────

    def _loop_cmd_vel(self, vx: float, vy: float, wz: float) -> None:
        """Publica CmdVel a CMD_VEL_HZ Hz até _stop_cmd_vel ser ativado."""
        sleep_s = 1.0 / CMD_VEL_HZ
        count   = 0
        while not self._stop_cmd_vel.is_set():
            self._w_cmd_vel.write(CmdVel(header=self._header(), vx=vx, vy=vy, wz=wz))
            count += 1
            if count % CMD_VEL_HZ == 0:
                log.debug("CmdVel: vx=%.2f vy=%.2f wz=%.2f", vx, vy, wz)
            time.sleep(sleep_s)
        # Para o robô ao terminar
        self._w_cmd_vel.write(CmdVel(header=self._header(), vx=0.0, vy=0.0, wz=0.0))
        log.info("CmdVel: parado (0, 0, 0)")

    def _navegar(self, destino: str, vx: float, vy: float, wz: float) -> None:
        """Publica CmdVel durante NAV_DURATION segundos, depois publica NavStatus DONE."""
        log.info("--- A navegar para '%s' durante %.1fs ---", destino, NAV_DURATION)
        self._stop_cmd_vel.clear()
        t = threading.Thread(target=self._loop_cmd_vel, args=(vx, vy, wz), daemon=True)
        t.start()
        time.sleep(NAV_DURATION)
        self._stop_cmd_vel.set()
        t.join(timeout=1.0)
        time.sleep(0.1)
        self._pub_nav_done()

    # ── Threads de fundo ──────────────────────────────────────────────────────

    def _monitor_thread(self) -> None:
        """Atualiza _current_phase com o estado real do orquestrador."""
        last_phase = None
        while True:
            for s in self._r_orch_state.take():
                if s:
                    with self._phase_lock:
                        self._current_phase = s.phase
                    if s.phase != last_phase:
                        log.info("[ORCH] Fase: %s | motion=%s | nav=%s",
                                 s.phase.name,
                                 "ON" if s.active_modules.motion     else "off",
                                 "ON" if s.active_modules.navigation else "off")
                        last_phase = s.phase

            for s in self._r_odometry.take():
                if s:
                    log.debug("[ODOM] pos=(%.2f, %.2f) vx=%.2f vy=%.2f wz=%.2f",
                              s.pose.position.x, s.pose.position.y,
                              s.vx, s.vy, s.wz)
            time.sleep(0.05)

    def _persons_thread(self) -> None:
        """Publica VisionPersons continuamente a 1 Hz."""
        while True:
            self._pub_vision_persons()
            time.sleep(1.0)

    # ── Cenário reativo ───────────────────────────────────────────────────────

    def run(self) -> None:
        log.info("=" * 55)
        log.info("  NAV SIM — Teste Orquestrador + Movimentação")
        log.info("=" * 55)

        # Iniciar threads de fundo
        threading.Thread(target=self._monitor_thread, daemon=True).start()
        threading.Thread(target=self._persons_thread,  daemon=True).start()

        # ── Passo 1: dados iniciais ────────────────────────────────────────
        log.info("\n[PASSO 1] A publicar Locations + Intent...")
        self._pub_locations()
        time.sleep(0.3)
        self._pub_intent()

        # ── Passo 2: aguardar LOCATING_OBJECT → publicar objeto ───────────
        log.info("\n[PASSO 2] A aguardar fase LOCATING_OBJECT...")
        if not self._wait_for_phase(Phase.LOCATING_OBJECT):
            return
        log.info("Orquestrador em LOCATING_OBJECT. A publicar VisionObjects...")
        for _ in range(5):
            self._pub_vision_objects()
            time.sleep(0.2)

        # ── Passo 3: aguardar NAVIGATING_TO_TABLE → CmdVel + DONE ─────────
        log.info("\n[PASSO 3] A aguardar fase NAVIGATING_TO_TABLE...")
        if not self._wait_for_phase(Phase.NAVIGATING_TO_TABLE):
            return
        log.info("Orquestrador em NAVIGATING_TO_TABLE. A navegar para a mesa...")
        self._navegar("mesa", *CMD_TO_TABLE)

        # ── Passo 4: aguardar GRASPING_OBJECT → GraspStatus DONE ──────────
        log.info("\n[PASSO 4] A aguardar fase GRASPING_OBJECT...")
        if not self._wait_for_phase(Phase.GRASPING_OBJECT):
            return
        log.info("Orquestrador em GRASPING_OBJECT. A simular grasping (3s)...")
        time.sleep(3.0)
        self._pub_grasp_done()

        # ── Passo 5: aguardar NAVIGATING_TO_PERSON → CmdVel + DONE ────────
        log.info("\n[PASSO 5] A aguardar fase NAVIGATING_TO_PERSON...")
        if not self._wait_for_phase(Phase.NAVIGATING_TO_PERSON):
            return
        log.info("Orquestrador em NAVIGATING_TO_PERSON. A navegar para a pessoa...")
        self._navegar("pessoa", *CMD_TO_PERSON)

        # ── Passo 6: aguardar DELIVERING → GraspStatus DONE ───────────────
        log.info("\n[PASSO 6] A aguardar fase DELIVERING...")
        if not self._wait_for_phase(Phase.DELIVERING):
            return
        log.info("Orquestrador em DELIVERING. A simular entrega (3s)...")
        time.sleep(3.0)
        self._pub_grasp_done()

        # ── Fim ────────────────────────────────────────────────────────────
        log.info("\n[FIM] Cenário completo. A monitorizar por mais 10s...")
        time.sleep(10.0)
        log.info("NavSim terminado.")


if __name__ == "__main__":
    try:
        NavSim().run()
    except KeyboardInterrupt:
        log.info("NavSim interrompido.")