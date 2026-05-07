#!/usr/bin/env python3
"""
main_test.py — Orquestrador em modo de teste isolado.

Diferenças em relação ao main.py de produção:
  - TEST_MODE = True  →  timeouts reduzidos para tornar o teste mais rápido
  - Não remove nenhuma lógica — o orquestrador corre normalmente,
    o nav_sim fornece todos os dados em falta (visão, grasping, etc.)
  - Logging mais detalhado para validar cada transição de fase

Para o teste normal de produção usa main.py.
Para este teste isolado (nav_sim + motion) usa este ficheiro.

Correr em terminais separados:
  Terminal 1:  python main_test.py
  Terminal 2:  python nav_sim.py
  Terminal 3:  python main_motion.py
"""

import time
import logging
import threading
from dataclasses import dataclass, field
from typing import Optional

from cyclonedds.domain import DomainParticipant
from cyclonedds.topic import Topic
from cyclonedds.pub import Publisher, DataWriter
from cyclonedds.sub import Subscriber, DataReader

from qos_profiles import (
    QOS_HMI, QOS_ORCHESTRATION, QOS_HEARTBEAT,
    QOS_VISION, QOS_SLAM_MAP,
    QOS_NAV, QOS_GRASP,
)

from idl_ri import (
    Header, Status, Image,
    Intent, Acao, Feedback, OrchestrationState as HmiState,
    OrchestratorState, ActiveModules, Phase, Heartbeat,
    NavGoal as Goal, GoalType, GoalData, NavStatusMsg as NavStatus,
    GraspCommand, GraspStatusMsg as GraspStatus, Posture,
    Objects as VisionObjects, Persons as VisionPersons,
    Locations,
)


# ─── Logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.DEBUG,   # DEBUG em modo de teste para ver tudo
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("orchestrator-test")


# ─── Constantes ──────────────────────────────────────────────────────────────

DOMAIN_ID        = 0
MAX_RETRIES      = 3
LOOP_HZ          = 20
VISION_MIN_CONF  = 0.6

# ── Timeouts reduzidos para o teste ──────────────────────────────────────────
# Em produção (main.py) os valores são maiores.
# Aqui são curtos para o teste não ficar à espera desnecessariamente.
PHASE_TIMEOUTS: dict[Phase, float] = {
    Phase.LOCATING_OBJECT:      8.0,    # nav_sim publica VisionObjects em ~1s
    Phase.NAVIGATING_TO_TABLE:  15.0,   # nav_sim navega por 5s + margem
    Phase.GRASPING_OBJECT:      8.0,    # nav_sim publica GraspStatus DONE em 3s
    Phase.NAVIGATING_TO_PERSON: 15.0,
    Phase.DELIVERING:           8.0,
}

TABLE_LOCATION_NAME   = "table"
LIP_MOVEMENT_MIN_CONF = 0.5


# ─── Mapeamento Fase → Módulos Activos ───────────────────────────────────────

PHASE_MODULES: dict[Phase, ActiveModules] = {
    Phase.IDLE: ActiveModules(
        vision_objects=False, vision_persons=False,
        navigation=False, grasping=False, motion=False,
    ),
    Phase.WAITING_FOR_INTENT: ActiveModules(
        vision_objects=False, vision_persons=False,
        navigation=False, grasping=False, motion=False,
    ),
    Phase.LOCATING_OBJECT: ActiveModules(
        vision_objects=True,  vision_persons=False,
        navigation=False, grasping=False, motion=False,
    ),
    Phase.NAVIGATING_TO_TABLE: ActiveModules(
        vision_objects=True,  vision_persons=False,
        navigation=True,  grasping=False, motion=True,
    ),
    Phase.GRASPING_OBJECT: ActiveModules(
        vision_objects=False, vision_persons=True,
        navigation=False, grasping=True,  motion=False,
    ),
    Phase.NAVIGATING_TO_PERSON: ActiveModules(
        vision_objects=False, vision_persons=True,
        navigation=True,  grasping=False, motion=True,
    ),
    Phase.DELIVERING: ActiveModules(
        vision_objects=False, vision_persons=True,
        navigation=False, grasping=True,  motion=False,
    ),
    Phase.RECOVERING: ActiveModules(
        vision_objects=False, vision_persons=False,
        navigation=False, grasping=False, motion=False,
    ),
    Phase.ABORTED: ActiveModules(
        vision_objects=False, vision_persons=False,
        navigation=False, grasping=False, motion=False,
    ),
}


# ─── Contexto interno ────────────────────────────────────────────────────────

@dataclass
class OrchestratorContext:
    current_intent:    Optional[Intent]    = None
    last_object_image: Optional[Image]     = None
    last_object_name:  str                 = ""
    last_person_id:    str                 = ""
    known_locations:   Optional[Locations] = None
    retry_counts: dict = field(default_factory=lambda: {p: 0 for p in Phase})


# ─── Orquestrador ────────────────────────────────────────────────────────────

class Orchestrator:

    def __init__(self):
        self._ctx   = OrchestratorContext()
        self._phase = Phase.IDLE
        self._lock  = threading.Lock()
        self._seq   = 0

        self._phase_start_time: float           = time.time()
        self._recover_until:    Optional[float] = None
        self._abort_until:      Optional[float] = None

        self._dp = DomainParticipant(DOMAIN_ID)
        pub      = Publisher(self._dp)
        sub      = Subscriber(self._dp)

        # Topics
        t_orch_state   = Topic(self._dp, "rt/orchestration/state",     OrchestratorState, qos=QOS_ORCHESTRATION)
        t_orch_hb      = Topic(self._dp, "rt/orchestration/heartbeat", Heartbeat,         qos=QOS_HEARTBEAT)
        t_hmi_intent   = Topic(self._dp, "rt/hmi/intent",              Intent,            qos=QOS_HMI)
        t_hmi_feedback = Topic(self._dp, "rt/hmi/feedback",            Feedback,          qos=QOS_HMI)
        t_nav_goal     = Topic(self._dp, "rt/nav/goal",                Goal,              qos=QOS_NAV)
        t_nav_status   = Topic(self._dp, "rt/nav/status",              NavStatus,         qos=QOS_NAV)
        t_grasp_cmd    = Topic(self._dp, "rt/grasp/command",           GraspCommand,      qos=QOS_GRASP)
        t_grasp_status = Topic(self._dp, "rt/grasp/status",            GraspStatus,       qos=QOS_GRASP)
        t_vision_obj   = Topic(self._dp, "rt/vision/objects",          VisionObjects,     qos=QOS_VISION)
        t_vision_per   = Topic(self._dp, "rt/vision/persons",          VisionPersons,     qos=QOS_VISION)
        t_slam_locs    = Topic(self._dp, "rt/slam/locations",          Locations,         qos=QOS_SLAM_MAP)

        # Writers
        self._w_orch_state   = DataWriter(pub, t_orch_state)
        self._w_orch_hb      = DataWriter(pub, t_orch_hb)
        self._w_hmi_feedback = DataWriter(pub, t_hmi_feedback)
        self._w_nav_goal     = DataWriter(pub, t_nav_goal)
        self._w_grasp_cmd    = DataWriter(pub, t_grasp_cmd)

        # Readers
        self._r_hmi_intent   = DataReader(sub, t_hmi_intent)
        self._r_nav_status   = DataReader(sub, t_nav_status)
        self._r_grasp_status = DataReader(sub, t_grasp_status)
        self._r_vision_obj   = DataReader(sub, t_vision_obj)
        self._r_vision_per   = DataReader(sub, t_vision_per)
        self._r_slam_locs    = DataReader(sub, t_slam_locs)

        log.info("Orquestrador TEST inicializado no domínio %d", DOMAIN_ID)

    # ── Máquina de estados ────────────────────────────────────────────────────

    def _transition(self, new_phase: Phase, reason: str = "") -> None:
        with self._lock:
            old_phase              = self._phase
            self._phase            = new_phase
            self._phase_start_time = time.time()

        log.info("")
        log.info("  ╔══ TRANSIÇÃO ══════════════════════════════")
        log.info("  ║  %s → %s", old_phase.name, new_phase.name)
        log.info("  ║  Razão: %s", reason)
        mods = PHASE_MODULES[new_phase]
        log.info("  ║  Módulos: motion=%s nav=%s grasping=%s vision_obj=%s vision_per=%s",
                 "ON" if mods.motion     else "off",
                 "ON" if mods.navigation else "off",
                 "ON" if mods.grasping   else "off",
                 "ON" if mods.vision_objects else "off",
                 "ON" if mods.vision_persons else "off")
        log.info("  ╚═══════════════════════════════════════════")
        log.info("")

        self._w_orch_state.write(OrchestratorState(
            header=self._make_header(),
            phase=new_phase,
            active_modules=PHASE_MODULES[new_phase],
            current_target_object=self._ctx.last_object_name,
            current_target_person=self._ctx.last_person_id,
            reason=reason,
        ))
        self._publish_hmi_feedback(new_phase, reason)

    def _check_timeout(self) -> None:
        timeout = PHASE_TIMEOUTS.get(self._phase)
        if timeout is None:
            return
        elapsed = time.time() - self._phase_start_time
        if elapsed > timeout:
            log.warning("Timeout na fase %s (%.1fs > %.1fs)",
                        self._phase.name, elapsed, timeout)
            self._handle_retry(self._phase, "timeout")

    def _step(self) -> None:
        self._check_timeout()
        phase = self._phase

        if phase == Phase.IDLE:
            self._transition(Phase.WAITING_FOR_INTENT, "pronto")
        elif phase == Phase.WAITING_FOR_INTENT:
            self._handle_waiting_for_intent()
        elif phase == Phase.LOCATING_OBJECT:
            self._handle_locating_object()
        elif phase == Phase.NAVIGATING_TO_TABLE:
            self._handle_nav_to_table()
        elif phase == Phase.GRASPING_OBJECT:
            self._handle_grasp_status()
        elif phase == Phase.NAVIGATING_TO_PERSON:
            self._handle_nav_to_person()
        elif phase == Phase.DELIVERING:
            self._handle_delivering()
        elif phase == Phase.RECOVERING:
            self._handle_recovering()
        elif phase == Phase.ABORTED:
            self._handle_aborted()

    # ── Handlers ──────────────────────────────────────────────────────────────

    def _handle_waiting_for_intent(self) -> None:
        sample = self._read_one(self._r_hmi_intent)
        if sample is None:
            return
        self._ctx.current_intent   = sample
        self._ctx.last_object_name = sample.alvo
        self._ctx.last_object_image = None
        log.info("Intent recebido → acao=%s  alvo='%s'", sample.acao.name, sample.alvo)

        if sample.acao in (Acao.ENTREGAR, Acao.RECOLHER):
            self._transition(Phase.LOCATING_OBJECT, f"à procura de '{sample.alvo}'")
        elif sample.acao == Acao.LARGA:
            self._w_grasp_cmd.write(GraspCommand(
                header=self._make_header(), objeto=sample.alvo,
                objeto_id="drop", image=Image(), postura=Posture.NEUTRAL,
            ))
            self._transition(Phase.DELIVERING, "a largar objeto")
        elif sample.acao == Acao.SEGUIR:
            self._transition(Phase.NAVIGATING_TO_PERSON, "a seguir pessoa")
        elif sample.acao == Acao.PARAR:
            self._transition(Phase.IDLE, "paragem solicitada")

    def _handle_locating_object(self) -> None:
        sample = self._read_one(self._r_vision_obj)
        if sample is None:
            return
        for det in sample.detections:
            if det.name == self._ctx.last_object_name and det.confidence >= VISION_MIN_CONF:
                self._ctx.last_object_image = det.image
                log.info("Objeto '%s' localizado (conf=%.2f)", det.name, det.confidence)
                self._w_nav_goal.write(Goal(
                    header=self._make_header(),
                    data=GoalData(name=TABLE_LOCATION_NAME),
                ))
                self._transition(Phase.NAVIGATING_TO_TABLE,
                                 f"a navegar para '{TABLE_LOCATION_NAME}'")
                return

    def _handle_nav_to_table(self) -> None:
        sample = self._read_one(self._r_nav_status)
        if sample is None:
            return
        log.debug("NavStatus recebido: %s", sample.status.name)
        if sample.status == Status.DONE:
            image = self._ctx.last_object_image or Image()
            self._w_grasp_cmd.write(GraspCommand(
                header=self._make_header(),
                objeto=self._ctx.last_object_name,
                objeto_id="", image=image,
                postura=Posture.EXTEND_ARM_FORWARD,
            ))
            log.info("GraspCommand enviado para '%s'", self._ctx.last_object_name)
            self._transition(Phase.GRASPING_OBJECT, "chegou à mesa")
        elif sample.status == Status.FAILED:
            self._handle_retry(self._phase, sample.reason)

    def _handle_grasp_status(self) -> None:
        sample = self._read_one(self._r_grasp_status)
        if sample is None:
            return
        log.debug("GraspStatus recebido: %s", sample.status.name)
        if sample.status == Status.DONE:
            if self._phase == Phase.GRASPING_OBJECT:
                self._w_grasp_cmd.write(GraspCommand(
                    header=self._make_header(),
                    objeto=self._ctx.last_object_name,
                    objeto_id="carry", image=Image(),
                    postura=Posture.NEUTRAL,
                ))
                goal = self._build_person_goal()
                if goal:
                    self._w_nav_goal.write(goal)
                    self._transition(Phase.NAVIGATING_TO_PERSON,
                                     f"a navegar para '{self._ctx.last_person_id}'")
                else:
                    log.error("Pessoa não encontrada no SLAM.")
                    self._handle_retry(Phase.NAVIGATING_TO_PERSON, "pessoa não encontrada")
            elif self._phase == Phase.DELIVERING:
                log.info("Entrega concluída com sucesso!")
                self._ctx.retry_counts = {p: 0 for p in Phase}
                self._transition(Phase.IDLE, "tarefa concluída")
        elif sample.status == Status.FAILED:
            self._handle_retry(self._phase, sample.reason)

    def _handle_nav_to_person(self) -> None:
        sample = self._read_one(self._r_nav_status)
        if sample is None:
            return
        log.debug("NavStatus recebido: %s", sample.status.name)
        if sample.status == Status.DONE:
            self._w_grasp_cmd.write(GraspCommand(
                header=self._make_header(),
                objeto=self._ctx.last_object_name,
                objeto_id="deliver", image=Image(),
                postura=Posture.EXTEND_ARM_FORWARD,
            ))
            log.info("A entregar '%s' à pessoa '%s'",
                     self._ctx.last_object_name, self._ctx.last_person_id)
            self._transition(Phase.DELIVERING, "chegou à pessoa")
        elif sample.status == Status.FAILED:
            self._handle_retry(self._phase, sample.reason)

    def _handle_delivering(self) -> None:
        self._handle_grasp_status()

    def _handle_recovering(self) -> None:
        if self._recover_until is None:
            if self._ctx.last_object_image is not None:
                log.warning("A largar objeto antes de recuperar...")
                self._w_grasp_cmd.write(GraspCommand(
                    header=self._make_header(), objeto="",
                    objeto_id="drop", image=Image(), postura=Posture.NEUTRAL,
                ))
            self._recover_until = time.time() + 3.0
            log.warning("A recuperar — aguarda 3 s...")
            return
        if time.time() >= self._recover_until:
            self._recover_until = None
            self._ctx.last_object_image = None
            self._transition(Phase.LOCATING_OBJECT, "a tentar novamente")

    def _handle_aborted(self) -> None:
        if self._abort_until is None:
            self._abort_until = time.time() + 2.0
            log.warning("Abortado. A aguardar novo intent.")
            return
        if time.time() >= self._abort_until:
            self._abort_until = None
            self._transition(Phase.WAITING_FOR_INTENT, "pronto para nova tarefa")

    # ── Retry ─────────────────────────────────────────────────────────────────

    def _handle_retry(self, failed_phase: Phase, reason: str) -> None:
        self._ctx.retry_counts[failed_phase] += 1
        attempts = self._ctx.retry_counts[failed_phase]
        if attempts <= MAX_RETRIES:
            log.warning("Fase %s falhou: '%s'. Tentativa %d/%d",
                        failed_phase.name, reason, attempts, MAX_RETRIES)
            self._transition(Phase.RECOVERING, f"retry {attempts}/{MAX_RETRIES} — {reason}")
        else:
            log.error("Fase %s falhou %d vezes. ABORTED.", failed_phase.name, MAX_RETRIES)
            self._ctx.retry_counts[failed_phase] = 0
            self._transition(Phase.ABORTED, f"max retries — {reason}")

    # ── Utilitários ───────────────────────────────────────────────────────────

    def _read_one(self, reader) -> Optional[object]:
        samples = reader.take(1)
        return samples[0] if samples else None

    def _build_person_goal(self) -> Optional[Goal]:
        if self._ctx.known_locations:
            for loc in self._ctx.known_locations.locations:
                if loc.name == self._ctx.last_person_id:
                    return Goal(
                        header=self._make_header(),
                        data=GoalData(name=loc.name),
                    )
        log.warning("Pessoa '%s' não encontrada nas localizações SLAM.",
                    self._ctx.last_person_id)
        return None

    def _make_header(self) -> Header:
        with self._lock:
            self._seq += 1
            seq = self._seq
        return Header(timestamp_ns=time.time_ns(), frame_id="orchestrator", seq=seq)

    def _publish_hmi_feedback(self, phase: Phase, message: str) -> None:
        phase_to_hmi = {
            Phase.IDLE:                 HmiState.IDLE,
            Phase.WAITING_FOR_INTENT:   HmiState.WAITING_FOR_INTENT,
            Phase.LOCATING_OBJECT:      HmiState.LOCATING_OBJECT,
            Phase.NAVIGATING_TO_TABLE:  HmiState.NAVIGATING_TO_TABLE,
            Phase.GRASPING_OBJECT:      HmiState.GRASPING_OBJECT,
            Phase.NAVIGATING_TO_PERSON: HmiState.NAVIGATING_TO_PERSON,
            Phase.DELIVERING:           HmiState.DELIVERING,
            Phase.RECOVERING:           HmiState.RECOVERING,
            Phase.ABORTED:              HmiState.ABORTED,
        }
        idle_phases = {Phase.IDLE, Phase.ABORTED}
        self._w_hmi_feedback.write(Feedback(
            header=self._make_header(),
            status=Status.DONE if phase in idle_phases else Status.RUNNING,
            message=message,
            state=phase_to_hmi.get(phase, HmiState.IDLE),
        ))

    def _publish_heartbeat(self) -> None:
        self._w_orch_hb.write(Heartbeat(
            header=self._make_header(),
            module_name="orchestrator",
            ready=True,
            error_msg="",
        ))

    def _poll_slam_locations(self) -> None:
        for sample in self._r_slam_locs.take():
            if sample is not None:
                self._ctx.known_locations = sample
                log.debug("Locations SLAM actualizadas: %d locais",
                          len(sample.locations))

    def _poll_vision_persons(self) -> None:
        for sample in self._r_vision_per.take():
            if sample is None:
                continue
            best = max(
                (d for d in sample.detections
                 if d.lip_movement_confidence >= LIP_MOVEMENT_MIN_CONF),
                key=lambda d: d.lip_movement_confidence,
                default=None,
            )
            if best:
                if best.id != self._ctx.last_person_id:
                    log.info("Pessoa detetada: '%s' (lip_conf=%.2f)",
                             best.id, best.lip_movement_confidence)
                self._ctx.last_person_id = best.id

    # ── Loop principal ────────────────────────────────────────────────────────

    def run(self) -> None:
        log.info("=" * 55)
        log.info("  Orquestrador RT — MODO TESTE")
        log.info("  Aguarda: nav_sim.py + main_motion.py")
        log.info("=" * 55)

        self._transition(Phase.IDLE, "arranque")

        heartbeat_counter = 0
        sleep_s = 1.0 / LOOP_HZ

        try:
            while True:
                self._poll_slam_locations()
                self._poll_vision_persons()
                self._step()

                heartbeat_counter += 1
                if heartbeat_counter >= LOOP_HZ:
                    self._publish_heartbeat()
                    heartbeat_counter = 0

                time.sleep(sleep_s)

        except KeyboardInterrupt:
            log.info("Interrompido.")
        finally:
            self._transition(Phase.IDLE, "shutdown")
            log.info("Orquestrador terminado.")


if __name__ == "__main__":
    Orchestrator().run()