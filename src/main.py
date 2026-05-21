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
    Header, Status, Pose6DOF,
    Intent, Acao, Feedback, OrchestrationState as HmiState,
    OrchestratorState, ActiveModules, Phase, Heartbeat,
    NavGoal as Goal, GoalType, GoalData, NavStatusMsg as NavStatus,
    GraspCommand, GraspStatusMsg as GraspStatus, Posture,
    Objects as VisionObjects, Persons as VisionPersons,
    Locations,
)


# ─── Logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("orchestrator")


# ─── Constantes ──────────────────────────────────────────────────────────────

DOMAIN_ID        = 0
MAX_RETRIES      = 3      # tentativas por fase antes de ABORTED
LOOP_HZ          = 20     # frequência do loop principal
VISION_MIN_CONF  = 0.6    # confiança mínima para aceitar uma detecção de objecto

# Timeouts por fase (segundos).
# Se uma fase ultrapassar este tempo sem avançar, entra no ciclo de retry.
# Fases sem entrada (IDLE, WAITING_FOR_INTENT, etc.) não têm timeout.
PHASE_TIMEOUTS: dict[Phase, float] = {
    Phase.LOCATING_OBJECT:      10.0,
    Phase.NAVIGATING_TO_TABLE:  20.0,
    Phase.GRASPING_OBJECT:      15.0,
    Phase.NAVIGATING_TO_PERSON: 20.0,
    Phase.DELIVERING:           10.0,
}

TABLE_LOCATION_NAME   = "table"   # nome da localização SLAM para a mesa
LIP_MOVEMENT_MIN_CONF = 0.5       # confiança mínima para deteção de pessoa por lábios


# ─── Mapeamento Fase → Módulos Activos ───────────────────────────────────────
#
# Quando o orquestrador entra numa fase, publica rt/orchestration/state
# com este ActiveModules. Cada módulo verifica o seu próprio campo —
# se for False, ignora dados de entrada e não processa nada.
#
# Nota: o campo 'motion' refere-se ao módulo de locomoção (cmd_vel).
# O braço é sempre controlado pelo módulo de grasping.

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
    # Intent actual recebido do HMI
    current_intent:    Optional[Intent] = None

    # Último objecto detectado pela visão (pose 6-DOF usada pelo grasping)
    last_object_pose: Optional[Pose6DOF] = None
    last_object_name: str                = ""

    # Última pessoa detetada por movimento dos lábios
    last_person_id:    str              = ""

    # Localizações conhecidas do SLAM (actualizadas em background)
    known_locations:   Optional[Locations] = None

    # Contadores de retry por fase (reset ao completar tarefa com sucesso)
    retry_counts: dict = field(default_factory=lambda: {p: 0 for p in Phase})


# ─── Orquestrador ────────────────────────────────────────────────────────────

class Orchestrator:

    def __init__(self):
        self._ctx   = OrchestratorContext()
        self._phase = Phase.IDLE
        self._lock  = threading.Lock()
        self._seq   = 0

        self._phase_start_time: float        = time.time()
        self._recover_until:    Optional[float] = None
        self._abort_until:      Optional[float] = None

        # ── DDS setup ──────────────────────────────────────────────────────
        self._dp = DomainParticipant(DOMAIN_ID)
        pub      = Publisher(self._dp)
        sub      = Subscriber(self._dp)

        # ── Topics ─────────────────────────────────────────────────────────
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

        # ── Writers ────────────────────────────────────────────────────────
        self._w_orch_state   = DataWriter(pub, t_orch_state)
        self._w_orch_hb      = DataWriter(pub, t_orch_hb)
        self._w_hmi_feedback = DataWriter(pub, t_hmi_feedback)
        self._w_nav_goal     = DataWriter(pub, t_nav_goal)
        self._w_grasp_cmd    = DataWriter(pub, t_grasp_cmd)

        # ── Readers ────────────────────────────────────────────────────────
        self._r_hmi_intent   = DataReader(sub, t_hmi_intent)
        self._r_nav_status   = DataReader(sub, t_nav_status)
        self._r_grasp_status = DataReader(sub, t_grasp_status)
        self._r_vision_obj   = DataReader(sub, t_vision_obj)
        self._r_vision_per   = DataReader(sub, t_vision_per)
        self._r_slam_locs    = DataReader(sub, t_slam_locs)

        log.info("Orquestrador inicializado no domínio %d", DOMAIN_ID)

    # ── Máquina de estados ────────────────────────────────────────────────────

    def _transition(self, new_phase: Phase, reason: str = "") -> None:

        with self._lock:
            old_phase              = self._phase
            self._phase            = new_phase
            self._phase_start_time = time.time()

        log.info("%-25s → %-25s  (%s)", old_phase.name, new_phase.name, reason)

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

    # ── Handlers por fase ─────────────────────────────────────────────────────

    def _handle_waiting_for_intent(self) -> None:

        sample = self._read_one(self._r_hmi_intent)
        if sample is None:
            return

        self._ctx.current_intent = sample
        log.info("Intent recebido → acao=%s  alvo='%s'  comando_grasping='%s'",
                 sample.acao.name, sample.alvo, sample.comando_grasping)

        if sample.acao in (Acao.ENTREGAR, Acao.RECOLHER):
            self._ctx.last_object_name  = sample.alvo
            self._ctx.last_object_pose = None
            self._transition(Phase.LOCATING_OBJECT,
                             f"à procura de '{sample.alvo}'")

        elif sample.acao == Acao.LARGA:
            # Largar objeto imediatamente — braço neutro + drop
            self._w_grasp_cmd.write(GraspCommand(
                header=self._make_header(),
                objeto=sample.alvo,
                objeto_id="drop",
                pose=Pose6DOF(),
                postura=Posture.NEUTRAL,
            ))
            self._transition(Phase.DELIVERING, "a largar objeto")

        elif sample.acao == Acao.SEGUIR:
            goal = self._build_person_goal()
            if goal:
                self._w_nav_goal.write(goal)
                self._transition(Phase.NAVIGATING_TO_PERSON,
                                    f"a seguir '{self._ctx.last_person_id}'")
            else:
                log.warning("SEGUIR pedido mas pessoa não identificada ainda — a aguardar.")
                # Não transita — fica em WAITING_FOR_INTENT até a pessoa ser conhecida)

        elif sample.acao == Acao.PARAR:
            self._transition(Phase.IDLE, "paragem solicitada pelo operador")

    def _handle_locating_object(self) -> None:

        sample = self._read_one(self._r_vision_obj)
        if sample is None:
            return

        for det in sample.detections:
            if det.name == self._ctx.last_object_name \
                    and det.confidence >= VISION_MIN_CONF:

                self._ctx.last_object_pose = det.pose
                log.info("Objecto '%s' localizado (conf=%.2f)", det.name, det.confidence)

                self._w_nav_goal.write(self._make_named_goal(TABLE_LOCATION_NAME))
                self._transition(Phase.NAVIGATING_TO_TABLE,
                                 f"a navegar para '{TABLE_LOCATION_NAME}'")
                return

    def _handle_nav_to_table(self) -> None:

        sample = self._read_one(self._r_nav_status)
        if sample is None:
            return

        if sample.status == Status.DONE:
            # Chegou à mesa — pede ao grasping para estender o braço e agarrar
            pose = self._ctx.last_object_pose or Pose6DOF()
            self._w_grasp_cmd.write(GraspCommand(
                header=self._make_header(),
                objeto=self._ctx.last_object_name,
                objeto_id="",
                pose=pose,
                postura=Posture.EXTEND_ARM_FORWARD,
            ))
            log.info("GraspCommand enviado para '%s' (braço estendido)",
                     self._ctx.last_object_name)
            self._transition(Phase.GRASPING_OBJECT, "chegou à mesa")

        elif sample.status == Status.FAILED:
            self._handle_retry(self._phase, sample.reason)

    def _handle_grasp_status(self) -> None:
        """Usado nas fases GRASPING_OBJECT e DELIVERING — ambas lêem rt/grasp/status."""

        sample = self._read_one(self._r_grasp_status)
        if sample is None:
            return

        if sample.status == Status.DONE:

            if self._phase == Phase.GRASPING_OBJECT:
                # Objeto agarrado — braço em postura de transporte (mão fechada)
                self._w_grasp_cmd.write(GraspCommand(
                    header=self._make_header(),
                    objeto=self._ctx.last_object_name,
                    objeto_id="carry",
                    pose=Pose6DOF(),
                    postura=Posture.EXTEND_ARM_FORWARD,
                ))

                # Navega até à pessoa identificada pelos lábios
                goal = self._build_person_goal()
                if goal:
                    self._w_nav_goal.write(goal)
                    self._transition(Phase.NAVIGATING_TO_PERSON,
                                     f"a navegar para '{self._ctx.last_person_id}'"
                                     if self._ctx.last_person_id else "a navegar para pessoa")
                else:
                    log.error("Não foi possível localizar a pessoa nas localizações SLAM.")
                    self._handle_retry(Phase.NAVIGATING_TO_PERSON,
                                       "pessoa não encontrada no SLAM")

            elif self._phase == Phase.DELIVERING:
                # Braço estendido — enviar comando para abrir a mão e retrair o braço
                self._w_grasp_cmd.write(GraspCommand(
                    header=self._make_header(),
                    objeto=self._ctx.last_object_name,
                    objeto_id="drop",
                    pose=Pose6DOF(),
                    postura=Posture.NEUTRAL,
                ))
                log.info("Objeto '%s' entregue — a abrir mão e retrair braço",
                         self._ctx.last_object_name)
                self._ctx.retry_counts = {p: 0 for p in Phase}
                self._transition(Phase.IDLE, "tarefa concluída")

        elif sample.status == Status.FAILED:
            self._handle_retry(self._phase, sample.reason)

    def _handle_nav_to_person(self) -> None:

        sample = self._read_one(self._r_nav_status)
        if sample is None:
            return

        if sample.status == Status.DONE:
            # Chegou à pessoa — pede ao grasping para estender o braço e entregar
            self._w_grasp_cmd.write(GraspCommand(
                header=self._make_header(),
                objeto=self._ctx.last_object_name,
                objeto_id="deliver",
                pose=Pose6DOF(),
                postura=Posture.EXTEND_ARM_FORWARD,
            ))
            log.info("A entregar '%s' à pessoa '%s'",
                     self._ctx.last_object_name, self._ctx.last_person_id)
            self._transition(Phase.DELIVERING, "chegou à pessoa")

        elif sample.status == Status.FAILED:
            self._handle_retry(self._phase, sample.reason)

    def _handle_delivering(self) -> None:
        """DELIVERING aguarda confirmação do grasping — reutiliza _handle_grasp_status."""
        self._handle_grasp_status()

    def _handle_recovering(self) -> None:

        if self._recover_until is None:
            # Se o robô tiver o objeto na mão, largar antes de recuperar
            if self._ctx.last_object_pose is not None:
                log.warning("A largar objeto antes de recuperar...")
                self._w_grasp_cmd.write(GraspCommand(
                    header=self._make_header(),
                    objeto="",
                    objeto_id="drop",
                    pose=Pose6DOF(),
                    postura=Posture.NEUTRAL,
                ))
            self._recover_until = time.time() + 3.0
            log.warning("A recuperar de erro — aguarda 3 s...")
            return

        if time.time() >= self._recover_until:
            self._recover_until = None
            self._ctx.last_object_pose = None
            self._transition(Phase.LOCATING_OBJECT, "a tentar novamente após recuperação")

    def _handle_aborted(self) -> None:

        if self._abort_until is None:
            self._abort_until = time.time() + 2.0
            log.warning("Tarefa abortada. A aguardar novo intent do operador.")
            return

        if time.time() >= self._abort_until:
            self._abort_until = None
            self._transition(Phase.WAITING_FOR_INTENT, "pronto para nova tarefa")

    # ── Retry ─────────────────────────────────────────────────────────────────

    def _handle_retry(self, failed_phase: Phase, reason: str) -> None:

        self._ctx.retry_counts[failed_phase] += 1
        attempts = self._ctx.retry_counts[failed_phase]

        if attempts <= MAX_RETRIES:
            log.warning("Fase %s falhou: '%s'. Tentativa %d/%d → RECOVERING.",
                        failed_phase.name, reason, attempts, MAX_RETRIES)
            self._transition(Phase.RECOVERING,
                             f"retry {attempts}/{MAX_RETRIES} — {reason}")
        else:
            log.error("Fase %s falhou %d vezes consecutivas. ABORTED.",
                      failed_phase.name, MAX_RETRIES)
            self._ctx.retry_counts[failed_phase] = 0
            self._transition(Phase.ABORTED,
                             f"max retries atingido — {reason}")

    # ── Utilitários ───────────────────────────────────────────────────────────

    def _read_one(self, reader) -> Optional[object]:
        samples = reader.take(1)
        return samples[0] if samples else None

    def _build_person_goal(self) -> Optional[Goal]:

        if self._ctx.known_locations:
            for loc in self._ctx.known_locations.locations:
                if loc.name == self._ctx.last_person_id:
                    return self._make_named_goal(loc.name)

        log.warning("Pessoa '%s' não encontrada nas localizações SLAM.",
                    self._ctx.last_person_id)
        return None

    def _make_named_goal(self, name: str) -> Goal:
        data = GoalData(name=name)
        return Goal(header=self._make_header(), data=data)

    def _make_header(self) -> Header:
        with self._lock:
            self._seq += 1
            seq = self._seq
        return Header(
            timestamp_ns=time.time_ns(),
            frame_id="orchestrator",
            seq=seq,
        )

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

    # ── Atualizações passivas em background ───────────────────────────────────

    def _poll_slam_locations(self) -> None:
        for sample in self._r_slam_locs.take():
            if sample is not None:
                self._ctx.known_locations = sample

    def _poll_vision_persons(self) -> None:

        for sample in self._r_vision_per.take():
            if sample is None:
                continue
            best = max(
                (det for det in sample.detections
                 if det.lip_movement_confidence >= LIP_MOVEMENT_MIN_CONF),
                key=lambda d: d.lip_movement_confidence,
                default=None,
            )
            if best:
                self._ctx.last_person_id = best.id

    # ── Loop principal ────────────────────────────────────────────────────────

    def run(self) -> None:

        log.info("=" * 60)
        log.info("  Orquestrador RT — G1 Unitree / MujoCo")
        log.info("=" * 60)

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
            log.info("Interrompido pelo utilizador.")
        finally:
            self._transition(Phase.IDLE, "shutdown")
            log.info("Orquestrador terminado.")


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    Orchestrator().run()