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
    QOS_NAV, QOS_GRASP, QOS_MOTION,
)

from idl_ri import (
    Header, Status, Image,
    Intent, Acao, Feedback, OrchestrationState as HmiState,
    OrchestratorState, ActiveModules, Phase, Heartbeat,
    NavGoal as Goal, GoalType, GoalData, NavStatusMsg as NavStatus,
    GraspCommand, GraspStatusMsg as GraspStatus,
    MotionCommand, Posture, MotionStatusMsg as MotionStatus,
    Objects as VisionObjects, Persons as VisionPersons, Locations
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

# ── Timeouts por fase ────────────────────────────────────────────
#
# Se uma fase ultrapassar este tempo (segundos) sem avançar, é tratada como falha e entra no ciclo de retry normal.
# Fases sem entrada aqui (IDLE, WAITING_FOR_INTENT, etc.) não têm timeout.

PHASE_TIMEOUTS: dict[Phase, float] = {                   
    Phase.LOCATING_OBJECT:      10.0,
    Phase.NAVIGATING_TO_TABLE:  20.0,
    Phase.GRASPING_OBJECT:      15.0,
    Phase.NAVIGATING_TO_PERSON: 20.0,
    Phase.DELIVERING:           10.0,
}

TABLE_LOCATION_NAME = "table"   # nome da localização SLAM usada para navegar até à mesa
LIP_MOVEMENT_MIN_CONF = 0.5     # confiança mínima para aceitar deteção de pessoa por lábios


# ─── Mapeamento Fase → Módulos Activos ───────────────────────────────────────
#
# Quando o orquestrador entra numa fase, publica rt/orchestration/state
# com este ActiveModules. Cada módulo subscreve esse tópico e verifica
# o seu próprio campo — se for False, o módulo ignora dados de entrada
# e não processa nada.

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
        navigation=True,  grasping=False, motion=False,
    ),
    Phase.GRASPING_OBJECT: ActiveModules(
        vision_objects=False, vision_persons=True,
        navigation=False, grasping=True,  motion=True,
    ),
    Phase.NAVIGATING_TO_PERSON: ActiveModules(
        vision_objects=False, vision_persons=True,
        navigation=True,  grasping=False, motion=False,
    ),
    Phase.DELIVERING: ActiveModules(
        vision_objects=False, vision_persons=True,
        navigation=False, grasping=False, motion=True,
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
    current_intent:   Optional[Intent]      = None

    # Último objecto detectado pela visão (imagem usada pelo grasping)
    last_object_image: Optional[Image]      = None   # vision.ObjectDetection.image
    last_object_name:  str                  = ""

    # Última pessoa detetada por movimento dos lábios
    last_person_id:   str                   = ""

    # Localizações conhecidas do SLAM (actualizadas em background)
    known_locations:  Optional[Locations]   = None

    # Contadores de retry por fase (reset ao completar tarefa com sucesso)
    retry_counts: dict = field(default_factory=lambda: {p: 0 for p in Phase})


# ─── Orquestrador ────────────────────────────────────────────────────────────

class Orchestrator:

    def __init__(self):
        self._ctx   = OrchestratorContext()
        self._phase = Phase.IDLE
        self._lock  = threading.Lock()
        self._seq   = 0

        self._phase_start_time: float       = time.time()
        self._recover_until:    Optional[float] = None   # timer não-bloqueante
        self._abort_until:      Optional[float] = None   # timer não-bloqueante

        # ── DDS setup ──────────────────────────────────────────────────────
        self._dp = DomainParticipant(DOMAIN_ID)
        pub      = Publisher(self._dp)
        sub      = Subscriber(self._dp)

        # ── Topics ─────────────────────────────────────────────────────────
        t_orch_state    = Topic(self._dp, "rt/orchestration/state",      OrchestratorState, qos=QOS_ORCHESTRATION)
        t_orch_hb       = Topic(self._dp, "rt/orchestration/heartbeat",  Heartbeat,         qos=QOS_HEARTBEAT)
        t_hmi_intent    = Topic(self._dp, "rt/hmi/intent",               Intent,            qos=QOS_HMI)
        t_hmi_feedback  = Topic(self._dp, "rt/hmi/feedback",             Feedback,          qos=QOS_HMI)
        t_nav_goal      = Topic(self._dp, "rt/nav/goal",                 Goal,              qos=QOS_NAV)
        t_nav_status    = Topic(self._dp, "rt/nav/status",               NavStatus,         qos=QOS_NAV)
        t_grasp_cmd     = Topic(self._dp, "rt/grasp/command",            GraspCommand,      qos=QOS_GRASP)
        t_grasp_status  = Topic(self._dp, "rt/grasp/status",             GraspStatus,       qos=QOS_GRASP)
        t_motion_cmd    = Topic(self._dp, "rt/motion/command",           MotionCommand,     qos=QOS_MOTION)
        t_motion_status = Topic(self._dp, "rt/motion/status",            MotionStatus,      qos=QOS_MOTION)
        t_vision_obj    = Topic(self._dp, "rt/vision/objects",           VisionObjects,     qos=QOS_VISION)
        t_vision_per    = Topic(self._dp, "rt/vision/persons",           VisionPersons,     qos=QOS_VISION)
        t_slam_locs     = Topic(self._dp, "rt/slam/locations",           Locations,         qos=QOS_SLAM_MAP)

        # ── Writers (o orquestrador publica estes tópicos) ─────────────────
        self._w_orch_state   = DataWriter(pub, t_orch_state)
        self._w_orch_hb      = DataWriter(pub, t_orch_hb)
        self._w_hmi_feedback = DataWriter(pub, t_hmi_feedback)
        self._w_nav_goal     = DataWriter(pub, t_nav_goal)
        self._w_grasp_cmd    = DataWriter(pub, t_grasp_cmd)
        self._w_motion_cmd   = DataWriter(pub, t_motion_cmd)

        # ── Readers (o orquestrador subscreve estes tópicos) ───────────────
        self._r_hmi_intent    = DataReader(sub, t_hmi_intent)
        self._r_nav_status    = DataReader(sub, t_nav_status)
        self._r_grasp_status  = DataReader(sub, t_grasp_status)
        self._r_motion_status = DataReader(sub, t_motion_status)
        self._r_vision_obj    = DataReader(sub, t_vision_obj)
        self._r_vision_per    = DataReader(sub, t_vision_per)
        self._r_slam_locs     = DataReader(sub, t_slam_locs)

        log.info("Orquestrador inicializado no domínio %d", DOMAIN_ID)

    # Máquina de estados

    def _transition(self, new_phase: Phase, reason: str = "") -> None:

        with self._lock:
            old_phase          = self._phase
            self._phase        = new_phase
            self._phase_start_time = time.time()    

        log.info("%-25s → %-25s  (%s)", old_phase.name, new_phase.name, reason)

        # Publica o novo estado
        self._w_orch_state.write(OrchestratorState(
            header=self._make_header(),
            phase=new_phase,
            active_modules=PHASE_MODULES[new_phase],
            current_target_object=self._ctx.last_object_name,
            current_target_person=self._ctx.last_person_id,
            reason=reason,
        ))

        # Mantém o HMI informado (interface do operador)
        self._publish_hmi_feedback(new_phase, reason)

    # ── Verificação de timeout ───────────────────────────────────

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

        # Verificar timeout antes de processar a fase
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
            self._handle_nav_status(
                on_done=Phase.DELIVERING,
                on_done_reason="chegou à pessoa",
            )

        elif phase == Phase.DELIVERING:
            self._handle_motion_status()

        elif phase == Phase.RECOVERING:
            self._handle_recovering()

        elif phase == Phase.ABORTED:
            self._handle_aborted()

    # Handlers por fase

    def _handle_waiting_for_intent(self) -> None:

        sample = self._read_one(self._r_hmi_intent)
        if sample is None:
            return

        self._ctx.current_intent = sample
        log.info("Intent recebido → acao=%s  alvo='%s'  comando_grasping='%s'",
                 sample.acao.name, sample.alvo, sample.comando_grasping)

        if sample.acao in (Acao.ENTREGAR, Acao.RECOLHER):
            self._ctx.last_object_name  = sample.alvo
            self._ctx.last_object_image = None   # será preenchido pela visão
            self._transition(Phase.LOCATING_OBJECT,
                             f"à procura de '{sample.alvo}'")

        elif sample.acao == Acao.LARGA:
            # Ordenar directamente ao grasping para largar o objeto
            self._w_grasp_cmd.write(GraspCommand(
                header=self._make_header(),
                objeto=sample.alvo,
                objeto_id="",
                image=Image(),
            ))
            self._transition(Phase.DELIVERING, "a largar objeto")

        elif sample.acao == Acao.SEGUIR:
            # Pessoa identificada dinamicamente pela visão (movimento dos lábios)
            self._transition(Phase.NAVIGATING_TO_PERSON, "a seguir pessoa")

        elif sample.acao == Acao.PARAR:
            self._transition(Phase.IDLE, "paragem solicitada pelo operador")

    def _handle_locating_object(self) -> None:

        sample = self._read_one(self._r_vision_obj)
        if sample is None:
            return

        for det in sample.detections:
            if det.name == self._ctx.last_object_name \
                    and det.confidence >= VISION_MIN_CONF:

                self._ctx.last_object_image = det.image
                log.info("Objecto '%s' localizado (conf=%.2f)", det.name, det.confidence)

                # Navega para a localização da mesa via SLAM (pose removida da visão)
                self._w_nav_goal.write(Goal(
                    header=self._make_header(),
                    data=GoalData(discriminator=GoalType.NAMED,
                                  name=TABLE_LOCATION_NAME),
                ))
                self._transition(Phase.NAVIGATING_TO_TABLE,
                                 f"a navegar para '{TABLE_LOCATION_NAME}'")
                return

    def _handle_nav_to_table(self) -> None:

        sample = self._read_one(self._r_nav_status)
        if sample is None:
            return

        if sample.status == Status.DONE:
            image = self._ctx.last_object_image or Image()
            self._w_grasp_cmd.write(GraspCommand(
                header=self._make_header(),
                objeto=self._ctx.last_object_name,
                objeto_id="",
                image=image,
            ))
            log.info("GraspCommand enviado para '%s'", self._ctx.last_object_name)
            self._transition(Phase.GRASPING_OBJECT, "chegou à mesa")

        elif sample.status == Status.FAILED:
            self._handle_retry(self._phase, sample.reason)

    def _handle_nav_status(self, on_done: Phase, on_done_reason: str) -> None:

        sample = self._read_one(self._r_nav_status)
        if sample is None:
            return

        if sample.status == Status.DONE:
            self._transition(on_done, on_done_reason)

        elif sample.status == Status.FAILED:
            self._handle_retry(self._phase, sample.reason)

    def _handle_grasp_status(self) -> None:

        sample = self._read_one(self._r_grasp_status)
        if sample is None:
            return

        if sample.status == Status.DONE:
            # Braço em postura de transporte
            self._w_motion_cmd.write(MotionCommand(
                header=self._make_header(),
                postura=Posture.NEUTRAL,
            ))

            # Navega até à pessoa (identificada via movimento dos lábios)
            goal = self._build_person_goal()
            if goal:
                self._w_nav_goal.write(goal)
                self._transition(Phase.NAVIGATING_TO_PERSON,
                                 f"a navegar para '{self._ctx.last_person_id}'"
                                 if self._ctx.last_person_id else "a navegar para pessoa")
            else:
                log.error("Não foi possível localizar a pessoa nas localizações SLAM.")
                self._handle_retry(Phase.NAVIGATING_TO_PERSON, "pessoa não encontrada no SLAM")

        elif sample.status == Status.FAILED:
            self._handle_retry(Phase.GRASPING_OBJECT, sample.reason)

    def _handle_motion_status(self) -> None:

        sample = self._read_one(self._r_motion_status)
        if sample is None:
            return

        if sample.status == Status.DONE:
            log.info("Entrega concluída com sucesso!")
            # Reset dos contadores de retry para a próxima tarefa
            self._ctx.retry_counts = {p: 0 for p in Phase}
            self._transition(Phase.IDLE, "tarefa concluída")

        elif sample.status == Status.FAILED:
            self._handle_retry(Phase.DELIVERING, sample.reason)

    def _handle_recovering(self) -> None:
        if self._recover_until is None:
            # Largar objeto se o robô o tiver (last_object_image é None antes do locating)
            if self._ctx.last_object_image is not None:
                log.warning("A largar objeto antes de recuperar...")
                self._w_motion_cmd.write(MotionCommand(
                    header=self._make_header(),
                    postura=Posture.NEUTRAL,
                ))
                self._w_grasp_cmd.write(GraspCommand(
                    header=self._make_header(),
                    objeto="",
                    objeto_id="drop",
                    image=Image(),
                ))
            self._recover_until = time.time() + 3.0
            log.warning("A recuperar de erro — aguarda 3 s...")
            return

        if time.time() >= self._recover_until:
            self._recover_until = None
            self._ctx.last_object_image = None
            self._transition(Phase.LOCATING_OBJECT, "a tentar novamente após recuperação")

    def _handle_aborted(self) -> None:
        
        if self._abort_until is None:
            self._abort_until = time.time() + 2.0
            log.warning("Tarefa abortada. A aguardar novo intent do operador.")
            return

        if time.time() >= self._abort_until:
            self._abort_until = None
            self._transition(Phase.WAITING_FOR_INTENT, "pronto para nova tarefa")

    # Retry

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

    # Utilitários

    def _read_one(self, reader) -> Optional[object]:

        samples = reader.take(1)
        return samples[0] if samples else None

    def _build_person_goal(self) -> Optional[Goal]:

        if self._ctx.known_locations:
            for loc in self._ctx.known_locations.locations:
                if loc.name == self._ctx.last_person_id:
                    return Goal(
                        header=self._make_header(),
                        data=GoalData(discriminator=GoalType.NAMED, name=loc.name),
                    )

        log.warning("Pessoa '%s' não encontrada nas localizações SLAM.",
                    self._ctx.last_person_id)
        return None  # aproximação final via motion/vision metrics

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

    # Atualizações passivas em background (lidas a cada iteração do loop)

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

    # Loop principal

    def run(self) -> None:

        log.info("=" * 60)
        log.info("  Orquestrador RT — G1 Unitree / MujoCo")
        log.info("=" * 60)

        self._transition(Phase.IDLE, "arranque")

        heartbeat_counter = 0
        sleep_s = 1.0 / LOOP_HZ

        try:
            while True:
                # -- Atualizações passivas ----------------------------------
                self._poll_slam_locations()
                self._poll_vision_persons()

                # -- Passo da máquina de estados ----------------------------
                self._step()

                # -- Heartbeat a 1 Hz (1 em cada LOOP_HZ iterações) ---------
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