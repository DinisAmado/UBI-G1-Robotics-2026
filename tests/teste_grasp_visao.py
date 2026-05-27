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
    QOS_VISION, QOS_GRASP,
)

from idl_ri import (
    Header, Status, Pose6DOF,
    Intent, Acao, Feedback, OrchestrationState as HmiState,
    OrchestratorState, ActiveModules, Phase, Heartbeat,
    GraspCommand, GraspStatusMsg as GraspStatus, Posture,
    Objects as VisionObjects,
)

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("orch_grasp_test")

# ─── Constantes ──────────────────────────────────────────────────────────────
DOMAIN_ID        = 0
MAX_RETRIES      = 3      
LOOP_HZ          = 20     
VISION_MIN_CONF  = 0.6    

# Timeouts apenas para as fases que vamos testar
PHASE_TIMEOUTS: dict[Phase, float] = {
    Phase.LOCATING_OBJECT: 20.0,
    Phase.GRASPING_OBJECT: 30.0,
}

# ─── Mapeamento Fase → Módulos Activos (Simplificado para Grasping) ──────────
PHASE_MODULES: dict[Phase, ActiveModules] = {
    Phase.IDLE: ActiveModules(vision_objects=False, grasping=False),
    Phase.WAITING_FOR_INTENT: ActiveModules(vision_objects=False, grasping=False),
    
    # Ativa apenas a visão para procurar o objeto
    Phase.LOCATING_OBJECT: ActiveModules(vision_objects=True, grasping=False),
    
    # Mantém a visão ativa para logs, e ativa o grasping
    Phase.GRASPING_OBJECT: ActiveModules(vision_objects=True, grasping=True),
    
    Phase.RECOVERING: ActiveModules(vision_objects=False, grasping=False),
    Phase.ABORTED: ActiveModules(vision_objects=False, grasping=False),
    
    # Ignoradas neste script de teste:
    Phase.NAVIGATING_TO_TABLE: ActiveModules(),
    Phase.NAVIGATING_TO_PERSON: ActiveModules(),
    Phase.DELIVERING: ActiveModules(),
}

# ─── Contexto interno ────────────────────────────────────────────────────────
@dataclass
class OrchestratorContext:
    current_intent:    Optional[Intent] = None
    last_object_pose:  Optional[Pose6DOF] = None
    last_object_name:  str = ""
    last_failed_phase: Optional[Phase] = None
    retry_counts:      dict = field(default_factory=lambda: {p: 0 for p in Phase})

# ─── Orquestrador Simplificado ───────────────────────────────────────────────
class OrchestratorGraspTest:
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

        # ── Topics & DataWriters/Readers ───────────────────────────────────
        t_orch_state   = Topic(self._dp, "rt/orchestration/state",     OrchestratorState, qos=QOS_ORCHESTRATION)
        t_orch_hb      = Topic(self._dp, "rt/orchestration/heartbeat", Heartbeat,         qos=QOS_HEARTBEAT)
        t_hmi_intent   = Topic(self._dp, "rt/hmi/intent",              Intent,            qos=QOS_HMI)
        t_hmi_feedback = Topic(self._dp, "rt/hmi/feedback",            Feedback,          qos=QOS_HMI)
        t_grasp_cmd    = Topic(self._dp, "rt/grasp/command",           GraspCommand,      qos=QOS_GRASP)
        t_grasp_status = Topic(self._dp, "rt/grasp/status",            GraspStatus,       qos=QOS_GRASP)
        t_vision_obj   = Topic(self._dp, "rt/vision/objects",          VisionObjects,     qos=QOS_VISION)

        self._w_orch_state   = DataWriter(pub, t_orch_state)
        self._w_orch_hb      = DataWriter(pub, t_orch_hb)
        self._w_hmi_feedback = DataWriter(pub, t_hmi_feedback)
        self._w_grasp_cmd    = DataWriter(pub, t_grasp_cmd)

        self._r_hmi_intent   = DataReader(sub, t_hmi_intent)
        self._r_grasp_status = DataReader(sub, t_grasp_status)
        self._r_vision_obj   = DataReader(sub, t_vision_obj)

        log.info("Orquestrador de TESTE DE GRASPING inicializado.")

    def _transition(self, new_phase: Phase, reason: str = "") -> None:
        with self._lock:
            old_phase              = self._phase
            self._phase            = new_phase
            self._phase_start_time = time.time()

        log.info("%-25s -> %-25s  (%s)", old_phase.name, new_phase.name, reason)

        self._w_orch_state.write(OrchestratorState(
            header=self._make_header(),
            phase=new_phase,
            active_modules=PHASE_MODULES.get(new_phase, ActiveModules()),
            current_target_object=self._ctx.last_object_name,
            reason=reason,
        ))

        self._w_hmi_feedback.write(Feedback(
            header=self._make_header(),
            status=Status.DONE if new_phase in (Phase.IDLE, Phase.ABORTED) else Status.RUNNING,
            message=reason,
            state=getattr(HmiState, new_phase.name, HmiState.IDLE),
        ))

    def _check_timeout(self) -> None:
        timeout = PHASE_TIMEOUTS.get(self._phase)
        if timeout is None: return

        elapsed = time.time() - self._phase_start_time
        if elapsed > timeout:
            log.warning("Timeout na fase %s", self._phase.name)
            self._handle_retry(self._phase, "timeout")

    def _step(self) -> None:
        self._check_timeout()

        if self._phase == Phase.IDLE:
            self._transition(Phase.WAITING_FOR_INTENT, "pronto para teste")

        elif self._phase == Phase.WAITING_FOR_INTENT:
            self._handle_waiting_for_intent()

        elif self._phase == Phase.LOCATING_OBJECT:
            self._handle_locating_object()

        elif self._phase == Phase.GRASPING_OBJECT:
            self._handle_grasp_status()

        elif self._phase == Phase.RECOVERING:
            self._handle_recovering()

        elif self._phase == Phase.ABORTED:
            self._handle_aborted()

    def _handle_waiting_for_intent(self) -> None:
        samples = self._r_hmi_intent.take(1)
        if not samples: return
        sample = samples[0]

        self._ctx.current_intent = sample
        log.info("Intent HMI recebido -> acao=%s alvo='%s'", sample.acao.name, sample.alvo)

        if sample.acao in (Acao.ENTREGAR, Acao.RECOLHER):
            self._ctx.last_object_name = sample.alvo
            self._ctx.last_object_pose = None
            
            # ATENÇÃO: Salta a navegação e vai direto para a visão
            self._transition(Phase.LOCATING_OBJECT, f"à procura de '{sample.alvo}' (Teste Grasping)")
            
        elif sample.acao == Acao.LARGA:
            self._w_grasp_cmd.write(GraspCommand(
                header=self._make_header(),
                objeto=sample.alvo,
                objeto_id="drop",
                postura=Posture.NEUTRAL,
            ))
            log.info("Comando para LARGAR objeto enviado.")

        elif sample.acao == Acao.PARAR:
            self._transition(Phase.IDLE, "paragem solicitada")

    def _handle_locating_object(self) -> None:
        samples = self._r_vision_obj.take(1)
        if not samples: return
        
        for det in samples[0].detections:
            if det.name == self._ctx.last_object_name and det.confidence >= VISION_MIN_CONF:
                self._ctx.last_object_pose = det.pose
                log.info("Objecto '%s' localizado (conf=%.2f)! A iniciar Grasping...", det.name, det.confidence)

                # Manda o braço agarrar a pose exata
                self._w_grasp_cmd.write(GraspCommand(
                    header=self._make_header(),
                    objeto=self._ctx.last_object_name,
                    objeto_id="",
                    pose=det.pose,
                    postura=Posture.EXTEND_ARM_FORWARD,
                ))
                self._transition(Phase.GRASPING_OBJECT, "a iniciar cinemática de grasp")
                return

    def _handle_grasp_status(self) -> None:
        samples = self._r_grasp_status.take(1)
        if not samples: return
        sample = samples[0]

        if sample.status == Status.DONE:
            log.info("=== SUCESSO: O objeto foi agarrado! ===")
            self._ctx.retry_counts = {p: 0 for p in Phase}
            # Fim do teste: volta ao IDLE
            self._transition(Phase.IDLE, "teste de grasping concluído com sucesso")

        elif sample.status == Status.FAILED:
            self._handle_retry(self._phase, sample.reason)

    def _handle_recovering(self) -> None:
        if self._recover_until is None:
            self._recover_until = time.time() + 3.0
            log.warning("A aguardar 3s para recuperar do erro...")
            return

        if time.time() >= self._recover_until:
            self._recover_until = None
            # Relocaliza o objeto para tentar de novo
            self._transition(Phase.LOCATING_OBJECT, "a tentar relocalizar o objeto")

    def _handle_aborted(self) -> None:
        if self._abort_until is None:
            self._abort_until = time.time() + 2.0
            log.error("=== TESTE ABORTADO ===")
            return

        if time.time() >= self._abort_until:
            self._abort_until = None
            self._transition(Phase.WAITING_FOR_INTENT, "pronto para novo teste")

    def _handle_retry(self, failed_phase: Phase, reason: str) -> None:
        self._ctx.last_failed_phase = failed_phase
        self._ctx.retry_counts[failed_phase] += 1
        attempts = self._ctx.retry_counts[failed_phase]

        if attempts <= MAX_RETRIES:
            log.warning("Falha no grasping. Tentativa %d/%d -> RECOVERING.", attempts, MAX_RETRIES)
            self._transition(Phase.RECOVERING, f"retry {attempts} - {reason}")
        else:
            self._ctx.retry_counts[failed_phase] = 0
            self._transition(Phase.ABORTED, "max retries atingido no grasping")

    def _make_header(self) -> Header:
        with self._lock:
            self._seq += 1
            return Header(timestamp_ns=time.time_ns(), frame_id="orch_test", seq=self._seq)

    def run(self) -> None:
        log.info("================================================")
        log.info("  ORQUESTRADOR G1 - MODO TESTE ISOLADO GRASPING ")
        log.info("================================================")
        self._transition(Phase.IDLE, "arranque")

        hb_count, sleep_s = 0, 1.0 / LOOP_HZ
        try:
            while True:
                self._step()
                hb_count += 1
                if hb_count >= LOOP_HZ:
                    self._w_orch_hb.write(Heartbeat(
                        header=self._make_header(), module_name="orch_test", ready=True
                    ))
                    hb_count = 0
                time.sleep(sleep_s)
        except KeyboardInterrupt:
            log.info("Teste interrompido.")
        finally:
            self._transition(Phase.IDLE, "shutdown")

if __name__ == "__main__":
    OrchestratorGraspTest().run()<import time
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
    QOS_VISION, QOS_GRASP,
)

from idl_ri import (
    Header, Status, Pose6DOF,
    Intent, Acao, Feedback, OrchestrationState as HmiState,
    OrchestratorState, ActiveModules, Phase, Heartbeat,
    GraspCommand, GraspStatusMsg as GraspStatus, Posture,
    Objects as VisionObjects,
)

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("orch_grasp_test")

# ─── Constantes ──────────────────────────────────────────────────────────────
DOMAIN_ID        = 0
MAX_RETRIES      = 3      
LOOP_HZ          = 20     
VISION_MIN_CONF  = 0.6    

# Timeouts apenas para as fases que vamos testar
PHASE_TIMEOUTS: dict[Phase, float] = {
    Phase.LOCATING_OBJECT: 20.0,
    Phase.GRASPING_OBJECT: 30.0,
}

# ─── Mapeamento Fase → Módulos Activos (Simplificado para Grasping) ──────────
PHASE_MODULES: dict[Phase, ActiveModules] = {
    Phase.IDLE: ActiveModules(vision_objects=False, grasping=False),
    Phase.WAITING_FOR_INTENT: ActiveModules(vision_objects=False, grasping=False),
    
    # Ativa apenas a visão para procurar o objeto
    Phase.LOCATING_OBJECT: ActiveModules(vision_objects=True, grasping=False),
    
    # Mantém a visão ativa para logs, e ativa o grasping
    Phase.GRASPING_OBJECT: ActiveModules(vision_objects=True, grasping=True),
    
    Phase.RECOVERING: ActiveModules(vision_objects=False, grasping=False),
    Phase.ABORTED: ActiveModules(vision_objects=False, grasping=False),
    
    # Ignoradas neste script de teste:
    Phase.NAVIGATING_TO_TABLE: ActiveModules(),
    Phase.NAVIGATING_TO_PERSON: ActiveModules(),
    Phase.DELIVERING: ActiveModules(),
}

# ─── Contexto interno ────────────────────────────────────────────────────────
@dataclass
class OrchestratorContext:
    current_intent:    Optional[Intent] = None
    last_object_pose:  Optional[Pose6DOF] = None
    last_object_name:  str = ""
    last_failed_phase: Optional[Phase] = None
    retry_counts:      dict = field(default_factory=lambda: {p: 0 for p in Phase})

# ─── Orquestrador Simplificado ───────────────────────────────────────────────
class OrchestratorGraspTest:
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

        # ── Topics & DataWriters/Readers ───────────────────────────────────
        t_orch_state   = Topic(self._dp, "rt/orchestration/state",     OrchestratorState, qos=QOS_ORCHESTRATION)
        t_orch_hb      = Topic(self._dp, "rt/orchestration/heartbeat", Heartbeat,         qos=QOS_HEARTBEAT)
        t_hmi_intent   = Topic(self._dp, "rt/hmi/intent",              Intent,            qos=QOS_HMI)
        t_hmi_feedback = Topic(self._dp, "rt/hmi/feedback",            Feedback,          qos=QOS_HMI)
        t_grasp_cmd    = Topic(self._dp, "rt/grasp/command",           GraspCommand,      qos=QOS_GRASP)
        t_grasp_status = Topic(self._dp, "rt/grasp/status",            GraspStatus,       qos=QOS_GRASP)
        t_vision_obj   = Topic(self._dp, "rt/vision/objects",          VisionObjects,     qos=QOS_VISION)

        self._w_orch_state   = DataWriter(pub, t_orch_state)
        self._w_orch_hb      = DataWriter(pub, t_orch_hb)
        self._w_hmi_feedback = DataWriter(pub, t_hmi_feedback)
        self._w_grasp_cmd    = DataWriter(pub, t_grasp_cmd)

        self._r_hmi_intent   = DataReader(sub, t_hmi_intent)
        self._r_grasp_status = DataReader(sub, t_grasp_status)
        self._r_vision_obj   = DataReader(sub, t_vision_obj)

        log.info("Orquestrador de TESTE DE GRASPING inicializado.")

    def _transition(self, new_phase: Phase, reason: str = "") -> None:
        with self._lock:
            old_phase              = self._phase
            self._phase            = new_phase
            self._phase_start_time = time.time()

        log.info("%-25s -> %-25s  (%s)", old_phase.name, new_phase.name, reason)

        self._w_orch_state.write(OrchestratorState(
            header=self._make_header(),
            phase=new_phase,
            active_modules=PHASE_MODULES.get(new_phase, ActiveModules()),
            current_target_object=self._ctx.last_object_name,
            reason=reason,
        ))

        self._w_hmi_feedback.write(Feedback(
            header=self._make_header(),
            status=Status.DONE if new_phase in (Phase.IDLE, Phase.ABORTED) else Status.RUNNING,
            message=reason,
            state=getattr(HmiState, new_phase.name, HmiState.IDLE),
        ))

    def _check_timeout(self) -> None:
        timeout = PHASE_TIMEOUTS.get(self._phase)
        if timeout is None: return

        elapsed = time.time() - self._phase_start_time
        if elapsed > timeout:
            log.warning("Timeout na fase %s", self._phase.name)
            self._handle_retry(self._phase, "timeout")

    def _step(self) -> None:
        self._check_timeout()

        if self._phase == Phase.IDLE:
            self._transition(Phase.WAITING_FOR_INTENT, "pronto para teste")

        elif self._phase == Phase.WAITING_FOR_INTENT:
            self._handle_waiting_for_intent()

        elif self._phase == Phase.LOCATING_OBJECT:
            self._handle_locating_object()

        elif self._phase == Phase.GRASPING_OBJECT:
            self._handle_grasp_status()

        elif self._phase == Phase.RECOVERING:
            self._handle_recovering()

        elif self._phase == Phase.ABORTED:
            self._handle_aborted()

    def _handle_waiting_for_intent(self) -> None:
        samples = self._r_hmi_intent.take(1)
        if not samples: return
        sample = samples[0]

        self._ctx.current_intent = sample
        log.info("Intent HMI recebido -> acao=%s alvo='%s'", sample.acao.name, sample.alvo)

        if sample.acao in (Acao.ENTREGAR, Acao.RECOLHER):
            self._ctx.last_object_name = sample.alvo
            self._ctx.last_object_pose = None
            
            # ATENÇÃO: Salta a navegação e vai direto para a visão
            self._transition(Phase.LOCATING_OBJECT, f"à procura de '{sample.alvo}' (Teste Grasping)")
            
        elif sample.acao == Acao.LARGA:
            self._w_grasp_cmd.write(GraspCommand(
                header=self._make_header(),
                objeto=sample.alvo,
                objeto_id="drop",
                postura=Posture.NEUTRAL,
            ))
            log.info("Comando para LARGAR objeto enviado.")

        elif sample.acao == Acao.PARAR:
            self._transition(Phase.IDLE, "paragem solicitada")

    def _handle_locating_object(self) -> None:
        samples = self._r_vision_obj.take(1)
        if not samples: return
        
        for det in samples[0].detections:
            if det.name == self._ctx.last_object_name and det.confidence >= VISION_MIN_CONF:
                self._ctx.last_object_pose = det.pose
                log.info("Objecto '%s' localizado (conf=%.2f)! A iniciar Grasping...", det.name, det.confidence)

                # Manda o braço agarrar a pose exata
                self._w_grasp_cmd.write(GraspCommand(
                    header=self._make_header(),
                    objeto=self._ctx.last_object_name,
                    objeto_id="",
                    pose=det.pose,
                    postura=Posture.EXTEND_ARM_FORWARD,
                ))
                self._transition(Phase.GRASPING_OBJECT, "a iniciar cinemática de grasp")
                return

    def _handle_grasp_status(self) -> None:
        samples = self._r_grasp_status.take(1)
        if not samples: return
        sample = samples[0]

        if sample.status == Status.DONE:
            log.info("=== SUCESSO: O objeto foi agarrado! ===")
            self._ctx.retry_counts = {p: 0 for p in Phase}
            # Fim do teste: volta ao IDLE
            self._transition(Phase.IDLE, "teste de grasping concluído com sucesso")

        elif sample.status == Status.FAILED:
            self._handle_retry(self._phase, sample.reason)

    def _handle_recovering(self) -> None:
        if self._recover_until is None:
            self._recover_until = time.time() + 3.0
            log.warning("A aguardar 3s para recuperar do erro...")
            return

        if time.time() >= self._recover_until:
            self._recover_until = None
            # Relocaliza o objeto para tentar de novo
            self._transition(Phase.LOCATING_OBJECT, "a tentar relocalizar o objeto")

    def _handle_aborted(self) -> None:
        if self._abort_until is None:
            self._abort_until = time.time() + 2.0
            log.error("=== TESTE ABORTADO ===")
            return

        if time.time() >= self._abort_until:
            self._abort_until = None
            self._transition(Phase.WAITING_FOR_INTENT, "pronto para novo teste")

    def _handle_retry(self, failed_phase: Phase, reason: str) -> None:
        self._ctx.last_failed_phase = failed_phase
        self._ctx.retry_counts[failed_phase] += 1
        attempts = self._ctx.retry_counts[failed_phase]

        if attempts <= MAX_RETRIES:
            log.warning("Falha no grasping. Tentativa %d/%d -> RECOVERING.", attempts, MAX_RETRIES)
            self._transition(Phase.RECOVERING, f"retry {attempts} - {reason}")
        else:
            self._ctx.retry_counts[failed_phase] = 0
            self._transition(Phase.ABORTED, "max retries atingido no grasping")

    def _make_header(self) -> Header:
        with self._lock:
            self._seq += 1
            return Header(timestamp_ns=time.time_ns(), frame_id="orch_test", seq=self._seq)

    def run(self) -> None:
        log.info("================================================")
        log.info("  ORQUESTRADOR G1 - MODO TESTE ISOLADO GRASPING ")
        log.info("================================================")
        self._transition(Phase.IDLE, "arranque")

        hb_count, sleep_s = 0, 1.0 / LOOP_HZ
        try:
            while True:
                self._step()
                hb_count += 1
                if hb_count >= LOOP_HZ:
                    self._w_orch_hb.write(Heartbeat(
                        header=self._make_header(), module_name="orch_test", ready=True
                    ))
                    hb_count = 0
                time.sleep(sleep_s)
        except KeyboardInterrupt:
            log.info("Teste interrompido.")
        finally:
            self._transition(Phase.IDLE, "shutdown")

if __name__ == "__main__":
    OrchestratorGraspTest().run()