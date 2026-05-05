from enum import auto
from dataclasses import dataclass, field
from cyclonedds.idl import IdlStruct, IdlEnum, IdlUnion
from cyclonedds.idl.types import sequence, uint8, case
from cyclonedds.idl.annotations import key


# ─── rt.common ───────────────────────────────────────────────────────────────

class Status(IdlEnum):
    RUNNING = auto()
    DONE    = auto()
    FAILED  = auto()


@dataclass
class Vector3(IdlStruct, typename="rt.common.Vector3"):
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class Quaternion(IdlStruct, typename="rt.common.Quaternion"):
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0


@dataclass
class Pose(IdlStruct, typename="rt.common.Pose"):
    position:    Vector3    = field(default_factory=Vector3)
    orientation: Quaternion = field(default_factory=Quaternion)


@dataclass
class Header(IdlStruct, typename="rt.common.Header"):
    timestamp_ns: int = 0
    frame_id:     str = ""
    seq:          int = 0


@dataclass
class Image(IdlStruct, typename="rt.common.Image"):
    width:    int             = 0
    height:   int             = 0
    encoding: str             = ""
    data:     sequence[uint8] = field(default_factory=list)


# ─── rt.hmi ──────────────────────────────────────────────────────────────────

class Acao(IdlEnum):
    ENTREGAR = auto()
    RECOLHER = auto()
    SEGUIR   = auto()
    PARAR    = auto()
    LARGA    = auto()


@dataclass
class Intent(IdlStruct, typename="rt.hmi.Intent"):
    header:           Header = field(default_factory=Header)
    acao:             Acao   = Acao.ENTREGAR
    alvo:             str    = ""
    comando_grasping: str    = ""


class OrchestrationState(IdlEnum):
    IDLE                 = auto()
    WAITING_FOR_INTENT   = auto()
    LOCATING_OBJECT      = auto()
    NAVIGATING_TO_TABLE  = auto()
    GRASPING_OBJECT      = auto()
    NAVIGATING_TO_PERSON = auto()
    DELIVERING           = auto()
    RECOVERING           = auto()
    ABORTED              = auto()


@dataclass
class Feedback(IdlStruct, typename="rt.hmi.Feedback"):
    header:  Header             = field(default_factory=Header)
    status:  Status             = Status.RUNNING
    message: str                = ""
    state:   OrchestrationState = OrchestrationState.IDLE


# ─── rt.vision ───────────────────────────────────────────────────────────────

@dataclass
class ObjectDetection(IdlStruct, typename="rt.vision.ObjectDetection"):
    name:       str   = ""
    confidence: float = 0.0
    image:      Image = field(default_factory=Image)


@dataclass
class Objects(IdlStruct, typename="rt.vision.Objects"):
    header:     Header                    = field(default_factory=Header)
    detections: sequence[ObjectDetection] = field(default_factory=list)


@dataclass
class PersonDetection(IdlStruct, typename="rt.vision.PersonDetection"):
    id:                      str   = ""
    lip_movement_confidence: float = 0.0


@dataclass
class Persons(IdlStruct, typename="rt.vision.Persons"):
    header:     Header                    = field(default_factory=Header)
    detections: sequence[PersonDetection] = field(default_factory=list)


# ─── rt.grasp ────────────────────────────────────────────────────────────────

class Posture(IdlEnum):
    """Postura do braço comandada pelo orquestrador via GraspCommand.
    O módulo de grasping é o único responsável pelo controlo do braço.
    """
    EXTEND_ARM_FORWARD = auto()   # estender braço para agarrar ou entregar
    NEUTRAL            = auto()   # braço recolhido para transporte
    READY_TO_RECEIVE   = auto()   # braço preparado para receber objeto


@dataclass
class GraspCommand(IdlStruct, typename="rt.grasp.Command"):
    """Publicado pelo orquestrador para o módulo de grasping.

    Fluxo de uso:
      1. Chegada à mesa   → postura=EXTEND_ARM_FORWARD, objeto+image preenchidos, objeto_id=""
      2. Objeto agarrado  → postura=NEUTRAL, objeto_id="carry"  (braço em transporte)
      3. Chegada à pessoa → postura=EXTEND_ARM_FORWARD, objeto_id="deliver"
      4. Recovery c/ obj  → postura=NEUTRAL, objeto_id="drop"
    """
    header:    Header  = field(default_factory=Header)
    objeto:    str     = ""
    objeto_id: str     = ""    # "carry" | "deliver" | "drop" | "" (agarrar)
    image:     Image   = field(default_factory=Image)
    postura:   Posture = Posture.NEUTRAL


@dataclass
class GraspStatusMsg(IdlStruct, typename="rt.grasp.StatusMsg"):
    header:   Header = field(default_factory=Header)
    status:   Status = Status.RUNNING
    reason:   str    = ""
    progress: float  = 0.0


# ─── rt.slam ─────────────────────────────────────────────────────────────────

@dataclass
class Location(IdlStruct, typename="rt.slam.Location"):
    name: str  = ""
    pose: Pose = field(default_factory=Pose)


@dataclass
class Locations(IdlStruct, typename="rt.slam.Locations"):
    header:    Header             = field(default_factory=Header)
    locations: sequence[Location] = field(default_factory=list)


@dataclass
class SlamPoseMsg(IdlStruct, typename="rt.slam.PoseMsg"):
    header: Header = field(default_factory=Header)
    pose:   Pose   = field(default_factory=Pose)


# ─── rt.nav ──────────────────────────────────────────────────────────────────

class GoalType(IdlEnum):
    NAMED = auto()
    POSE  = auto()


@dataclass
class GoalData(IdlUnion, typename="rt.nav.GoalData", discriminator=GoalType):
    name: case[GoalType.NAMED, str]  = ""
    pose: case[GoalType.POSE,  Pose] = field(default_factory=Pose)


@dataclass
class NavGoal(IdlStruct, typename="rt.nav.Goal"):
    header: Header   = field(default_factory=Header)
    data:   GoalData = field(default_factory=GoalData)


@dataclass
class NavStatusMsg(IdlStruct, typename="rt.nav.StatusMsg"):
    header:   Header = field(default_factory=Header)
    status:   Status = Status.RUNNING
    reason:   str    = ""
    progress: float  = 0.0


@dataclass
class NavPath(IdlStruct, typename="rt.nav.Path"):
    header:    Header         = field(default_factory=Header)
    waypoints: sequence[Pose] = field(default_factory=list)


# ─── rt.motion ───────────────────────────────────────────────────────────────
#
# O módulo de motion é um seguidor de velocidades puro.
# Recebe CmdVel da navegação e publica OdometryMsg para o SLAM/navegação.
# O controlo do braço é responsabilidade exclusiva do módulo de grasping.

@dataclass
class CmdVel(IdlStruct, typename="rt.motion.CmdVel"):
    """Publicado pela navegação → consumido pelo motion a 50 Hz."""
    header: Header = field(default_factory=Header)
    vx:     float  = 0.0   # velocidade linear frente/trás  (m/s)
    vy:     float  = 0.0   # velocidade lateral esq/dir     (m/s)
    wz:     float  = 0.0   # velocidade angular yaw          (rad/s)


@dataclass
class OdometryMsg(IdlStruct, typename="rt.motion.OdometryMsg"):
    """Publicado pelo motion → consumido pelo SLAM e navegação."""
    header: Header = field(default_factory=Header)
    pose:   Pose   = field(default_factory=Pose)
    vx:     float  = 0.0
    vy:     float  = 0.0
    wz:     float  = 0.0


# ─── rt.orchestration ────────────────────────────────────────────────────────

@dataclass
class ActiveModules(IdlStruct, typename="rt.orchestration.ActiveModules"):
    vision_objects:  bool = False
    vision_persons:  bool = False
    navigation:      bool = False
    grasping:        bool = False
    motion:          bool = False


class Phase(IdlEnum):
    IDLE                 = auto()
    WAITING_FOR_INTENT   = auto()
    LOCATING_OBJECT      = auto()
    NAVIGATING_TO_TABLE  = auto()
    GRASPING_OBJECT      = auto()
    NAVIGATING_TO_PERSON = auto()
    DELIVERING           = auto()
    RECOVERING           = auto()
    ABORTED              = auto()


@dataclass
class OrchestratorState(IdlStruct, typename="rt.orchestration.State"):
    header:                Header        = field(default_factory=Header)
    phase:                 Phase         = Phase.IDLE
    active_modules:        ActiveModules = field(default_factory=ActiveModules)
    current_target_object: str           = ""
    current_target_person: str           = ""
    reason:                str           = ""


@dataclass
class Heartbeat(IdlStruct, typename="rt.orchestration.Heartbeat"):
    header:      Header = field(default_factory=Header)
    module_name: str    = ""
    ready:       bool   = False
    error_msg:   str    = ""