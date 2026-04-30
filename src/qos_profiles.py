from cyclonedds.qos import Qos, Policy

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _ms(ms: float) -> int:
    return int(ms * 1_000_000)


# GRUPO 1 — Orquestração  (rt/orchestration/*)

#   Reliable | Transient Local | Keep Last 3 | Deadline 5 s | Liveliness 10 s
#   Resource limits: max_samples=10, max_instances=3

# Justificação: o estado da orquestração tem de ser fiável e disponível para
# módulos que entrem na rede mais tarde (Transient Local). Keep Last 3 retém
# os três estados mais recentes para diagnóstico.

QOS_ORCHESTRATION = Qos(
    Policy.Reliability.Reliable(max_blocking_time=_ms(100)),
    Policy.Durability.TransientLocal,
    Policy.History.KeepLast(depth=3),
    Policy.Deadline(deadline=_ms(5_000)),
    Policy.Liveliness.Automatic(lease_duration=_ms(10_000)),
    Policy.ResourceLimits(max_samples=10, max_instances=3,
                          max_samples_per_instance=10),
)

# Heartbeat — idêntico ao estado da orquestração
QOS_HEARTBEAT = QOS_ORCHESTRATION


# GRUPO 2 — SLAM  (rt/slam/*)

#   Pose  : Reliable | Volatile       | Keep Last 1 | Deadline 1 s  | Liveliness 10 s
#   Mapa  : Reliable | Transient Local| Keep Last 2 | Deadline 15 s | Liveliness 15 s

# Justificação: a pose do robô é publicada continuamente (não precisa de
# durabilidade), enquanto o mapa é raro e deve estar disponível para
# recém-chegados (Transient Local).

QOS_SLAM_POSE = Qos(
    Policy.Reliability.Reliable(max_blocking_time=_ms(100)),
    Policy.Durability.Volatile,
    Policy.History.KeepLast(depth=1),
    Policy.Deadline(deadline=_ms(1_000)),
    Policy.Liveliness.Automatic(lease_duration=_ms(10_000)),
    Policy.ResourceLimits(max_samples=2, max_instances=1,
                          max_samples_per_instance=2),
)

QOS_SLAM_MAP = Qos(
    Policy.Reliability.Reliable(max_blocking_time=_ms(100)),
    Policy.Durability.TransientLocal,
    Policy.History.KeepLast(depth=2),
    Policy.Deadline(deadline=_ms(15_000)),
    Policy.Liveliness.Automatic(lease_duration=_ms(15_000)),
    Policy.ResourceLimits(max_samples=2, max_instances=1,
                          max_samples_per_instance=2),
)


# GRUPO 3 — Visão  (rt/vision/*)

#   Reliable | Volatile | Keep Last 3 | Deadline 5 s | Liveliness 5 s
#   Resource limits: max_samples=3, max_instances=3

# Justificação: dados de visão são produzidos continuamente; não faz sentido
# guardar histórico para recém-chegados (Volatile). Keep Last 3 suaviza
# falhas pontuais de deteção sem acumular memória.

QOS_VISION = Qos(
    Policy.Reliability.Reliable(max_blocking_time=_ms(100)),
    Policy.Durability.Volatile,
    Policy.History.KeepLast(depth=3),
    Policy.Deadline(deadline=_ms(5_000)),
    Policy.Liveliness.Automatic(lease_duration=_ms(5_000)),
    Policy.ResourceLimits(max_samples=3, max_instances=3,
                          max_samples_per_instance=3),
)

# Métricas de visão (yaw/depth para Motion) — mesmo perfil
QOS_VISION_METRICS = QOS_VISION


# GRUPO 4 — Grasping  (rt/grasp/*)

#   Reliable | Transient Local | Keep Last 1 | Deadline 10 s | Liveliness 10 s
#   Resource limits: max_samples=1, max_instances=1

# Justificação: cada comando de grasping é único e não deve ser repetido
# (Keep Last 1). Transient Local garante que o grasping recebe o comando
# mesmo que arranque ligeiramente depois do orquestrador.

QOS_GRASP = Qos(
    Policy.Reliability.Reliable(max_blocking_time=_ms(100)),
    Policy.Durability.TransientLocal,
    Policy.History.KeepLast(depth=1),
    Policy.Deadline(deadline=_ms(10_000)),
    Policy.Liveliness.Automatic(lease_duration=_ms(10_000)),
    Policy.ResourceLimits(max_samples=1, max_instances=1,
                          max_samples_per_instance=1),
)


# GRUPO 2 (SLAM) — Navegação  (rt/nav/*)
# O grupo de Navegação partilha o Grupo 2 com o SLAM.
#
#   Reliable | Transient Local | Keep Last 1 | Deadline 2 s | Liveliness 2 s
#   Resource limits: max_samples=1, max_instances=1

QOS_NAV = Qos(
    Policy.Reliability.Reliable(max_blocking_time=_ms(100)),
    Policy.Durability.TransientLocal,
    Policy.History.KeepLast(depth=1),
    Policy.Deadline(deadline=_ms(2_000)),
    Policy.Liveliness.Automatic(lease_duration=_ms(2_000)),
    Policy.ResourceLimits(max_samples=1, max_instances=1,
                          max_samples_per_instance=1),
)


# GRUPO 6 — Movimentação  (rt/motion/*)

#   Best Effort | Volatile | Keep Last 1 | Deadline 20 ms | Liveliness 20 ms
#   Resource limits: max_samples=1, max_instances=1

# Justificação: controlo de movimento em tempo real — a latência é crítica.
# Best Effort evita retransmissões que causariam atrasos. Deadline de 20 ms
# alinha com a frequência de controlo do robô.

QOS_MOTION = Qos(
    Policy.Reliability.BestEffort,
    Policy.Durability.Volatile,
    Policy.History.KeepLast(depth=1),
    Policy.Deadline(deadline=_ms(20)),
    Policy.Liveliness.Automatic(lease_duration=_ms(20)),
    Policy.ResourceLimits(max_samples=1, max_instances=1,
                          max_samples_per_instance=1),
)


# GRUPO 5 - HMI  (rt/hmi/*)  

# Intent e Feedback do operador: Reliable + Transient Local para garantir
# que o orquestrador não perde comandos mesmo com arranque assíncrono.

QOS_HMI = Qos(
    Policy.Reliability.Reliable(max_blocking_time=_ms(100)),
    Policy.Durability.TransientLocal,
    Policy.History.KeepLast(depth=50),
    Policy.Deadline(deadline=_ms(6_000)),
    Policy.Liveliness.Automatic(lease_duration=_ms(6_000)),
    Policy.ResourceLimits(max_samples=250, max_instances=5,
                          max_samples_per_instance=50),
)


# Mapeamento tópico → perfil QoS  (referência rápida)

# Pode ser usado para iterar sobre todos os perfis em testes ou diagnóstico.

TOPIC_QOS_MAP: dict[str, Qos] = {
    # HMI
    "rt/hmi/intent":                QOS_HMI,
    "rt/hmi/feedback":              QOS_HMI,
    # Orchestration
    "rt/orchestration/state":       QOS_ORCHESTRATION,
    "rt/orchestration/heartbeat":   QOS_HEARTBEAT,
    # Vision
    "rt/vision/objects":            QOS_VISION,
    "rt/vision/persons":            QOS_VISION,
    "rt/vision/metrics":            QOS_VISION_METRICS,
    # SLAM
    "rt/slam/locations":            QOS_SLAM_MAP,
    "rt/slam/pose":                 QOS_SLAM_POSE,
    # Navigation
    "rt/nav/goal":                  QOS_NAV,
    "rt/nav/status":                QOS_NAV,
    "rt/nav/path":                  QOS_NAV,
    # Grasping
    "rt/grasp/command":             QOS_GRASP,
    "rt/grasp/status":              QOS_GRASP,
    # Motion
    "rt/motion/command":            QOS_MOTION,
    "rt/motion/status":             QOS_MOTION,
}