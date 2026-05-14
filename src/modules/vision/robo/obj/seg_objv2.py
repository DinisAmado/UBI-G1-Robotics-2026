import cv2
import math
import logging
import sys
import argparse
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from ultralytics import YOLO

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("g1_vision")

# ──────────────────────────────────────────────
# Constantes & Configuração
# ──────────────────────────────────────────────
WINDOW_NAME = "G1 ZMQ YOLO 6-DOF POSE"

# Intrínsecos RealSense D435i @ 640×480
FX, FY = 600.0, 600.0
CX, CY = 320.0, 240.0

# Confiança mínima por classe (mais baixa para objetos pequenos/transparentes)
CLASS_CONF: dict[str, float] = {
    "pasta":        0.85,   # pasta de dentes — superfície brilhante/transparente
    "bola":         0.90,
    "cubo":         0.90,
}
DEFAULT_CONF = 0.90

# Cores BGR por classe
CUSTOM_COLORS: dict[str, tuple] = {
    "bola":         (0,   255, 0),
    "pasta":        (128, 0,   128),
    "cubo":         (0,   0,   255),
}

ALPHA        = 0.40   # transparência da máscara
DEPTH_PATCH  = 5      # patch (px) para mediana quando se usa centroide
MASK_SAMPLES = 50     # máx. de pontos amostrados da máscara para normal/depth
NORMAL_OFFSETS = [    # offsets (du,dv) para estimar normal quando máscara é pequena
    (15,  0), (-15, 0), (0, 15), (0, -15),
    (10, 10), (-10, 10), (10, -10), (-10, -10),
]

# Filtro exponencial de pose (EMA): alpha próximo de 1 → mais reativo; perto de 0 → mais suave
EMA_ALPHA = 0.35


# ──────────────────────────────────────────────
# Filtro de pose por objeto (EMA por label)
# ──────────────────────────────────────────────
@dataclass
class PoseFilter:
    """Filtro de média exponencial (EMA) para suavizar pose entre frames."""
    alpha: float = EMA_ALPHA
    _pose: Optional[list[float]] = field(default=None, repr=False)

    def update(self, new_pose: list[float]) -> list[float]:
        if self._pose is None:
            self._pose = new_pose[:]
        else:
            self._pose = [
                self.alpha * n + (1 - self.alpha) * o
                for n, o in zip(new_pose, self._pose)
            ]
        return [round(v, 4) for v in self._pose]

    def reset(self) -> None:
        self._pose = None


# Dicionário global: label → PoseFilter
_pose_filters: dict[str, PoseFilter] = {}

# ──────────────────────────────────────────────
# Estado global partilhado entre threads
# ──────────────────────────────────────────────
_state: dict = {"rgbd": None, "depth_raw": None, "rgb_raw": None}
_state_lock = threading.Lock()


# ──────────────────────────────────────────────
# ZMQ Receiver (thread separada)
# ──────────────────────────────────────────────
def _rx_realsense(stop: threading.Event, robot_ip: str) -> None:
    """Recebe frames RGB-D via ZMQ e atualiza o estado global."""
    try:
        import zmq
        import lz4.frame

        log.info(f"[ZMQ] A ligar a {robot_ip}:5555 …")
        context = zmq.Context()
        socket  = context.socket(zmq.SUB)
        socket.connect(f"tcp://{robot_ip}:5555")
        socket.setsockopt(zmq.SUBSCRIBE, b"g1_vision")
        socket.setsockopt(zmq.RCVTIMEO, 1000)

        last_ts = time.perf_counter()

        while not stop.is_set():
            # Receber frame
            try:
                frames = socket.recv_multipart()
            except zmq.error.Again:
                continue

            if len(frames) < 3:
                continue

            _topic, color_bytes, depth_compressed = frames

            # Esvaziar buffer para eliminar lag
            try:
                while True:
                    socket.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                pass

            # Descodificar RGB
            color_arr = np.frombuffer(color_bytes, dtype=np.uint8)
            rgb = cv2.imdecode(color_arr, cv2.IMREAD_COLOR)
            if rgb is None:
                continue

            # Descodificar Depth (uint16 → mm)
            depth_bytes = lz4.frame.decompress(depth_compressed)
            depth_image = np.frombuffer(depth_bytes, dtype=np.uint16).reshape((480, 640))

            # Visualização colorida para o painel direito
            depth_bgr = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET,
            )

            combo = cv2.hconcat([rgb, depth_bgr])
            fps   = 1.0 / (time.perf_counter() - last_ts + 1e-9)
            last_ts = time.perf_counter()

            cv2.putText(
                combo, f"ZMQ RGB+Depth  {fps:5.1f} FPS",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
            )

            with _state_lock:
                _state["rgbd"]      = combo
                _state["depth_raw"] = depth_image
                _state["rgb_raw"]   = rgb

        socket.close()
        context.term()

    except Exception as exc:
        log.error(f"[ZMQ] Erro fatal: {exc}", exc_info=True)


# ──────────────────────────────────────────────
# Geometria 3-D
# ──────────────────────────────────────────────
def _depth_patch(u: int, v: int, depth_raw: np.ndarray) -> float:
    """
    Profundidade robusta via mediana de patch DEPTH_PATCH×DEPTH_PATCH.
    Exclui pixels inválidos (z == 0).  Retorna 0.0 se não houver válidos.
    """
    h, w = depth_raw.shape
    r  = DEPTH_PATCH // 2
    u0, u1 = max(0, u - r), min(w, u + r + 1)
    v0, v1 = max(0, v - r), min(h, v + r + 1)
    patch = depth_raw[v0:v1, u0:u1].astype(np.float32)
    valid = patch[patch > 0]
    return float(np.median(valid)) / 1000.0 if valid.size else 0.0


def _depth_from_mask(
    mask_pts: np.ndarray, depth_raw: np.ndarray
) -> float:
    """
    Profundidade robusta usando TODOS os pixels dentro da máscara.

    Estratégia (em cascata):
      1. Cria máscara binária da segmentação.
      2. Lê todos os valores de profundidade válidos (> 0) dentro dela.
      3. Retorna o percentil-25 (região mais próxima = face visível do objeto).
         Usar a mediana num objeto cilíndrico (pasta) incluiria a parte de trás
         estimada pelo sensor, enviesando para mais longe.
      4. Fallback: mediana de patch 5×5 no centroide se a máscara falhar.
    """
    h, w = depth_raw.shape
    canvas = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(canvas, np.int32([mask_pts]), 255)

    valid_depth = depth_raw[canvas == 255].astype(np.float32)
    valid_depth = valid_depth[valid_depth > 0]

    if valid_depth.size >= 10:
        # percentil-25: face mais próxima — melhor para objetos finos/cilíndricos
        return float(np.percentile(valid_depth, 25)) / 1000.0

    # Fallback: centroide + patch
    M = cv2.moments(mask_pts)
    if M["m00"] == 0:
        return 0.0
    u = int(M["m10"] / M["m00"])
    v = int(M["m01"] / M["m00"])
    return _depth_patch(u, v, depth_raw)


def pixel_to_3d(
    u: int, v: int, z_m: float
) -> Optional[np.ndarray]:
    """
    Projeta pixel (u, v) com profundidade z_m (metros) para ponto 3-D.
    Aceita z_m pré-calculado (evita releituras do depth_raw).
    """
    if z_m <= 0.0:
        return None
    x = (u - CX) * z_m / FX
    y = (v - CY) * z_m / FY
    return np.array([x, y, z_m], dtype=np.float64)


def _sample_mask_points_3d(
    mask_pts: np.ndarray, depth_raw: np.ndarray, n: int = MASK_SAMPLES
) -> list[np.ndarray]:
    """
    Amostra até n pontos 3-D uniformemente dentro da máscara.
    Usado para estimar a normal por SVD com muito mais pontos do que
    os 5 offsets fixos da v2 — especialmente útil para máscaras pequenas.
    """
    h, w = depth_raw.shape
    canvas = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(canvas, np.int32([mask_pts]), 255)
    ys, xs = np.where(canvas == 255)

    if len(xs) == 0:
        return []

    # Sub-amostragem uniforme
    idx = np.linspace(0, len(xs) - 1, min(n, len(xs)), dtype=int)
    pts = []
    for i in idx:
        u, v = int(xs[i]), int(ys[i])
        z = float(depth_raw[v, u]) / 1000.0
        if z > 0:
            x = (u - CX) * z / FX
            y = (v - CY) * z / FY
            pts.append(np.array([x, y, z], dtype=np.float64))
    return pts


def estimate_normal_svd(
    u: int, v: int, depth_raw: np.ndarray,
    mask_pts: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """
    Estima a normal à superfície com SVD.

    Prioridade de pontos:
      1. Pontos amostrados da máscara completa (se disponível e ≥ 3 pts)
      2. Offsets fixos à volta do centroide (fallback)

    A normal aponta sempre para a câmara (z < 0).
    """
    # Tentar com a máscara
    if mask_pts is not None:
        pts = _sample_mask_points_3d(mask_pts, depth_raw)
        if len(pts) >= 3:
            pts_arr  = np.stack(pts)
            centroid = pts_arr.mean(axis=0)
            _, _, Vt = np.linalg.svd(pts_arr - centroid)
            normal   = Vt[-1]
            if normal[2] > 0:
                normal = -normal
            norm = np.linalg.norm(normal)
            return normal / norm if norm > 1e-9 else None

    # Fallback: offsets fixos ao redor do centroide
    pts = []
    z0 = _depth_patch(u, v, depth_raw)
    p0 = pixel_to_3d(u, v, z0)
    if p0 is None:
        return None
    pts.append(p0)

    for du, dv in NORMAL_OFFSETS:
        z = _depth_patch(u + du, v + dv, depth_raw)
        p = pixel_to_3d(u + du, v + dv, z)
        if p is not None:
            pts.append(p)

    if len(pts) < 3:
        return None

    pts_arr  = np.stack(pts)
    centroid = pts_arr.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts_arr - centroid)
    normal   = Vt[-1]
    if normal[2] > 0:
        normal = -normal
    norm = np.linalg.norm(normal)
    return normal / norm if norm > 1e-9 else None


def compute_pose_6dof(
    u: int, v: int,
    depth_raw: np.ndarray,
    mask_pts: np.ndarray,
) -> Optional[list[float]]:
    """
    Calcula a pose 6-DOF completa em radianos:
        [X, Y, Z  (metros),  roll, pitch, yaw  (radianos)]

    Referencial câmara RealSense:
        X → direita,  Y → baixo,  Z → frente (para o robô)
        Roll  = rotação em torno de Z
        Pitch = rotação em torno de X
        Yaw   = rotação em torno de Y  (orientação 2D da máscara)

    Retorna None se não for possível calcular.
    """
    # 1. Profundidade robusta via máscara completa
    z_m = _depth_from_mask(mask_pts, depth_raw)
    p_center = pixel_to_3d(u, v, z_m)
    if p_center is None:
        return None

    # 2. Normal via SVD sobre pontos amostrados da máscara
    normal = estimate_normal_svd(u, v, depth_raw, mask_pts)
    if normal is None:
        return None

    # 3. Roll e Pitch a partir da normal à superfície
    pitch = math.atan2(normal[1], normal[2])
    roll  = math.atan2(-normal[0], math.sqrt(normal[1]**2 + normal[2]**2))

    # 4. Yaw a partir da orientação 2D da máscara
    #    minAreaRect devolve ângulo em [-90, 0); converter e normalizar
    rect_2d  = cv2.minAreaRect(np.int32(mask_pts))
    yaw_deg  = rect_2d[2]
    # Normalizar: se a caixa é mais alta do que larga, compensar 90°
    if rect_2d[1][0] < rect_2d[1][1]:
        yaw_deg += 90.0
    yaw = math.radians(yaw_deg)   # radianos, intervalo aprox. [-π/2, π/2]

    return [
        round(p_center[0], 4),   # X (m)
        round(p_center[1], 4),   # Y (m)
        round(p_center[2], 4),   # Z (m)
        round(roll,        5),   # Roll  (rad)
        round(pitch,       5),   # Pitch (rad)
        round(yaw,         5),   # Yaw   (rad)
    ]


# ──────────────────────────────────────────────
# Inferência & Visualização
# ──────────────────────────────────────────────
def _run_inference(
    rgb_raw:   np.ndarray,
    depth_raw: np.ndarray,
    frame_viz: np.ndarray,
    model:     YOLO,
) -> np.ndarray:
    """
    Executa YOLO, calcula pose 6-DOF, aplica filtro EMA e anota o frame.
    Retorna frame_viz anotado (não modifica os arrays de entrada).
    """
    # Usar a menor confiança definida para não perder deteções fracas no modelo
    min_conf = min(CLASS_CONF.values())
    results  = model.predict(source=rgb_raw, conf=min_conf, verbose=False, device=0)

    # Labels detetados neste frame — para reset do filtro de objetos ausentes
    detected_labels: set[str] = set()

    for r in results:
        # Overlay criado UMA vez por resultado (correção do bug v1)
        overlay = frame_viz.copy()

        if r.masks is not None:
            for mask_pts, box in zip(r.masks.xy, r.boxes):
                label = model.names[int(box.cls[0])]
                conf  = float(box.conf[0])

                # Filtrar pela confiança específica da classe
                if conf < CLASS_CONF.get(label, DEFAULT_CONF):
                    continue

                detected_labels.add(label)
                color = CUSTOM_COLORS.get(label, (255, 255, 255))

                # Centroide da máscara
                M = cv2.moments(mask_pts)
                if M["m00"] == 0:
                    continue
                u = int(M["m10"] / M["m00"])
                v = int(M["m01"] / M["m00"])

                # Pose 6-DOF (usa máscara completa para profundidade e normal)
                raw_pose = compute_pose_6dof(u, v, depth_raw, mask_pts)
                if raw_pose is None:
                    log.debug(f"[{label}] Sem profundidade válida em ({u},{v})")
                    continue

                # Filtro EMA por label (suaviza jitter entre frames)
                if label not in _pose_filters:
                    _pose_filters[label] = PoseFilter()
                pose = _pose_filters[label].update(raw_pose)

                log.info(
                    f"[{label:12s}] conf={conf:.2f}  "
                    f"X={pose[0]:+.3f}m  Y={pose[1]:+.3f}m  Z={pose[2]:.3f}m  "
                    f"roll={pose[3]:+.4f}rad  pitch={pose[4]:+.4f}rad  yaw={pose[5]:+.4f}rad"
                )

                # Máscara colorida no overlay
                cv2.fillPoly(overlay, np.int32([mask_pts]), color)

                # Anotações no frame principal
                cv2.circle(frame_viz, (u, v), 5, (255, 255, 255), -1)
                # Linha indicando o eixo Yaw estimado a partir do centroide
                yaw_len = 30
                ux = int(u + yaw_len * math.cos(pose[5]))
                uy = int(v + yaw_len * math.sin(pose[5]))
                cv2.arrowedLine(frame_viz, (u, v), (ux, uy), color, 2, tipLength=0.3)
                # Texto: distância e roll em graus (legível no vídeo)
                cv2.putText(
                    frame_viz,
                    f"Z={pose[2]:.2f}m R={math.degrees(pose[3]):.1f}\u00b0",
                    (u + 8, v - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1,
                )

        # Fundir overlay UMA vez (correção do bug v1)
        frame_viz = cv2.addWeighted(overlay, ALPHA, frame_viz, 1 - ALPHA, 0)

        # Bounding boxes com label e confiança
        for box in r.boxes:
            label = model.names[int(box.cls[0])]
            conf  = float(box.conf[0])
            if conf < CLASS_CONF.get(label, DEFAULT_CONF):
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = CUSTOM_COLORS.get(label, (255, 255, 255))
            cv2.rectangle(frame_viz, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame_viz,
                f"{label} {conf:.2f}",
                (x1, max(y1 - 5, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
            )

    # Resetar filtros de objetos que saíram de cena (evita valores obsoletos)
    for lbl in list(_pose_filters):
        if lbl not in detected_labels:
            _pose_filters[lbl].reset()

    return frame_viz


# ──────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────
def main() -> None:
    cv2.destroyAllWindows()

    parser = argparse.ArgumentParser(description="G1 ZMQ + YOLO 6-DOF Pose (v3)")
    parser.add_argument("--robot-ip",   default="192.168.123.164")
    parser.add_argument("--model-path", default=(
        "/home/nova-lincs-04/unitree_sdk2_python/RI/3/"
        "UBI-G1-Robotics-2026/src/modules/vision/robo/obj/best.pt"
    ))
    parser.add_argument(
        "--ema-alpha", type=float, default=EMA_ALPHA,
        help="Fator EMA para suavização de pose [0=máx. suave, 1=sem filtro]",
    )
    args = parser.parse_args()

    log.info(f"A carregar modelo: {args.model_path}")
    model = YOLO(args.model_path)

    # Atualizar alpha do filtro EMA se passado via CLI
    global EMA_ALPHA
    EMA_ALPHA = args.ema_alpha

    stop_event = threading.Event()
    rx_thread  = threading.Thread(
        target=_rx_realsense,
        args=(stop_event, args.robot_ip),
        daemon=True,
    )
    rx_thread.start()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    log.info("Pipeline iniciado. Prima 'q' para sair.")

    try:
        while True:
            # Cópia atómica: rgb e depth devem ser do MESMO frame
            with _state_lock:
                frame_viz  = _state["rgbd"]
                depth_raw  = _state["depth_raw"]
                rgb_raw    = _state["rgb_raw"]

            if frame_viz is None:
                time.sleep(0.01)
                continue

            # Copiar fora do lock (não bloqueia a thread ZMQ durante a cópia)
            frame_viz_copy = frame_viz.copy()
            depth_raw_copy = depth_raw.copy()
            rgb_raw_copy   = rgb_raw.copy()

            # Inferência sobre cópias independentes (sem race condition)
            frame_out = _run_inference(
                rgb_raw   = rgb_raw_copy,
                depth_raw = depth_raw_copy,
                frame_viz = frame_viz_copy,
                model     = model,
            )

            cv2.imshow(WINDOW_NAME, frame_out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        log.info("Interrompido pelo utilizador.")
    finally:
        stop_event.set()
        cv2.destroyAllWindows()
        log.info("Pipeline terminado.")


if __name__ == "__main__":
    main()