import argparse
import logging
import math
import threading
import time
from typing import Optional

import cv2
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
# Constantes & Configuracao
# ──────────────────────────────────────────────
WINDOW_NAME = "G1 ZMQ YOLO 6-DOF POSE"

# Intrínsecos RealSense D435i @ 640x480
FX, FY = 600.0, 600.0
CX, CY = 320.0, 240.0

# Confiança mínima por classe
CLASS_CONF = {
    "pasta":        0.55,
    "bola":         0.75,
    "cubo":         0.75,
}
DEFAULT_CONF = 0.90

# Cores BGR por classe
CUSTOM_COLORS = {
    "bola":         (0,   255,   0),
    "pasta":        (128,   0, 128),
    "cubo":         (0,     0, 255),
}

ALPHA        = 0.40
DEPTH_PATCH  = 5
MASK_SAMPLES = 50
EMA_ALPHA    = 0.35

NORMAL_OFFSETS = [
    ( 15,   0), (-15,   0),
    (  0,  15), (  0, -15),
    ( 10,  10), (-10,  10),
    ( 10, -10), (-10, -10),
]

# ──────────────────────────────────────────────
# Estado global partilhado entre threads
# ──────────────────────────────────────────────
_state      = {"rgbd": None, "depth_raw": None, "rgb_raw": None}
_state_lock = threading.Lock()


# ──────────────────────────────────────────────
# Filtro EMA de pose por objecto
# ──────────────────────────────────────────────
class PoseFilter:
    """Filtro de media exponencial (EMA) para suavizar pose entre frames."""

    def __init__(self, alpha: float = EMA_ALPHA):
        self.alpha         = alpha
        self._pose         = None   # type: Optional[list]

    def update(self, new_pose: list) -> list:
        if self._pose is None:
            self._pose = new_pose[:]
        else:
            self._pose = [
                self.alpha * n + (1.0 - self.alpha) * o
                for n, o in zip(new_pose, self._pose)
            ]
        return [round(v, 4) for v in self._pose]

    def reset(self) -> None:
        self._pose = None


# Dicionario global: label -> PoseFilter
_pose_filters = {}   # type: dict


# ──────────────────────────────────────────────
# ZMQ Receiver (thread separada)
# ──────────────────────────────────────────────
def _rx_realsense(stop, robot_ip):
    # type: (threading.Event, str) -> None
    """Recebe frames RGB-D via ZMQ e actualiza o estado global."""
    try:
        import zmq
        import lz4.frame

        log.info("[ZMQ] A ligar a %s:5555 ...", robot_ip)
        ctx    = zmq.Context()
        socket = ctx.socket(zmq.SUB)
        socket.connect("tcp://{}:5555".format(robot_ip))
        socket.setsockopt(zmq.SUBSCRIBE, b"g1_vision")
        socket.setsockopt(zmq.RCVTIMEO, 1000)

        last_ts = time.perf_counter()

        while not stop.is_set():
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

            # Descodificar Depth (uint16, mm)
            depth_bytes = lz4.frame.decompress(depth_compressed)
            depth_image = np.frombuffer(depth_bytes, dtype=np.uint16).reshape((480, 640))

            # Painel visual: RGB | Depth colorido
            depth_bgr = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET,
            )
            combo   = cv2.hconcat([rgb, depth_bgr])
            fps     = 1.0 / (time.perf_counter() - last_ts + 1e-9)
            last_ts = time.perf_counter()

            cv2.putText(
                combo, "ZMQ RGB+Depth  {:.1f} FPS".format(fps),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
            )

            # Actualizacao atomica: rgb e depth do MESMO frame
            with _state_lock:
                _state["rgbd"]      = combo
                _state["depth_raw"] = depth_image
                _state["rgb_raw"]   = rgb

        socket.close()
        ctx.term()

    except Exception as exc:
        log.error("[ZMQ] Erro fatal: %s", exc, exc_info=True)


# ──────────────────────────────────────────────
# Geometria 3-D
# ──────────────────────────────────────────────
def _depth_patch(u, v, depth_raw):
    # type: (int, int, np.ndarray) -> float
    """Mediana de patch DEPTH_PATCH x DEPTH_PATCH excluindo pixels invalidos."""
    h, w = depth_raw.shape
    r    = DEPTH_PATCH // 2
    u0, u1 = max(0, u - r), min(w, u + r + 1)
    v0, v1 = max(0, v - r), min(h, v + r + 1)
    patch  = depth_raw[v0:v1, u0:u1].astype(np.float32)
    valid  = patch[patch > 0]
    return float(np.median(valid)) / 1000.0 if valid.size else 0.0


def _depth_from_mask(mask_pts, depth_raw):
    # type: (np.ndarray, np.ndarray) -> float
    """
    Profundidade robusta usando todos os pixels validos dentro da mascara.
    Retorna o percentil-25 (face mais proxima do objecto).
    Fallback: patch 5x5 no centroide.
    """
    h, w   = depth_raw.shape
    canvas = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(canvas, np.int32([mask_pts]), 255)

    valid_depth = depth_raw[canvas == 255].astype(np.float32)
    valid_depth = valid_depth[valid_depth > 0]

    if valid_depth.size >= 10:
        return float(np.percentile(valid_depth, 25)) / 1000.0

    # Fallback: centroide + patch
    M = cv2.moments(mask_pts)
    if M["m00"] == 0:
        return 0.0
    u = int(M["m10"] / M["m00"])
    v = int(M["m01"] / M["m00"])
    return _depth_patch(u, v, depth_raw)


def pixel_to_3d(u, v, z_m):
    # type: (int, int, float) -> Optional[np.ndarray]
    """Projeta pixel (u,v) com profundidade z_m para ponto 3-D (metros)."""
    if z_m <= 0.0:
        return None
    x = (u - CX) * z_m / FX
    y = (v - CY) * z_m / FY
    return np.array([x, y, z_m], dtype=np.float64)


def _sample_mask_points_3d(mask_pts, depth_raw, n=MASK_SAMPLES):
    # type: (np.ndarray, np.ndarray, int) -> list
    """
    Amostra ate n pontos 3-D uniformemente dentro da mascara.
    Usado para estimar a normal por SVD.
    """
    h, w   = depth_raw.shape
    canvas = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(canvas, np.int32([mask_pts]), 255)
    ys, xs = np.where(canvas == 255)

    if len(xs) == 0:
        return []

    idx = np.linspace(0, len(xs) - 1, min(n, len(xs)), dtype=int)
    pts = []
    for i in idx:
        ui, vi = int(xs[i]), int(ys[i])
        z = float(depth_raw[vi, ui]) / 1000.0
        if z > 0:
            x = (ui - CX) * z / FX
            y = (vi - CY) * z / FY
            pts.append(np.array([x, y, z], dtype=np.float64))
    return pts


def estimate_normal_svd(u, v, depth_raw, mask_pts=None):
    # type: (int, int, np.ndarray, Optional[np.ndarray]) -> Optional[np.ndarray]
    """
    Estima a normal a superficie com SVD.
    Prioridade: pontos da mascara completa > offsets fixos ao centroide.
    A normal aponta sempre para a camera (z < 0).
    """
    # Tentar com pontos da mascara
    if mask_pts is not None:
        pts = _sample_mask_points_3d(mask_pts, depth_raw)
        if len(pts) >= 3:
            arr      = np.stack(pts)
            centroid = arr.mean(axis=0)
            _, _, Vt = np.linalg.svd(arr - centroid)
            normal   = Vt[-1]
            if normal[2] > 0:
                normal = -normal
            norm = np.linalg.norm(normal)
            return normal / norm if norm > 1e-9 else None

    # Fallback: offsets fixos
    pts = []
    z0  = _depth_patch(u, v, depth_raw)
    p0  = pixel_to_3d(u, v, z0)
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

    arr      = np.stack(pts)
    centroid = arr.mean(axis=0)
    _, _, Vt = np.linalg.svd(arr - centroid)
    normal   = Vt[-1]
    if normal[2] > 0:
        normal = -normal
    norm = np.linalg.norm(normal)
    return normal / norm if norm > 1e-9 else None


def compute_pose_6dof(u, v, depth_raw, mask_pts):
    # type: (int, int, np.ndarray, np.ndarray) -> Optional[list]
    """
    Calcula pose 6-DOF: [X, Y, Z (m), roll, pitch, yaw (rad)]

    Referencial camera RealSense:
        X -> direita   Y -> baixo   Z -> frente
        Roll  = rotacao em torno de Z
        Pitch = rotacao em torno de X
        Yaw   = rotacao em torno de Y (orientacao 2D da mascara)
    """
    # 1. Profundidade via mascara completa
    z_m      = _depth_from_mask(mask_pts, depth_raw)
    p_center = pixel_to_3d(u, v, z_m)
    if p_center is None:
        return None

    # 2. Normal via SVD sobre pontos da mascara
    normal = estimate_normal_svd(u, v, depth_raw, mask_pts)
    if normal is None:
        return None

    # 3. Roll e Pitch a partir da normal
    pitch = math.atan2(normal[1], normal[2])
    roll  = math.atan2(-normal[0], math.sqrt(normal[1] ** 2 + normal[2] ** 2))

    # 4. Yaw a partir da orientacao 2D da mascara
    rect_2d = cv2.minAreaRect(np.int32(mask_pts))
    yaw_deg = rect_2d[2]
    # Compensar quando caixa e mais alta do que larga
    if rect_2d[1][0] < rect_2d[1][1]:
        yaw_deg += 90.0
    yaw = math.radians(yaw_deg)

    return [
        round(p_center[0], 4),  # X (m)
        round(p_center[1], 4),  # Y (m)
        round(p_center[2], 4),  # Z (m)
        round(roll,        5),  # Roll  (rad)
        round(pitch,       5),  # Pitch (rad)
        round(yaw,         5),  # Yaw   (rad)
    ]


# ──────────────────────────────────────────────
# Inferencia & Visualizacao
# ──────────────────────────────────────────────
def _run_inference(rgb_raw, depth_raw, frame_viz, model, ema_alpha=EMA_ALPHA):
    # type: (np.ndarray, np.ndarray, np.ndarray, YOLO, float) -> np.ndarray
    """
    Executa YOLO, calcula pose 6-DOF, aplica filtro EMA e anota o frame.
    Retorna frame_viz anotado.
    """
    min_conf = min(CLASS_CONF.values())
    results  = model.predict(source=rgb_raw, conf=min_conf, verbose=False, device=0)

    detected_labels = set()

    for r in results:
        # Overlay criado UMA vez por resultado (fix bug v1)
        overlay = frame_viz.copy()

        if r.masks is not None:
            for mask_pts, box in zip(r.masks.xy, r.boxes):
                label = model.names[int(box.cls[0])]
                conf  = float(box.conf[0])

                if conf < CLASS_CONF.get(label, DEFAULT_CONF):
                    continue

                detected_labels.add(label)
                color = CUSTOM_COLORS.get(label, (255, 255, 255))

                # Centroide da mascara
                M = cv2.moments(mask_pts)
                if M["m00"] == 0:
                    continue
                u = int(M["m10"] / M["m00"])
                v = int(M["m01"] / M["m00"])

                # Pose 6-DOF
                raw_pose = compute_pose_6dof(u, v, depth_raw, mask_pts)
                if raw_pose is None:
                    log.debug("[%s] Sem profundidade valida em (%d,%d)", label, u, v)
                    continue

                # Filtro EMA
                if label not in _pose_filters:
                    _pose_filters[label] = PoseFilter(alpha=ema_alpha)
                pose = _pose_filters[label].update(raw_pose)

                log.info(
                    "[%-12s] conf=%.2f  X=%+.3fm  Y=%+.3fm  Z=%.3fm  "
                    "roll=%+.4frad  pitch=%+.4frad  yaw=%+.4frad",
                    label, conf,
                    pose[0], pose[1], pose[2],
                    pose[3], pose[4], pose[5],
                )

                # Mascara colorida
                cv2.fillPoly(overlay, np.int32([mask_pts]), color)

                # Anotacoes visuais
                cv2.circle(frame_viz, (u, v), 5, (255, 255, 255), -1)
                yaw_len = 30
                ux = int(u + yaw_len * math.cos(pose[5]))
                uy = int(v + yaw_len * math.sin(pose[5]))
                cv2.arrowedLine(frame_viz, (u, v), (ux, uy), color, 2, tipLength=0.3)
                cv2.putText(
                    frame_viz,
                    "Z={:.2f}m R={:.1f}deg".format(pose[2], math.degrees(pose[3])),
                    (u + 8, v - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1,
                )

        # Fundir overlay UMA vez por resultado (fix bug v1)
        frame_viz = cv2.addWeighted(overlay, ALPHA, frame_viz, 1 - ALPHA, 0)

        # Bounding boxes
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
                "{} {:.2f}".format(label, conf),
                (x1, max(y1 - 5, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
            )

    # Resetar filtros de objectos que sairam de cena
    for lbl in list(_pose_filters):
        if lbl not in detected_labels:
            _pose_filters[lbl].reset()

    return frame_viz


# ──────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────
def main():
    cv2.destroyAllWindows()

    parser = argparse.ArgumentParser(description="G1 ZMQ + YOLO 6-DOF Pose (v3)")
    parser.add_argument("--robot-ip",   default="192.168.123.164")
    parser.add_argument("--model-path", default=(
        "/home/nova-lincs-04/unitree_sdk2_python/RI/3/"
        "UBI-G1-Robotics-2026/src/modules/vision/robo/obj/best.pt"
    ))
    parser.add_argument(
        "--ema-alpha", type=float, default=EMA_ALPHA,
        help="Fator EMA [0=max. suave  1=sem filtro]  (default: 0.35)",
    )
    args = parser.parse_args()

    log.info("A carregar modelo: %s", args.model_path)
    model     = YOLO(args.model_path)
    ema_alpha = args.ema_alpha

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
            # Leitura atomica: rgb e depth do MESMO frame
            with _state_lock:
                frame_viz = _state["rgbd"]
                depth_raw = _state["depth_raw"]
                rgb_raw   = _state["rgb_raw"]

            if frame_viz is None:
                time.sleep(0.01)
                continue

            # Copiar fora do lock para nao bloquear a thread ZMQ
            frame_viz_copy = frame_viz.copy()
            depth_raw_copy = depth_raw.copy()
            rgb_raw_copy   = rgb_raw.copy()

            frame_out = _run_inference(
                rgb_raw   = rgb_raw_copy,
                depth_raw = depth_raw_copy,
                frame_viz = frame_viz_copy,
                model     = model,
                ema_alpha = ema_alpha,
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