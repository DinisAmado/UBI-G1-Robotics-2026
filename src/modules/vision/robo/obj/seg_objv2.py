import cv2
import numpy as np
import threading
import time
import sys
import argparse
import math
from ultralytics import YOLO

# ──────────────────────────────────────────────
# Estado global
# ──────────────────────────────────────────────
_state = {"rgbd": None, "depth_raw": None, "rgb_raw": None}
_state_lock = threading.Lock()

WINDOW_NAME = "G1 ZMQ YOLO 6-DOF POSE"


# ──────────────────────────────────────────────
# ZMQ Receiver
# ──────────────────────────────────────────────
def _rx_realsense(stop: threading.Event, robot_ip: str) -> None:
    try:
        import zmq
        import lz4.frame

        print(f"[ZMQ] Connecting to {robot_ip}:5555")
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.connect(f"tcp://{robot_ip}:5555")
        socket.setsockopt(zmq.SUBSCRIBE, b"g1_vision")
        socket.setsockopt(zmq.RCVTIMEO, 1000)

        last = time.perf_counter()

        while not stop.is_set():
            try:
                frames = socket.recv_multipart()
            except zmq.error.Again:
                continue

            if len(frames) < 3:
                continue

            topic, color_bytes, depth_compressed = frames

            # Flush buffer para evitar lag (Original)
            try:
                while True:
                    socket.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                pass

            # RGB
            color_array = np.frombuffer(color_bytes, dtype=np.uint8)
            rgb = cv2.imdecode(color_array, cv2.IMREAD_COLOR)

            # Depth Raw e Visual
            depth_bytes = lz4.frame.decompress(depth_compressed)
            depth_image = np.frombuffer(depth_bytes, dtype=np.uint16).reshape((480, 640))

            depth_bgr = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET
            )

            combo = cv2.hconcat([rgb, depth_bgr])
            fps = 1.0 / (time.perf_counter() - last + 1e-6)
            last = time.perf_counter()

            cv2.putText(combo, f"ZMQ RGB+Depth {fps:5.1f} FPS", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            with _state_lock:
                _state["rgbd"] = combo
                _state["depth_raw"] = depth_image
                _state["rgb_raw"] = rgb

        socket.close()
        context.term()
    except Exception as e:
        print("[ZMQ ERROR]", e, file=sys.stderr)


# ──────────────────────────────────────────────
# Funções Matemáticas de Pose
# ──────────────────────────────────────────────
def get_3d_point(u, v, depth_raw):
    """ Projeta pixel 2D para ponto 3D em metros """
    if not (0 <= v < 480 and 0 <= u < 640): return None
    z = depth_raw[v, u] / 1000.0
    if z <= 0: return None

    # Intrínsecos RealSense D435
    fx, fy, cx, cy = 600.0, 600.0, 320.0, 240.0
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.array([x, y, z])


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    cv2.destroyAllWindows()

    parser = argparse.ArgumentParser(description="G1 ZMQ + YOLO 6-DOF")
    parser.add_argument("--robot-ip", default="192.168.123.164")
    args = parser.parse_args()

    model_path = "/home/nova-lincs-04/unitree_sdk2_python/RI/3/UBI-G1-Robotics-2026/src/modules/vision/robo/obj/best.pt"
    model = YOLO(model_path)

    custom_colors = {'bola': (0, 255, 0), 'pasta': (128, 0, 128), 'cubo': (0, 0, 255)}
    ALPHA = 0.4

    stop_event = threading.Event()
    t = threading.Thread(target=_rx_realsense, args=(stop_event, args.robot_ip), daemon=True)
    t.start()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        while True:
            with _state_lock:
                frame_viz = _state["rgbd"]
                depth_raw = _state["depth_raw"]
                rgb_raw = _state["rgb_raw"]

            if frame_viz is None:
                time.sleep(0.01)
                continue

            overlay = frame_viz.copy()
            results = model.predict(source=rgb_raw, conf=0.9, verbose=False, device=0)

            for r in results:
                if r.masks is not None:
                    for mask_pts, box in zip(r.masks.xy, r.boxes):
                        label = model.names[int(box.cls[0])]
                        color = custom_colors.get(label, (255, 255, 255))

                        # 1. Centroide
                        M = cv2.moments(mask_pts)
                        if M["m00"] == 0: continue
                        u, v = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

                        # 2. Cálculo de Pose 6-DOF (X, Y, Z, R, P, Y)
                        # Amostramos 3 pontos próximos no centro do objeto para definir o plano
                        p_center = get_3d_point(u, v, depth_raw)
                        p_x = get_3d_point(u + 10, v, depth_raw)
                        p_y = get_3d_point(u, v + 10, depth_raw)

                        if p_center is not None and p_x is not None and p_y is not None:
                            # Vetores no plano da face do objeto
                            v1 = p_x - p_center
                            v2 = p_y - p_center
                            normal = np.cross(v1, v2)
                            normal /= np.linalg.norm(normal)

                            # Pitch e Roll baseados na Normal
                            pitch_rad = math.atan2(normal[1], normal[2])
                            roll_rad = math.atan2(-normal[0], math.sqrt(normal[1] ** 2 + normal[2] ** 2))

                            # Yaw baseado na rotação da máscara 2D
                            rect = cv2.minAreaRect(np.int32([mask_pts]))
                            yaw = rect[2]

                            # Resultado final para o próximo grupo
                            full_pose = [
                                round(p_center[0], 3), round(p_center[1], 3), round(p_center[2], 3),  # X, Y, Z
                                round(math.degrees(roll_rad), 2),  # Roll
                                round(math.degrees(pitch_rad), 2),  # Pitch
                                round(yaw, 2)  # Yaw
                            ]

                            print(f"[{label:7}] Pose: {full_pose}")

                            # Desenhos Visuais
                            cv2.fillPoly(overlay, np.int32([mask_pts]), color)
                            cv2.circle(frame_viz, (u, v), 5, (255, 255, 255), -1)
                            cv2.putText(frame_viz, f"{p_center[2]:.2f}m", (u, v - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Aplicar transparência e desenhar Boxes (Original)
                frame_viz = cv2.addWeighted(overlay, ALPHA, frame_viz, 1 - ALPHA, 0)
                for box in r.boxes:
                    label = model.names[int(box.cls[0])]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame_viz, (x1, y1), (x2, y2), custom_colors.get(label, (255, 255, 255)), 2)
                    cv2.putText(frame_viz, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow(WINDOW_NAME, frame_viz)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        stop_event.set()
        cv2.destroyAllWindows()
        print("[FINALIZADO]")