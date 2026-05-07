import cv2
import numpy as np
import threading
import time
import sys
import argparse
from ultralytics import YOLO

# ──────────────────────────────────────────────
# Estado global
# ──────────────────────────────────────────────
_state = {"rgbd": None}
_state_lock = threading.Lock()

WINDOW_NAME = "G1 ZMQ YOLO MAIN"

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

            # flush buffer (evita lag)
            try:
                while True:
                    topic, color_bytes, depth_compressed = socket.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                pass

            # RGB
            color_array = np.frombuffer(color_bytes, dtype=np.uint8)
            rgb = cv2.imdecode(color_array, cv2.IMREAD_COLOR)

            # Depth
            depth_bytes = lz4.frame.decompress(depth_compressed)
            depth_image = np.frombuffer(depth_bytes, dtype=np.uint16).reshape((480, 640))

            depth_bgr = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET
            )

            combo = cv2.hconcat([rgb, depth_bgr])

            fps = 1.0 / (time.perf_counter() - last + 1e-6)
            last = time.perf_counter()

            cv2.putText(combo,
                        f"ZMQ RGB+Depth {fps:5.1f} FPS",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2)

            with _state_lock:
                _state["rgbd"] = combo

        socket.close()
        context.term()

    except Exception as e:
        print("[ZMQ ERROR]", e, file=sys.stderr)


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":

    cv2.destroyAllWindows()  # 🔥 garante limpeza inicial

    parser = argparse.ArgumentParser(description="G1 ZMQ + YOLO System")

    parser.add_argument("--robot-ip", default="192.168.123.164")
    parser.add_argument("--clear", type=float, default=18.0)

    args = parser.parse_args()

    print(f"[CONFIG] Robot IP: {args.robot_ip}")

    # ───────── YOLO ─────────
    model_path = "/home/nova-lincs-04/unitree_sdk2_python/RI/3/UBI-G1-Robotics-2026/src/modules/vision/robo/obj/best.pt"

    print(f"[YOLO] Loading: {model_path}")

    model = YOLO(model_path)
    print("[OK] YOLO loaded")

    # ───────── COLORS ─────────
    custom_colors = {
        'bola': (0, 255, 0),
        'pasta': (128, 0, 128),
        'cubo': (0, 0, 255)
    }

    ALPHA = 0.4

    # ───────── THREAD ─────────
    stop_event = threading.Event()

    t = threading.Thread(
        target=_rx_realsense,
        args=(stop_event, args.robot_ip),
        daemon=True
    )
    t.start()

    print("\n[INFO] SYSTEM STARTED (ZMQ + YOLO)")

    # cria janela fixa (evita duplicação)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        while True:

            with _state_lock:
                frame = _state["rgbd"]

            # ───── fallback
            if frame is None:
                debug = np.zeros((480, 640, 3), dtype=np.uint8)

                cv2.putText(debug,
                            "WAITING FOR ZMQ STREAM...",
                            (50, 220),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 255),
                            2)

                cv2.putText(debug,
                            f"Robot IP: {args.robot_ip}",
                            (50, 260),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            1)

                cv2.imshow(WINDOW_NAME, debug)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                continue

            # ───── YOLO
            overlay = frame.copy()

            results = model.predict(
                source=frame,
                conf=0.9,
                verbose=False,
                device=0
            )

            for r in results:

                if r.masks is not None:
                    for mask, box in zip(r.masks.xy, r.boxes):
                        label = model.names[int(box.cls[0])]
                        color = custom_colors.get(label, (255, 255, 255))
                        cv2.fillPoly(overlay, np.int32([mask]), color)

                frame = cv2.addWeighted(overlay, ALPHA, frame, 1 - ALPHA, 0)

                for box in r.boxes:
                    label = model.names[int(box.cls[0])]
                    conf = float(box.conf[0])
                    color = custom_colors.get(label, (255, 255, 255))

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame,
                                f"{label} {conf:.2f}",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                color,
                                2)

            cv2.imshow(WINDOW_NAME, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        stop_event.set()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        print("[FINALIZADO]")
