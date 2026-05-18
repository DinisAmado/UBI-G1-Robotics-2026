import json
import os
import queue
import sys
import threading
import time

import cv2
import lz4.frame
import mediapipe as mp
import numpy as np
import sounddevice as sd
import zmq

# ── CycloneDDS ────────────────────────────────────────────────────────────────
from cyclonedds.domain import DomainParticipant
from cyclonedds.pub import DataWriter, Publisher
from cyclonedds.sub import DataReader, Subscriber
from cyclonedds.topic import Topic
from insightface.app import FaceAnalysis
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from idl_ri import (
    Header,
    OrchestratorState,
    PersonDetection,
    Persons,
)
from qos_profiles import QOS_ORCHESTRATION, QOS_VISION

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
FACE_MATCH_THRESH = 1.05
DB_FILE_PATH = "assinaturas.json"
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "face_landmarker.task")
DOMAIN_ID = 0

# ──────────────────────────────────────────────
# Estado global partilhado entre threads
# ──────────────────────────────────────────────
_state = {"frame": None}
_state_lock = threading.Lock()

# Pessoa alvo recebida do orquestrador
_target_person = ""
_target_person_lock = threading.Lock()

# Sequência global para headers DDS
_seq = 0
_seq_lock = threading.Lock()


def _next_seq() -> int:
    global _seq
    with _seq_lock:
        _seq += 1
        return _seq


def _make_header(frame_id: str = "camera") -> Header:
    return Header(timestamp_ns=time.time_ns(), frame_id=frame_id, seq=_next_seq())


# ──────────────────────────────────────────────
# NMS
# ──────────────────────────────────────────────
def nms(boxes, scores, iou_thresh=0.4):
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes)
    scores = np.array(scores)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        iou = (w * h) / (areas[i] + areas[order[1:]] - w * h + 1e-6)
        order = order[np.where(iou < iou_thresh)[0] + 1]
    return keep


# ──────────────────────────────────────────────
# ZMQ Receiver — webcam do robô
# ──────────────────────────────────────────────
def _rx_g1_webcam(stop_evt: threading.Event, robot_ip: str):
    audio_q = queue.Queue(maxsize=50)

    def audio_playback_thread():
        stream = None
        try:
            while not stop_evt.is_set():
                try:
                    pcm_bytes, sr, ch = audio_q.get(timeout=0.1)
                except queue.Empty:
                    continue
                arr = (
                    np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
                    / 32767.0
                )
                arr = arr.reshape(-1, 1) if ch == 1 else arr.reshape(-1, ch)
                if stream is None or stream.samplerate != sr:
                    if stream:
                        stream.stop()
                        stream.close()
                    stream = sd.OutputStream(samplerate=sr, channels=ch)
                    stream.start()
                stream.write(arr)
        except Exception as e:
            print(f"[G1 Audio] Playback error: {e}", file=sys.stderr)
        finally:
            if stream:
                stream.stop()
                stream.close()

    threading.Thread(target=audio_playback_thread, daemon=True).start()

    def parse_timestamp(part):
        if len(part) != 8:
            return None
        return float(np.frombuffer(part, dtype=np.float64)[0])

    def parse_video_parts(parts):
        if len(parts) >= 3:
            ts = parse_timestamp(parts[1])
            if ts is not None:
                return ts, parts[2]
        if len(parts) >= 2:
            return None, parts[1]
        return None, None

    def parse_audio_parts(parts):
        if len(parts) == 4:
            ts = parse_timestamp(parts[1])
            if ts is not None:
                return ts, parts[2], parts[3]
        if len(parts) > 4:
            ts = parse_timestamp(parts[1])
            if ts is not None:
                return ts, parts[2], b"".join(parts[3:])
        if len(parts) == 3:
            return None, parts[1], parts[2]
        if len(parts) > 3:
            return None, parts[1], b"".join(parts[2:])
        if len(parts) == 2:
            return None, parts[1][:5], parts[1][5:]
        return None, None, None

    def _process_audio(parts):
        _, header, pcm_compressed = parse_audio_parts(parts)
        if header is None or pcm_compressed is None:
            return
        try:
            sr = int.from_bytes(header[:4], "little")
            ch = header[4]
            pcm = lz4.frame.decompress(pcm_compressed)
            try:
                audio_q.put_nowait((pcm, sr, ch))
            except queue.Full:
                try:
                    audio_q.get_nowait()
                    audio_q.put_nowait((pcm, sr, ch))
                except queue.Empty:
                    pass
        except Exception:
            pass

    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.connect(f"tcp://{robot_ip}:5556")
    sock.setsockopt(zmq.SUBSCRIBE, b"g1_webcam")
    sock.setsockopt(zmq.SUBSCRIBE, b"g1_audio")
    sock.setsockopt(zmq.RCVTIMEO, 1000)

    while not stop_evt.is_set():
        try:
            parts = sock.recv_multipart()
        except zmq.error.Again:
            continue

        try:
            while True:
                next_parts = sock.recv_multipart(flags=zmq.NOBLOCK)
                if next_parts[0] == b"g1_audio":
                    _process_audio(next_parts)
                parts = next_parts
        except zmq.Again:
            pass

        topic = parts[0]
        if topic == b"g1_webcam":
            _, jpg_bytes = parse_video_parts(parts)
            if jpg_bytes is not None:
                arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is not None:
                    with _state_lock:
                        _state["usb_cam"] = frame
        elif topic == b"g1_audio":
            _process_audio(parts)

    sock.close()
    ctx.term()


# ──────────────────────────────────────────────
# DDS Subscriber — recebe pessoa alvo do orquestrador
# ──────────────────────────────────────────────
def _rx_orchestrator(stop_evt: threading.Event, reader: DataReader):
    global _target_person
    while not stop_evt.is_set():
        samples = reader.take(10)
        for sample in samples:
            if sample is None:
                continue
            target = sample.current_target_person.strip()
            with _target_person_lock:
                if target != _target_person:
                    print(f"[DDS] Nova pessoa alvo: '{target}'")
                    _target_person = target
        time.sleep(0.05)


# ──────────────────────────────────────────────
# Face Database
# ──────────────────────────────────────────────
class FaceDatabase:
    def __init__(self, path):
        self.path = path
        self.db = {}
        self.counter = 0

        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
            for k, v in data.items():
                self.db[k] = {
                    "encs": [np.array(e, dtype=np.float32) for e in v["encs"]],
                    "score": v["score"],
                }

    def save(self):
        data = {
            k: {"encs": [e.tolist() for e in v["encs"]], "score": v["score"]}
            for k, v in self.db.items()
        }
        with open(self.path, "w") as f:
            json.dump(data, f)

    def register(self, emb):
        best_id, best_d = None, 999
        for k, v in self.db.items():
            for e in v["encs"]:
                d = np.linalg.norm(e - emb)
                if d < best_d:
                    best_d, best_id = d, k
        if best_id and best_d < FACE_MATCH_THRESH:
            self.db[best_id]["encs"].append(emb)
            return best_id
        self.counter += 1
        new_id = f"P{self.counter}"
        self.db[new_id] = {"encs": [emb], "score": 0}
        self.save()
        return new_id

    def add_interaction(self, pid):
        if pid in self.db:
            self.db[pid]["score"] += 1

    def get_current_master(self):
        if not self.db:
            return None
        best_pid = max(self.db, key=lambda p: self.db[p]["score"])
        return best_pid if self.db[best_pid]["score"] > 0 else None


# ──────────────────────────────────────────────
# Speaker Tracker (MAR)
# ──────────────────────────────────────────────
class SpeakerTracker:
    def __init__(self):
        self._prev = {}
        self._energy = {}
        self.ATTACK = 0.2
        self.RELEASE = 2.0
        self.GATE = 0.005
        self.THRESH = 0.002

    def push(self, pid, mar):
        if pid not in self._prev:
            self._prev[pid] = mar
            self._energy[pid] = 0
            return
        vel = abs(mar - self._prev[pid])
        vel = 0 if vel < self.GATE else vel
        e = self._energy[pid]
        alpha = self.ATTACK if vel > e else self.RELEASE
        self._energy[pid] = alpha * vel + (1 - alpha) * e
        self._prev[pid] = mar

    def is_speaking(self, pid):
        return self._energy.get(pid, 0) > self.THRESH


# ──────────────────────────────────────────────
# Main Perception System
# ──────────────────────────────────────────────
class G1Perception:
    IMG_WIDTH = 1600.0
    IMG_CENTER_X = IMG_WIDTH / 2.0

    def __init__(self, w_persons: DataWriter):
        print("[INIT] InsightFace a carregar...")
        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))

        self.db = FaceDatabase(DB_FILE_PATH)

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            num_faces=1,
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options)
        self.speaker = SpeakerTracker()
        self.prev_time = time.time()

        # Writer DDS injetado no construtor
        self.w_persons = w_persons

    def _mar_tasks(self, crop):
        if crop is None or crop.size == 0:
            return 0.0, False
        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_crop)
        result = self.face_landmarker.detect(mp_image)
        if not result.face_landmarks:
            return 0.0, False
        marks = result.face_landmarks[0]

        def dist(p1, p2):
            return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

        p13, p14 = marks[13], marks[14]
        p78, p308 = marks[78], marks[308]
        vertical = dist(p13, p14)
        horizontal = dist(p78, p308)
        mar = vertical / (horizontal + 1e-6)
        return mar, True

    def _publish_persons(self, detections: list):
        """Publica lista de PersonDetection em rt/vision/persons."""
        if not detections:
            return
        self.w_persons.write(
            Persons(
                header=_make_header(),
                detections=detections,
            )
        )
        print(f"[DDS -> rt/vision/persons] {len(detections)} pessoa(s) publicada(s)")

    def run(self):
        print("[RUN] A aguardar stream do G1...")

        while True:
            time.sleep(0.22)

            with _state_lock:
                frame = _state.get("usb_cam")

            if frame is None:
                time.sleep(0.005)
                print("Não recebe imagem")
                continue

            fps = 1.0 / (time.time() - self.prev_time + 1e-6)
            self.prev_time = time.time()
            master_id = self.db.get_current_master()

            faces = self.app.get(frame)
            boxes = [f.bbox for f in faces]
            scores = [f.det_score for f in faces]
            keep = nms(boxes, scores)
            faces = [faces[i] for i in keep]

            # Lista de deteções DDS para este frame
            dds_detections = []

            with _target_person_lock:
                target = _target_person

            for f in faces:
                x1, y1, x2, y2 = f.bbox.astype(int)

                if f.det_score < 0.60 or (y2 - y1) < 40:
                    continue

                y1 = max(0, y1)
                y2 = min(frame.shape[0], y2)
                x1 = max(0, x1)
                x2 = min(frame.shape[1], x2)
                crop = frame[y1:y2, x1:x2]
                mar, ok = self._mar_tasks(crop)
                pid = self.db.register(f.normed_embedding)

                if ok:
                    self.speaker.push(pid, mar)
                    if self.speaker.is_speaking(pid):
                        self.db.add_interaction(pid)

                speaking = self.speaker.is_speaking(pid)
                score = self.db.db.get(pid, {}).get("score", 0)
                is_master = pid == master_id and pid != "??"

                # ── Yaw normalizado [-1.0, 1.0] ───────────────────────
                face_w = x2 - x1
                cx_face = x1 + face_w // 2
                yaw_norm = round((cx_face - self.IMG_CENTER_X) / self.IMG_CENTER_X, 4)

                # ── Publicar no DDS ───────────────────────────────────
                if pid == master_id:
                    dds_detections.append(
                        PersonDetection(
                            id=pid,
                            yaw=yaw_norm,
                        )
                    )

                # ── Visualização ──────────────────────────────────────
                color = (0, 255, 0) if pid == master_id else (100, 100, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                label = "{} {} S:{}  yaw:{:.2f}".format(
                    pid, "SPEAK" if speaking else "", score, yaw_norm
                )
                cv2.putText(
                    frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                )

                # Destacar pessoa alvo
                if target and pid == target:
                    cv2.putText(
                        frame,
                        "[ALVO]",
                        (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                    )

                if is_master:
                    cv2.putText(
                        frame,
                        "ELEITO",
                        (x1, y2 + 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2,
                        cv2.LINE_AA,
                    )

                    print(f"[RL STATE] id={pid}  yaw={yaw_norm}  speaking={speaking}")

            # ── Estado geral no frame ──────────────────────────────────
            state_msg, state_color = (
                (f"ELEITO {master_id} A LIDERAR", (0, 255, 0))
                if master_id
                else ("A AVALIAR CANDIDATOS", (0, 255, 255))
            )
            cv2.putText(
                frame, state_msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, state_color, 2
            )
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 0),
                2,
            )

            # Pessoa alvo atual no canto inferior
            cv2.putText(
                frame,
                "ALVO: {}".format(target if target else "---"),
                (20, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

            # ── Publicar todas as pessoas detetadas ───────────────────
            self._publish_persons(dds_detections)

            cv2.imshow("G1 PERCEPTION", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()


# ──────────────────────────────────────────────
# DDS Debug Monitor
# ──────────────────────────────────────────────
def debug_topic(stop_evt, reader, topic_name):
    print(f"[DEBUG DDS] A escutar '{topic_name}'")

    while not stop_evt.is_set():
        try:
            samples = reader.take(10)

            for sample in samples:
                if sample is None:
                    continue

                print(f"\n[DDS RX] {topic_name}")
                print(sample)

        except Exception as e:
            print(f"[DDS DEBUG ERROR] {e}")

        time.sleep(0.1)


# ──────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────
if __name__ == "__main__":
    robot_ip = sys.argv[1] if len(sys.argv) > 1 else "192.168.123.164"

    # ── DDS setup ─────────────────────────────────────────────────────────────
    print("[DDS] A inicializar domínio {} ...".format(DOMAIN_ID))
    dp = DomainParticipant(DOMAIN_ID)
    pub = Publisher(dp)
    sub = Subscriber(dp)

    t_persons = Topic(dp, "rt/vision/persons", Persons, qos=QOS_VISION)
    t_orch = Topic(
        dp, "rt/orchestration/state", OrchestratorState, qos=QOS_ORCHESTRATION
    )

    w_persons = DataWriter(pub, t_persons)
    r_orch = DataReader(sub, t_orch)
    r_persons_debug = DataReader(sub, t_persons)

    print("[DDS] Tópicos prontos.")

    # ── Threads ───────────────────────────────────────────────────────────────
    stop_evt = threading.Event()

    """
    ===========================================================
    = DEBUG: monitorizar mensagens DDS em rt/vision/persons   =
    ===========================================================
    """
    threading.Thread(
        target=debug_topic,
        args=(stop_evt, r_persons_debug, "rt/vision/persons"),
        daemon=True,
    ).start()

    threading.Thread(
        target=_rx_g1_webcam,
        args=(stop_evt, robot_ip),
        daemon=True,
    ).start()

    threading.Thread(
        target=_rx_orchestrator,
        args=(stop_evt, r_orch),
        daemon=True,
    ).start()

    # ── Run ───────────────────────────────────────────────────────────────────
    G1Perception(w_persons=w_persons).run()
