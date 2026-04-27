import json
import os
import threading
import time

import cv2
import mediapipe as mp
import numpy as np
from insightface.app import FaceAnalysis

# ──────────────────────────────────────────────
# Configurações Globais GPU SOTA
# ──────────────────────────────────────────────

FACE_MATCH_THRESH = 1.05
LIP_SKIP_FRAMES = 1
FACE_MODEL_PATH = "face_landmarker.task"
DB_FILE_PATH = "biometrics_db_arcface.json"


# ──────────────────────────────────────────────
# Sondagem de Hardware (Câmaras)
# ──────────────────────────────────────────────
def select_camera(max_tested=5) -> int:
    available_cameras = []
    print("[INIT] A sondar barramentos de vídeo via AVFoundation (aguarde)...")

    for i in range(max_tested):
        # SOTA Mac OS: Forçar a API AVFoundation para câmaras USB externas
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
        if cap.isOpened():
            # Tentar ler alguns frames para dar tempo ao hardware USB de estabilizar
            ret = False
            for _ in range(3):
                ret, _ = cap.read()
                if ret:
                    break
                time.sleep(0.1)

            if ret:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                available_cameras.append((i, f"Câmara {i} (Resolução Base: {w}x{h})"))
            cap.release()

    if not available_cameras:
        print("[ERRO] Nenhuma câmara detetada. A forçar o índice 0 (Defeito).")
        return 0

    print("\n" + "═" * 40)
    print(" HARDWARE DE VÍDEO DISPONÍVEL")
    print("═" * 40)
    for idx, desc in available_cameras:
        print(f" [{idx}] {desc}")
    print("═" * 40)

    while True:
        try:
            choice = input("\nIntroduza o ID da câmara a utilizar: ")
            choice_idx = int(choice)
            if any(idx == choice_idx for idx, _ in available_cameras):
                return choice_idx
            else:
                print(" ID inválido. Selecione um número da lista.")
        except ValueError:
            print(" Erro de sintaxe. Introduza um número inteiro.")


# ──────────────────────────────────────────────
# Vector Database SOTA (JSON 512D + Mutex)
# ──────────────────────────────────────────────
# ──────────────────────────────────────────────
# Vector Database SOTA (Dinâmica Ilimitada)
# ──────────────────────────────────────────────
class FaceDatabase:
    _COLORS = [(0, 220, 255), (50, 255, 100), (255, 140, 0), (200, 50, 255)]

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.db: dict[str, dict] = {}
        self._counter = 0
        self.file_lock = threading.Lock()
        self.load_from_disk()

    def load_from_disk(self):
        if not os.path.exists(self.db_path):
            return
        try:
            with self.file_lock:
                if os.path.getsize(self.db_path) == 0:
                    return
                with open(self.db_path, "r") as f:
                    data = json.load(f)

            for pid, entry in data.items():
                self.db[pid] = {
                    "encs": [np.array(enc, dtype=np.float32) for enc in entry["encs"]],
                    "color": tuple(entry["color"]),
                    "score": entry["score"],
                }
                pid_num = int(pid.replace("P", ""))
                if pid_num > self._counter:
                    self._counter = pid_num
        except Exception as e:
            print(f"[ERRO DB] Falha ao carregar: {e}")

    def save_to_disk(self):
        export_data = {
            pid: {
                "encs": [enc.tolist() for enc in entry["encs"]],
                "color": entry["color"],
                "score": entry["score"],
            }
            for pid, entry in self.db.items()
        }
        threading.Thread(
            target=self._write_json, args=(export_data,), daemon=True
        ).start()

    def _write_json(self, data):
        with self.file_lock:
            try:
                temp_path = self.db_path + ".tmp"
                with open(temp_path, "w") as f:
                    json.dump(data, f)
                os.replace(temp_path, self.db_path)
            except Exception:
                pass

    def register(self, target_enc: np.ndarray) -> str:
        best_id, best_d = None, float("inf")
        for pid, entry in self.db.items():
            for saved_enc in entry["encs"]:
                d = float(np.linalg.norm(saved_enc - target_enc))
                if d < best_d:
                    best_d, best_id = d, pid

        if best_id and best_d < FACE_MATCH_THRESH:
            if len(self.db[best_id]["encs"]) < 5:
                self.db[best_id]["encs"].append(target_enc.copy())
            else:
                self.db[best_id]["encs"] = (
                    [self.db[best_id]["encs"][0]]
                    + self.db[best_id]["encs"][-3:]
                    + [target_enc.copy()]
                )
            return best_id

        # Registo ilimitado de novas identidades
        self._counter += 1
        new_id = f"P{self._counter}"
        self.db[new_id] = {
            "encs": [target_enc.copy()],
            "color": self._COLORS[len(self.db) % len(self._COLORS)],
            "score": 0,
        }
        self.save_to_disk()
        return new_id

    def add_interaction(self, pid: str):
        if pid in self.db and pid != "??":
            self.db[pid]["score"] += 1
            self.save_to_disk()  # Atualiza o JSON imediatamente com o novo score

    def get_score(self, pid: str) -> int:
        return self.db[pid]["score"] if pid in self.db else 0

    def get_current_master(self) -> str | None:
        if not self.db:
            return None
        # Procura a chave (pid) com o valor de score mais alto
        best_pid = max(self.db, key=lambda p: self.db[p]["score"])
        # Garante que alguém só é eleito se já tiver falado pelo menos uma vez
        if self.db[best_pid]["score"] > 0:
            return best_pid
        return None


# ──────────────────────────────────────────────
# V-SAD Visual SOTA (Cinemática Assimétrica)
# ──────────────────────────────────────────────
class SpeakerTracker:
    def __init__(self):
        self._prev_mar: dict[str, float] = {}
        self._energy: dict[str, float] = {}

        # Configurações do Filtro Assimétrico
        self.ATTACK_ALPHA = 0.20  # Suavidade ao iniciar a fala
        self.RELEASE_ALPHA = 2.00  # Resposta rápida ao parar de falar

        self.JITTER_GATE = 0.005  # Corta micro-tremores (< 0.5% da face)
        self.SPEAK_THRESHOLD = 0.002  # Limiar de estado

    def push(self, pid: str, mar: float):
        if pid not in self._prev_mar:
            self._prev_mar[pid] = mar
            self._energy[pid] = 0.0
            return

        # 1. Delta Bruto
        raw_velocity = abs(mar - self._prev_mar[pid])

        # 2. Noise Gate
        clean_velocity = 0.0 if raw_velocity < self.JITTER_GATE else raw_velocity

        # 3. Integração Assimétrica (Attack vs Release)
        current_energy = self._energy[pid]

        if clean_velocity > current_energy:
            # Fase de carga (Fala a acelerar)
            alpha = self.ATTACK_ALPHA
        else:
            # Fase de corte (Boca parada / Fala a abrandar)
            alpha = self.RELEASE_ALPHA

        self._energy[pid] = (alpha * clean_velocity) + ((1 - alpha) * current_energy)
        self._prev_mar[pid] = mar

    def get_energy(self, pid: str) -> float:
        return self._energy.get(pid, 0.0)

    def is_speaking(self, pid: str) -> bool:
        return self.get_energy(pid) > self.SPEAK_THRESHOLD

    # métodos usados : MediaPipe FaceMesh para calcular o Mouth Aspect Ratio (MAR) a partir do recorte da face,
    # e um filtro de ataque/libertação para suavizar a deteção de fala com base no MAR.
    # O método push() é chamado a cada frame para atualizar o estado de fala de cada pessoa identificada,
    #  e is_speaking() retorna se a pessoa está atualmente a falar com base na energia calculada.


# ──────────────────────────────────────────────
# Motor Principal de Percepção GPU
# ──────────────────────────────────────────────
class G1InsightFacePerception:
    def __init__(self, camera_index: int):
        self.camera_index = camera_index

        print("\n[INIT] A alocar recursos (Aceleração Neural Apple Silicon)...")
        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))

        # SOTA Fallback: Usar a API nativa FaceMesh (Mais estável, dispensa ficheiros .task)
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,  # O InsightFace já recorta 1 face de cada vez
            refine_landmarks=False,
        )

        self.face_db = FaceDatabase(DB_FILE_PATH)
        self.speaker = SpeakerTracker()
        self.prev_time = time.time()

    def _get_mar(self, crop_bgr: np.ndarray) -> tuple[float, bool]:
        if crop_bgr.size == 0:
            return 0.0, False

        small = cv2.resize(crop_bgr, (160, 160), interpolation=cv2.INTER_CUBIC)
        rgb_crop = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_crop)

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark

            # Abertura Interna (Ponto central superior e inferior)
            inner_lip_open = abs(lm[13].y - lm[14].y)
            # Altura rígida da face (Nariz ao Queixo)
            face_height = abs(lm[1].y - lm[152].y) + 1e-6

            # Rácio imune a sorrisos
            return inner_lip_open / face_height, True

        return 0.0, False

    def run(self):
        print(f"[RUN] A iniciar captura de vídeo no barramento {self.camera_index}...")
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)

        # Forçar alta resolução se suportado pelo hardware da câmara
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            fps = 1.0 / (time.time() - self.prev_time + 1e-6)
            self.prev_time = time.time()

            master_id = self.face_db.get_current_master()

            faces = self.app.get(frame)

            for face in faces:
                x1, y1, x2, y2 = face.bbox.astype(int)

                if face.det_score < 0.60 or (y2 - y1) < 40:
                    continue

                embedding = face.normed_embedding
                pid = self.face_db.register(embedding)

                margin = int((y2 - y1) * 0.2)
                h, w = frame.shape[:2]
                cy1 = max(0, y1 - margin)
                cy2 = min(h, y2 + margin)
                cx1 = max(0, x1 - margin)
                cx2 = min(w, x2 + margin)

                face_crop = frame[cy1:cy2, cx1:cx2]

                ratio, found = self._get_mar(face_crop)
                if found and pid != "??":
                    self.speaker.push(pid, ratio)
                    if self.speaker.is_speaking(pid):
                        self.face_db.add_interaction(pid)

                score = self.face_db.get_score(pid)
                color = (
                    self.face_db.db[pid]["color"]
                    if pid in self.face_db.db
                    else (100, 100, 100)
                )
                is_master = pid == master_id and pid != "??"
                speaking = self.speaker.is_speaking(pid) if pid != "??" else False

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if is_master else 1)
                tag = "[A FALAR]" if speaking else "[SILENCIO]"
                label = f"{pid} {tag} Score:{score}"
                cv2.rectangle(
                    frame,
                    (x1, y1 - 25),
                    (x1 + len(label) * 8, y1),
                    (15, 15, 15),
                    cv2.FILLED,
                )
                cv2.putText(
                    frame,
                    label,
                    (x1 + 3, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )

                if is_master:
                    cv2.putText(
                        frame,
                        "ELEITO",
                        (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2,
                        cv2.LINE_AA,
                    )

                    # ──────────────────────────────────────────────
                    # Extração de Estado para Reinforcement Learning
                    # ──────────────────────────────────────────────
                    face_w = x2 - x1
                    face_h = y2 - y1
                    cx = x1 + (face_w // 2)

                    # Resolução física da câmara SOTA
                    IMG_WIDTH = 1600.0
                    IMG_CENTER_X = IMG_WIDTH / 2.0

                    # Setpoint de distância (Ajustar empiricamente no laboratório)
                    TARGET_FACE_H = 300.0

                    # Normalização [-1.0, 1.0]
                    norm_yaw_obs = (cx - IMG_CENTER_X) / IMG_CENTER_X
                    norm_depth_obs = (face_h - TARGET_FACE_H) / TARGET_FACE_H

                    # Payload a enviar para o Agente RL (via Sockets, ROS, etc.)
                    rl_state = {
                        "id": pid,
                        "yaw": round(norm_yaw_obs, 4),
                        "depth": round(norm_depth_obs, 4),
                        "visible": True,
                    }

                    # Para debug no terminal (comentar em produção para não afetar I/O)
                    # print(f"[RL STATE] {json.dumps(rl_state)}") Porem dizer se é esquerda ou direita do centro da imagem, e se está perto ou longe do setpoint
                    # Se yaw for negativo, é à esquerda do centro; se positivo, é à direita. Se depth for negativo, está mais perto do setpoint; se positivo, está mais longe.
                    print(
                        f"""[RL STATE] Yaw: {rl_state["yaw"]}  | Depth: {rl_state["depth"]})"""
                    )
                    # log.info(f"""[RL STATE] ID: {rl_state["id"]} | Yaw: {rl_state["yaw"]} | Depth: {rl_state["depth"]} | Visible: {rl_state["visible"]} | if yaw < 0: Esquerda do Centro | if yaw > 0: Direita do Centro | if depth < 0: Mais Perto do Setpoint | if depth > 0: Mais Longe do Setpoint""")

            if not master_id:
                state_msg, state_color = "A AVALIAR CANDIDATOS", (0, 255, 255)
            else:
                state_msg, state_color = f"ELEITO {master_id} A LIDERAR", (0, 255, 0)

            cv2.rectangle(frame, (10, 10), (450, 40), (0, 0, 0), cv2.FILLED)
            cv2.putText(
                frame,
                f"G1 STATE: {state_msg}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                state_color,
                2,
            )
            cv2.putText(
                frame,
                # Linha 414 — mudar o label
                f"DB: {len(self.face_db.db)} IDS | FPS: {fps:.1f} | CAM {self.camera_index} | Linux Desktop"(
                    10, 60
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )

            view = cv2.resize(frame, (1280, 720))
            cv2.imshow(f"G1 Perception - Cam {self.camera_index}", view)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # 1. Executa a sondagem no terminal
    selected_cam = select_camera()

    # 2. Arranca a pipeline visual injetando o ID correto
    G1InsightFacePerception(camera_index=selected_cam).run()
