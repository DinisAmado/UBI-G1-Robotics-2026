from collections import deque

import cv2
import face_recognition
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class G1FullFeedbackPerception:
    def __init__(self):
        self.device = "cpu"
        self.state = "SEARCHING"  # Estados: SEARCHING, MISSION, RECOGNIZING

        # 1. MediaPipe
        model_path = "face_landmarker.task"
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=3,  # Permitir até 3 pessoas
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

        # Vars
        self.master_encoding = None
        self.speaker_histories = {}  # {face_index: deque}
        self.var_threshold = 0.0002  # Isto depois vai depender da distancia

    def get_all_faces(self, frame):
        """Extrai dados de todas as faces presentes."""
        h, w = frame.shape[:2]
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )
        res = self.detector.detect(mp_image)

        faces_data = []
        if res.face_landmarks:
            for i, lms in enumerate(res.face_landmarks):
                x_coords = [int(lm.x * w) for lm in lms]
                y_coords = [int(lm.y * h) for lm in lms]
                bbox = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
                aperture = abs(lms[13].y - lms[14].y) / abs(lms[10].y - lms[152].y)
                faces_data.append({"id": i, "bbox": bbox, "aperture": aperture})
        return faces_data

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            faces = self.get_all_faces(frame)

            for face in faces:
                idx = face["id"]
                x1, y1, x2, y2 = face["bbox"]
                aperture = face["aperture"]

                # Gestão de Histórico por Face
                if idx not in self.speaker_histories:
                    self.speaker_histories[idx] = deque(maxlen=15)
                self.speaker_histories[idx].append(aperture)

                # Cálculo de Variância (Fala)
                v = (
                    np.var(self.speaker_histories[idx])
                    if len(self.speaker_histories[idx]) > 5
                    else 0
                )
                is_speaking = v > self.var_threshold

                color = (0, 255, 0) if is_speaking else (255, 255, 255)
                if self.state == "MISSION":
                    color = (255, 165, 0)  # Laranja em missão

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                status = "A FALAR" if is_speaking else "SILENCIO"
                info_txt = f"ID:{idx} | {status} | Var:{v * 1000:.2f}"
                cv2.putText(
                    frame,
                    info_txt,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                )

                # --- LÓGICA DE TRANSIÇÃO ---
                if self.state == "SEARCHING" and is_speaking:
                    # Tenta capturar o mestre
                    try:
                        face_img = frame[max(0, y1) : y2, max(0, x1) : x2]
                        if face_img.size > 0:
                            small_face = cv2.resize(face_img, (128, 128))
                            rgb_face = cv2.cvtColor(small_face, cv2.COLOR_BGR2RGB)
                            encs = face_recognition.face_encodings(
                                rgb_face, [(0, 128, 128, 0)]
                            )
                            if encs:
                                self.master_encoding = encs[0]
                                self.state = "MISSION"
                                print(f"[OK] Pessoa: ID {idx}")
                    except:
                        pass

                # --- RECONHECIMENTO ---
                if self.state == "RECOGNIZING":
                    try:
                        face_img = frame[max(0, y1) : y2, max(0, x1) : x2]
                        small_face = cv2.resize(face_img, (128, 128))
                        rgb_face = cv2.cvtColor(small_face, cv2.COLOR_BGR2RGB)
                        encs = face_recognition.face_encodings(
                            rgb_face, [(0, 128, 128, 0)]
                        )
                        if encs:
                            match = face_recognition.compare_faces(
                                [self.master_encoding], encs[0], tolerance=0.5
                            )
                            if match[0]:
                                cv2.putText(
                                    frame,
                                    "!!! PESSOA IDENTIFICADA !!!",
                                    (x1, y2 + 25),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7,
                                    (0, 255, 0),
                                    2,
                                )
                    except:
                        pass

            # Overlay Geral de Estado
            cv2.rectangle(frame, (10, 10), (250, 40), (0, 0, 0), -1)
            cv2.putText(
                frame, f"SISTEMA G1: {self.state}", (20, 30), 1, 1, (255, 255, 255), 1
            )

            cv2.imshow("G1 Perception - Full Feedback", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("m"):
                self.state = "RECOGNIZING"
            if key == ord("r"):
                self.state = "SEARCHING"
                self.master_encoding = None
            if key == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    G1FullFeedbackPerception().run()
