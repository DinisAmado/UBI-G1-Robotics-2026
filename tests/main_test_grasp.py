#!/usr/bin/env python3
"""
teste_visao_grasping.py — Teste integrado: Visão de Objetos + Orquestração + Grasping
Grupo 5 — Robótica Inteligente 2025/2026

Módulos reais a correr em paralelo:
  - main2.py          (orquestrador)
  - main__3_.py       (visão de objetos — câmara real + YOLO)
  - main_grasp.py     (grasping — braço real)

O teste apenas:
  1. Publica Intent (simula HMI — "vai buscar a bola_de_tenis")
  2. Monitoriza Feedback do orquestrador e mostra as transições de fase
  3. Publica NavStatus DONE quando necessário (simula navegação — robô não anda)

Fluxo esperado:
  IDLE → WAITING_FOR_INTENT
       → LOCATING_OBJECT     (main__3_.py deteta o objeto com a câmara)
       → NAVIGATING_TO_TABLE (teste responde DONE imediatamente — sem andar)
       → GRASPING_OBJECT     (main_grasp.py agarra o objeto)
       → NAVIGATING_TO_PERSON (teste responde DONE imediatamente — sem andar)
       → DELIVERING          (main_grasp.py entrega)
       → IDLE

Pré-requisitos:
  - main2.py a correr (Terminal 1)
  - main__3_.py a correr com câmara ligada (Terminal 2)
  - main_grasp.py a correr com robô ligado (Terminal 3)
  - Este teste no Terminal 4
"""

import sys
import os
import time
import logging
import threading

# ── Path para src/ ────────────────────────────────────────────────────────────
_pasta_atual = os.path.dirname(os.path.abspath(__file__))
_pasta_src   = os.path.abspath(os.path.join(_pasta_atual, '..', '..'))
if _pasta_src not in sys.path:
    sys.path.insert(0, _pasta_src)

from cyclonedds.domain import DomainParticipant
from cyclonedds.topic  import Topic
from cyclonedds.pub    import Publisher, DataWriter
from cyclonedds.sub    import Subscriber, DataReader

from qos_profiles import (
    QOS_HMI, QOS_NAV, QOS_VISION, QOS_ORCHESTRATION,
)
from idl_ri import (
    Header, Status,
    Intent, Acao, Feedback, OrchestrationState as HmiState,
    NavStatusMsg as NavStatus,
    Objects as VisionObjects,
    Persons as VisionPersons, PersonDetection,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] TESTE: %(message)s"
)
log = logging.getLogger("teste_visao_grasp")

DOMAIN_ID   = 0
OBJETO_ALVO = "bola_de_tenis"   # tem de coincidir com o nome detetado pelo YOLO
PESSOA_ID   = "pessoa_1"


class TesteVisaoGrasping:

    def __init__(self):
        self._seq = 0
        self._dp  = DomainParticipant(DOMAIN_ID)
        pub = Publisher(self._dp)
        sub = Subscriber(self._dp)

        # ── Writers ───────────────────────────────────────────────────────────
        # Simula HMI
        self._w_intent     = DataWriter(pub, Topic(self._dp, "rt/hmi/intent",    Intent,       qos=QOS_HMI))
        # Simula navegação (robô não anda, responde DONE imediatamente)
        self._w_nav_status = DataWriter(pub, Topic(self._dp, "rt/nav/status",    NavStatus,    qos=QOS_NAV))
        # Simula visão de pessoa (para NAVIGATING_TO_PERSON)
        self._w_vision_per = DataWriter(pub, Topic(self._dp, "rt/vision/persons",VisionPersons,qos=QOS_VISION))

        # ── Readers ───────────────────────────────────────────────────────────
        self._r_feedback   = DataReader(sub, Topic(self._dp, "rt/hmi/feedback",  Feedback,     qos=QOS_HMI))

        # ── Estado interno ────────────────────────────────────────────────────
        self._fase_atual   = ""
        self._fases_vistas = []
        self._running      = False
        self._lock         = threading.Lock()

    # ── Utilitários ───────────────────────────────────────────────────────────

    def _header(self) -> Header:
        self._seq += 1
        return Header(timestamp_ns=time.time_ns(), frame_id="teste", seq=self._seq)

    def _set_fase(self, fase: str):
        with self._lock:
            if fase != self._fase_atual:
                self._fase_atual = fase
                self._fases_vistas.append(fase)
                log.info(">>> FASE: %s", fase)

    def _get_fase(self) -> str:
        with self._lock:
            return self._fase_atual

    # ── Thread: monitoriza Feedback ───────────────────────────────────────────

    def _monitor_feedback(self):
        while self._running:
            for fb in self._r_feedback.take():
                self._set_fase(fb.state.name)
                log.info(
                    "[FEEDBACK] status=%-8s  fase=%-25s  msg=%s",
                    fb.status.name, fb.state.name, fb.message
                )
            time.sleep(0.05)

    # ── Thread: publica visão de pessoa em background ─────────────────────────

    def _publicar_pessoa_continuo(self):
        """Publica deteção de pessoa a 2Hz — necessário para NAVIGATING_TO_PERSON."""
        while self._running:
            msg = VisionPersons(
                header=self._header(),
                detections=[
                    PersonDetection(
                        id=PESSOA_ID,
                        lip_movement_confidence=0.90,
                    )
                ]
            )
            self._w_vision_per.write(msg)
            time.sleep(0.5)

    # ── Esperar por fase ──────────────────────────────────────────────────────

    def _esperar_fase(self, fase: str, timeout: float = 60.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._get_fase() == fase:
                return True
            time.sleep(0.1)
        log.warning("TIMEOUT a esperar por fase=%s (estava em %s)", fase, self._get_fase())
        return False

    # ── Publicadores ──────────────────────────────────────────────────────────

    def _pub_intent(self):
        msg = Intent(
            header=self._header(),
            acao=Acao.RECOLHER,
            alvo=OBJETO_ALVO,
            comando_grasping=OBJETO_ALVO,
        )
        self._w_intent.write(msg)
        log.info("[HMI] Intent publicado → RECOLHER '%s'", OBJETO_ALVO)

    def _pub_nav_done(self, reason: str = ""):
        msg = NavStatus(
            header=self._header(),
            status=Status.DONE,
            reason=reason,
            progress=1.0,
        )
        self._w_nav_status.write(msg)
        log.info("[NAV] Status DONE publicado (reason='%s')", reason)

    # ── Sequência principal ───────────────────────────────────────────────────

    def run(self):
        self._running = True

        # Threads de background
        monitor  = threading.Thread(target=self._monitor_feedback,       daemon=True)
        t_pessoa = threading.Thread(target=self._publicar_pessoa_continuo, daemon=True)
        monitor.start()
        t_pessoa.start()

        try:
            log.info("=" * 60)
            log.info("  TESTE: VISÃO OBJETOS + ORQUESTRAÇÃO + GRASPING")
            log.info("  Objeto alvo: %s", OBJETO_ALVO)
            log.info("  Certifica-te que correm em paralelo:")
            log.info("    Terminal 1: python main2.py")
            log.info("    Terminal 2: python main__3_.py")
            log.info("    Terminal 3: python main_grasp.py")
            log.info("=" * 60)

            # ── PASSO 1: Aguardar orquestrador pronto ─────────────────────────
            log.info("--- [1/5] A aguardar WAITING_FOR_INTENT ---")
            if not self._esperar_fase("WAITING_FOR_INTENT", timeout=15.0):
                log.error("Orquestrador não ficou pronto. Está o main2.py a correr?")
                return

            # ── PASSO 2: Publicar Intent (simula HMI) ─────────────────────────
            log.info("--- [2/5] A publicar Intent ---")
            time.sleep(0.5)
            self._pub_intent()

            # ── PASSO 3: Aguardar LOCATING_OBJECT ─────────────────────────────
            # O main__3_.py (câmara real) deteta o objeto automaticamente.
            # Não há nada a simular aqui — só aguardar.
            log.info("--- [3/5] A aguardar LOCATING_OBJECT ---")
            log.info("          O main__3_.py vai detetar '%s' com a câmara...", OBJETO_ALVO)
            if not self._esperar_fase("LOCATING_OBJECT", timeout=10.0):
                log.error("Orquestrador não entrou em LOCATING_OBJECT")
                return

            # ── PASSO 4: Aguardar NAVIGATING_TO_TABLE e responder DONE ────────
            # O orquestrador transita para NAVIGATING_TO_TABLE após a visão detetar.
            # Como não queremos que o robô ande, respondemos DONE imediatamente.
            log.info("--- [4/5] A aguardar NAVIGATING_TO_TABLE ---")
            if not self._esperar_fase("NAVIGATING_TO_TABLE", timeout=30.0):
                log.error("Orquestrador não entrou em NAVIGATING_TO_TABLE")
                return

            log.info("          Nav: a responder DONE (robô não anda)")
            time.sleep(0.5)
            self._pub_nav_done("chegou_mesa")

            # ── PASSO 5: Aguardar GRASPING_OBJECT ─────────────────────────────
            # O main_grasp.py trata de tudo a partir daqui.
            # Só monitorizamos até ao fim.
            log.info("--- [5/5] A aguardar GRASPING_OBJECT ---")
            log.info("          O main_grasp.py vai agarrar o objeto...")
            if not self._esperar_fase("GRASPING_OBJECT", timeout=10.0):
                log.error("Orquestrador não entrou em GRASPING_OBJECT")
                return

            # Aguardar NAVIGATING_TO_PERSON e responder DONE
            log.info("          A aguardar NAVIGATING_TO_PERSON...")
            if not self._esperar_fase("NAVIGATING_TO_PERSON", timeout=60.0):
                log.error("Orquestrador não entrou em NAVIGATING_TO_PERSON")
                return

            log.info("          Nav: a responder DONE (robô não anda)")
            time.sleep(0.5)
            self._pub_nav_done("chegou_pessoa")

            # Aguardar DELIVERING e depois IDLE
            log.info("          A aguardar DELIVERING...")
            if not self._esperar_fase("DELIVERING", timeout=10.0):
                log.error("Orquestrador não entrou em DELIVERING")
                return

            log.info("          O main_grasp.py vai entregar o objeto...")
            if not self._esperar_fase("IDLE", timeout=30.0):
                log.error("Orquestrador não regressou a IDLE")
                return

            # ── Resultado ─────────────────────────────────────────────────────
            log.info("=" * 60)
            log.info("  TESTE CONCLUÍDO COM SUCESSO!")
            log.info("  Fases: %s", " → ".join(self._fases_vistas))
            log.info("=" * 60)

        except KeyboardInterrupt:
            log.warning("Teste interrompido manualmente.")
        finally:
            self._running = False
            monitor.join(timeout=1.0)
            t_pessoa.join(timeout=1.0)


if __name__ == "__main__":
    TesteVisaoGrasping().run()
