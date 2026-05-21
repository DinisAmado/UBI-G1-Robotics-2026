#!/usr/bin/env python3
"""
main_hmi.py — Módulo HMI para integração com a Orquestração
Grupo 5 — Robótica Inteligente 2025/2026

Publica:    rt/hmi/intent    (Intent)    — intenção do operador
Subscreve:  rt/hmi/feedback  (Feedback)  — estado da orquestração

Integra o sistema HRI completo (hri_funciona.py):
  Microfone ZMQ → Whisper → Classificador → FSM → Intent DDS + TTS + LEDs

FIXES aplicados:
  1. DDS movido para dentro de HmiNode (não corre no import)
  2. MAPA_ACAO cobre agora ENTREGAR e SEGUIR
  3. MAPA_TARGET usa .get() com fallback explícito e log de aviso
  4. Import de OrchestrationState (enum hmi) clarificado
"""

import time
import logging
import os

from cyclonedds.domain import DomainParticipant
from cyclonedds.topic import Topic
from cyclonedds.pub import Publisher, DataWriter
from cyclonedds.sub import Subscriber, DataReader

# QoS e IDL partilhados com a orquestração
from qos_profiles import QOS_HMI
from idl_ri import (
    Header, Intent, Acao, Feedback, Status,
    OrchestrationState,   # enum rt.hmi (IDLE, WAITING_FOR_INTENT, ...)
)

# Componentes HRI
from hri_funciona import (
    WhisperModel,
    gravar,
    classificar,
    normalizar,
    falar,
    RobotSpeaker,
    LedController,
    frase_confirmacao,
    frase_execucao,
    frase_imediata,
    NOME_TARGET,
    ACOES_COM_CONFIRMACAO,
    ACOES_IMEDIATAS,
    WHISPER_MODEL,
)

# ==============================================================================
# CONFIGURAÇÃO
# ==============================================================================
DOMAIN_ID = 0
log = logging.getLogger("hmi")

# Mapeamento: ações HRI → Acao da orquestração
# FIX: adicionados ENTREGAR → Acao.ENTREGAR e SEGUIR → Acao.SEGUIR
MAPA_ACAO = {
    "TRAZER":         Acao.RECOLHER,
    "AGARRAR":        Acao.RECOLHER,
    "ENTREGAR":       Acao.ENTREGAR,   # ← era ausente
    "SEGUIR":         Acao.SEGUIR,     # ← era ausente
    "PARAR":          Acao.PARAR,
    "LARGAR":         Acao.LARGA,
    "ANDAR":          Acao.PARAR,      # sem equivalente direto — orquestrador ignora
    "RECUAR":         Acao.PARAR,
    "LEVANTAR":       Acao.PARAR,      # não mapeado diretamente
    "SENTAR":         Acao.PARAR,      # não mapeado diretamente
}

# Mapeamento: targets HRI → string para orquestração
# FIX: targets desconhecidos passam a gerar aviso em vez de silêncio
MAPA_TARGET = {
    "BOLA_DE_TENIS":   "bola_de_tenis",
    "CUBO_DE_RUBIK":   "cubo_de_rubik",
    "PASTA_DE_DENTES": "pasta_de_dentes",
    "NENHUM":          "",
    "DESCONHECIDO":    "",
}

# Mensagens de feedback do orquestrador → texto para o utilizador
FEEDBACK_ESTADO = {
    OrchestrationState.IDLE:                 "",
    OrchestrationState.WAITING_FOR_INTENT:   "Estou pronto para receber comandos.",
    OrchestrationState.LOCATING_OBJECT:      "A localizar o objeto, um momento.",
    OrchestrationState.NAVIGATING_TO_TABLE:  "A navegar até à mesa.",
    OrchestrationState.GRASPING_OBJECT:      "A agarrar o objeto.",
    OrchestrationState.NAVIGATING_TO_PERSON: "A trazer o objeto até si.",
    OrchestrationState.DELIVERING:           "A entregar o objeto.",
    OrchestrationState.RECOVERING:           "Ocorreu um problema, a tentar recuperar.",
    OrchestrationState.ABORTED:              "Operação cancelada.",
}


# ==============================================================================
# HMI NODE — DDS encapsulado numa classe (não corre no import)
# ==============================================================================
class HmiNode:
    """
    Encapsula toda a lógica DDS do HMI.
    O DomainParticipant só é criado quando HmiNode() é instanciado,
    nunca durante o import do módulo.
    """

    def __init__(self):
        self._seq = 0

        # FIX: DDS inicializado aqui, não a nível de módulo
        self._dp  = DomainParticipant(DOMAIN_ID)
        pub = Publisher(self._dp)
        sub = Subscriber(self._dp)

        t_intent   = Topic(self._dp, "rt/hmi/intent",   Intent,   qos=QOS_HMI)
        t_feedback = Topic(self._dp, "rt/hmi/feedback", Feedback, qos=QOS_HMI)

        self._w_intent   = DataWriter(pub, t_intent)
        self._r_feedback = DataReader(sub, t_feedback)

        log.info("HmiNode inicializado no domínio %d", DOMAIN_ID)

    def _make_header(self) -> Header:
        self._seq += 1
        return Header(timestamp_ns=time.time_ns(), frame_id="hmi", seq=self._seq)

    # ── Publicar Intent ───────────────────────────────────────────────────────

    def send_intent(self, acao: Acao, alvo: str, comando_grasping: str = "") -> None:
        intent = Intent(
            header=self._make_header(),
            acao=acao,
            alvo=alvo,
            comando_grasping=comando_grasping,
        )
        self._w_intent.write(intent)
        log.info("[INTENT] acao=%s  alvo=%s  grasping=%s", acao.name, alvo, comando_grasping)

    # ── Ler Feedback ──────────────────────────────────────────────────────────

    def poll_feedback(self) -> Feedback | None:
        samples = self._r_feedback.take(1)
        return samples[0] if samples else None

    # ── Converter HRI → Intent e publicar ────────────────────────────────────

    def publicar_intent_hri(self, action: str, target: str) -> bool:
        """
        Converte classificação HRI para Intent da orquestração e publica.
        Retorna True se publicado, False se a ação não tem mapeamento.
        """
        acao_orq = MAPA_ACAO.get(action)
        if acao_orq is None:
            log.warning("[HMI] Acao '%s' nao tem mapeamento para orquestracao", action)
            return False

        # FIX: aviso explícito para targets desconhecidos (evita intent silencioso)
        if target not in MAPA_TARGET:
            log.warning("[HMI] Target '%s' nao reconhecido — a usar string vazia", target)
        alvo_orq = MAPA_TARGET.get(target, "")
        grasping = alvo_orq if alvo_orq else ""

        self.send_intent(acao_orq, alvo=alvo_orq, comando_grasping=grasping)
        return True


# ==============================================================================
# PROCESSAR FEEDBACK
# ==============================================================================
def processar_feedback(fb: Feedback, speaker, leds) -> None:
    """Lê o feedback e apresenta ao utilizador via TTS e LEDs."""
    log.info("[FEEDBACK] status=%s  estado=%s  msg=%s",
             fb.status.name, fb.state.name, fb.message)

    msg_utilizador = fb.message if fb.message else FEEDBACK_ESTADO.get(fb.state, "")
    if not msg_utilizador:
        return

    if fb.status == Status.DONE:
        leds.falar()
    elif fb.status == Status.FAILED:
        leds.nao_percebeu()
    elif fb.state in (
        OrchestrationState.NAVIGATING_TO_TABLE,
        OrchestrationState.NAVIGATING_TO_PERSON,
        OrchestrationState.GRASPING_OBJECT,
        OrchestrationState.LOCATING_OBJECT,
    ):
        leds.pendente()

    falar(msg_utilizador, speaker, leds)


# ==============================================================================
# LOOP PRINCIPAL
# ==============================================================================
def main() -> None:
    logging.basicConfig(level=logging.INFO)
    log.info("HMI a iniciar no domínio %d", DOMAIN_ID)

    # DDS só inicializa aqui (dentro de main, não no import)
    hmi = HmiNode()

    print("A carregar Whisper...")
    whisper = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="float16")
    print("Whisper pronto!")

    speaker = RobotSpeaker()
    leds    = LedController(speaker.audio_client)

    pending = None

    print("=" * 52)
    print("   SISTEMA HMI -- UNITREE G1  (Ctrl+C para sair)")
    print("=" * 52 + "\n")

    falar("Olá! Eu sou o Johnny. Em que posso ajudá-lo?", speaker, leds)

    try:
        while True:
            # --- Verificar feedback do orquestrador ---
            fb = hmi.poll_feedback()
            if fb:
                processar_feedback(fb, speaker, leds)

            # --- Gravar e transcrever ---
            leds.ouvir()
            ficheiro = gravar()
            if not ficheiro:
                continue

            try:
                leds.pendente()
                segs, _ = whisper.transcribe(
                    ficheiro, language="pt",
                    beam_size=5, best_of=5,
                    temperature=0.0,
                    condition_on_previous_text=False,
                    initial_prompt=(
                        "Comandos possíveis em português: anda, para, recua, levanta-te, senta-te, "
                        "vira à esquerda, vira à direita, olha para mim, olha em frente, "
                        "vai buscar a bola de ténis, traz a bola de ténis, agarra a bola de ténis, "
                        "vai buscar o cubo de Rubik, traz o cubo de Rubik, agarra o cubo de Rubik, "
                        "vai buscar a pasta de dentes, traz a pasta de dentes, agarra a pasta de dentes, "
                        "sim, não, cancela, ok."
                    )
                )
                texto = "".join(s.text for s in segs).strip()

                if not texto:
                    print("[Whisper] Não percebi nada.")
                    continue

                print(f"[Utilizador]: {texto}")
                result = classificar(texto)
                action, target = result["action"], result["target"]
                print(f"[Classificação]: {action} / {target}")

                # --- FSM ---
                if action == "CONFIRMAR" and pending:
                    hmi.publicar_intent_hri(pending["action"], pending["target"])
                    resposta = frase_execucao(pending["action"], pending["target"])
                    pending = None

                elif action == "CONFIRMAR":
                    resposta = "Não há nenhuma ação pendente."

                elif action == "CANCELAR" and pending:
                    leds.cancelar(); time.sleep(1)
                    resposta = "Ok, fico aqui então."
                    pending = None

                elif action == "CANCELAR":
                    leds.cancelar(); time.sleep(1)
                    resposta = "Ok, sem problema."

                elif action in ACOES_COM_CONFIRMACAO:
                    pending = {"action": action, "target": target}
                    leds.pendente()
                    resposta = frase_confirmacao(action, target)

                elif action in ACOES_IMEDIATAS:
                    hmi.publicar_intent_hri(action, target)
                    resposta = frase_imediata(action)

                elif action == "DESCONHECIDA" and target != "NENHUM":
                    pending = {"action": "TRAZER", "target": target}
                    leds.pendente()
                    resposta = frase_confirmacao("TRAZER", target)

                else:
                    leds.nao_percebeu()
                    resposta = "Não percebi o comando. Podes repetir por favor?"

                print(f"[Johnny]: {resposta}")
                falar(resposta, speaker, leds)
                time.sleep(0.4)

            finally:
                if os.path.exists(ficheiro):
                    os.remove(ficheiro)

    except KeyboardInterrupt:
        leds.desligar()
        print("\nA desligar o Johnny... Até logo!")


if __name__ == "__main__":
    main()
