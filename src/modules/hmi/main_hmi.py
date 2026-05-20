#!/usr/bin/env python3
"""
main_hmi.py — Módulo HMI para integração com a Orquestração
Grupo 5 — Robótica Inteligente 2025/2026

Publica:    rt/hmi/intent    (Intent)    — intenção do operador
Subscreve:  rt/hmi/feedback  (Feedback)  — estado da orquestração

Integra o sistema HRI completo (hri_funciona.py):
  Microfone ZMQ → Whisper → Classificador → FSM → Intent DDS + TTS + LEDs
"""

import time
import logging
import asyncio
import os

from cyclonedds.domain import DomainParticipant
from cyclonedds.topic import Topic
from cyclonedds.pub import Publisher, DataWriter
from cyclonedds.sub import Subscriber, DataReader

# QoS e IDL partilhados com a orquestração
from qos_profiles import QOS_HMI
from idl_ri import Header, Intent, Acao, Feedback, OrchestrationState, Status

# Componentes HRI (reutiliza tudo do módulo principal)
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
MAPA_ACAO = {
    "TRAZER":         Acao.RECOLHER,   # vai buscar e traz
    "AGARRAR":        Acao.RECOLHER,   # agarra objeto
    "PARAR":          Acao.PARAR,      # para imediatamente
    "LARGAR":         Acao.LARGA,      # larga objeto
    "ANDAR":          Acao.PARAR,      # sem equivalente direto — orquestrador ignora
    "RECUAR":         Acao.PARAR,      # recuar
    "LEVANTAR":       Acao.PARAR,      # não mapeado diretamente
    "SENTAR":         Acao.PARAR,      # não mapeado diretamente
}

# Mapeamento: targets HRI → string para orquestração
MAPA_TARGET = {
    "BOLA_DE_TENIS":   "bola_de_tenis",
    "CUBO_DE_RUBIK":   "cubo_de_rubik",
    "PASTA_DE_DENTES": "pasta_de_dentes",
    "NENHUM":          "",
    "DESCONHECIDO":    "",
}

# Mensagens de feedback do orquestrador → texto para o utilizador
# Baseado em OrchestrationState (fb.state) e Status (fb.status)
FEEDBACK_ESTADO = {
    OrchestrationState.IDLE:                 "",   # silêncio quando inativo
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
# DDS — SETUP
# ==============================================================================
dp  = DomainParticipant(DOMAIN_ID)
pub = Publisher(dp)
sub = Subscriber(dp)

t_intent   = Topic(dp, "rt/hmi/intent",   Intent,   qos=QOS_HMI)
t_feedback = Topic(dp, "rt/hmi/feedback", Feedback, qos=QOS_HMI)

w_intent   = DataWriter(pub, t_intent)
r_feedback = DataReader(sub, t_feedback)

seq = 0

# ==============================================================================
# PUBLICAR INTENT
# ==============================================================================
def send_intent(acao: Acao, alvo: str, comando_grasping: str = "") -> None:
    global seq
    seq += 1
    intent = Intent(
        header=Header(
            timestamp_ns=time.time_ns(),
            frame_id="hmi",
            seq=seq,
        ),
        acao=acao,
        alvo=alvo,
        comando_grasping=comando_grasping,
    )
    w_intent.write(intent)
    log.info("[INTENT] acao=%s  alvo=%s  grasping=%s", acao.name, alvo, comando_grasping)

# ==============================================================================
# LER FEEDBACK DO ORQUESTRADOR
# ==============================================================================
def poll_feedback() -> Feedback | None:
    samples = r_feedback.take(1)
    return samples[0] if samples else None

def processar_feedback(fb: Feedback, speaker, leds) -> None:
    """Lê o feedback e apresenta ao utilizador via TTS e LEDs.

    fb.status — Status.RUNNING / Status.DONE / Status.FAILED
    fb.state  — OrchestrationState (fase atual da orquestração)
    fb.message — mensagem livre do orquestrador (pode estar vazia)
    """
    log.info("[FEEDBACK] status=%s  estado=%s  msg=%s",
             fb.status.name, fb.state.name, fb.message)

    # Prioridade: mensagem livre do orquestrador, depois mapeamento de estado
    msg_utilizador = fb.message if fb.message else FEEDBACK_ESTADO.get(fb.state, "")

    if not msg_utilizador:
        return

    # Cor do LED conforme o resultado
    if fb.status == Status.DONE:
        leds.falar()        # verde — sucesso
    elif fb.status == Status.FAILED:
        leds.nao_percebeu() # vermelho — falhou
    elif fb.state in (OrchestrationState.NAVIGATING_TO_TABLE,
                      OrchestrationState.NAVIGATING_TO_PERSON,
                      OrchestrationState.GRASPING_OBJECT,
                      OrchestrationState.LOCATING_OBJECT):
        leds.pendente()     # laranja — a executar

    falar(msg_utilizador, speaker, leds)

# ==============================================================================
# MAPEAMENTO HRI → ORQUESTRAÇÃO
# ==============================================================================
def publicar_intent_hri(texto: str, action: str, target: str) -> None:
    """Converte classificação HRI para Intent da orquestração e publica."""
    acao_orq = MAPA_ACAO.get(action)
    if acao_orq is None:
        log.warning("[HMI] Acao '%s' nao tem mapeamento para orquestracao", action)
        return

    alvo_orq = MAPA_TARGET.get(target, "")
    # comando_grasping = nome do objeto para o módulo de grasping
    grasping = alvo_orq if alvo_orq else ""

    send_intent(acao_orq, alvo=alvo_orq, comando_grasping=grasping)

# ==============================================================================
# LOOP PRINCIPAL
# ==============================================================================
def main() -> None:
    logging.basicConfig(level=logging.INFO)
    log.info("HMI a iniciar no domínio %d", DOMAIN_ID)

    print("A carregar Whisper...")
    whisper = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="float16")
    print("Whisper pronto!")

    speaker = RobotSpeaker()
    leds    = LedController(speaker.audio_client)

    historico = []
    pending   = None

    print("=" * 52)
    print("   SISTEMA HMI -- UNITREE G1  (Ctrl+C para sair)")
    print("=" * 52 + "\n")

    # Saudação inicial
    falar("Olá! Eu sou o Johnny. Em que posso ajudá-lo?", speaker, leds)

    try:
        while True:
            # --- Verificar feedback do orquestrador ---
            fb = poll_feedback()
            if fb:
                processar_feedback(fb, speaker, leds)

            # --- Gravar e transcrever ---
            leds.ouvir()
            ficheiro = gravar()
            if not ficheiro:
                continue

            try:
                leds.pendente()  # laranja durante processamento Whisper
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

                # --- Máquina de estados FSM ---
                if action == "CONFIRMAR" and pending:
                    publicar_intent_hri(pending["texto"], pending["action"], pending["target"])
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
                    pending = {"action": action, "target": target, "texto": texto}
                    leds.pendente()
                    resposta = frase_confirmacao(action, target)

                elif action in ACOES_IMEDIATAS:
                    publicar_intent_hri(texto, action, target)
                    resposta = frase_imediata(action)

                elif action == "DESCONHECIDA" and target != "NENHUM":
                    # Detetou objeto mas não percebeu ação — assume TRAZER
                    pending = {"action": "TRAZER", "target": target, "texto": texto}
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
