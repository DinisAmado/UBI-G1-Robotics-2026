#!/usr/bin/env python3
"""
hri_unitree.py — Sistema HRI completo para o Robô Unitree G1
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Classificação: palavras-chave (instantânea, sem GPU)
Conversa:      Ollama local  (muito mais rápido que transformers em CPU)

Pré-requisitos:
  1. pip install faster-whisper speechrecognition pygame edge-tts cyclonedds ollama
  2. Ollama instalado e a correr: https://ollama.com
  3. Modelo descarregado:  ollama pull qwen2.5:1.5b

Para listar microfones:  python hri_unitree.py --list-mics
"""

import os
import re
import sys
import asyncio
import unicodedata
from datetime import datetime
from dataclasses import dataclass
from typing import Optional

# ── Listar microfones e sair ───────────────────────────────────────────────────
if "--list-mics" in sys.argv:
    import speech_recognition as sr
    print("Microfones disponíveis:")
    for i, name in enumerate(sr.Microphone.list_microphone_names()):
        print(f"  [{i}] {name}")
    sys.exit(0)

import pygame
import speech_recognition as sr
from faster_whisper import WhisperModel
import edge_tts
import ollama as ollama_client
from cyclonedds.idl import IdlStruct
from cyclonedds.domain import DomainParticipant
from cyclonedds.topic import Topic
from cyclonedds.pub import DataWriter


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURAÇÃO
# ══════════════════════════════════════════════════════════════════════════════
WHISPER_MODEL    = "medium"         # "small" (rápido) ou "medium" (mais preciso para PT)
OLLAMA_MODEL     = "qwen2.5:1.5b"  # modelo de conversa
MIC_DEVICE_INDEX = None             # None = microfone padrão. Muda para 0,1,2... se falhar
MIC_THRESHOLD    = 1500             # Aumenta se ouvir ruído, baixa se não detetar voz
TOPIC_NAME       = "HRICommands"
AUDIO_TEMP       = "temp_hri.wav"
AUDIO_RESP       = "resposta_hri.mp3"

# Ações que precisam de confirmação antes de publicar no DDS
ACOES_COM_CONFIRMACAO = {"IR_BUSCAR", "TRAZER", "AGARRAR"}

# Ações que publicam imediatamente no DDS
ACOES_IMEDIATAS = {
    "ANDAR", "PARAR", "RECUAR", "LEVANTAR", "SENTAR",
    "VIRAR_ESQUERDA", "VIRAR_DIREITA", "OLHAR_INTERLOCUTOR",
    "OLHAR_FRENTE", "CUMPRIMENTAR", "APRESENTAR", "ESTADO_ATUAL", "REPETIR",
    "LARGAR",
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. TIPO DDS
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class HRICommand(IdlStruct):
    source: str
    original_text: str
    action: str
    target: str
    confirmed: bool
    timestamp: str


# ══════════════════════════════════════════════════════════════════════════════
# 2. CLASSIFICADOR DE PALAVRAS-CHAVE (instantâneo, sem GPU)
# ══════════════════════════════════════════════════════════════════════════════
def normalizar(texto: str) -> str:
    """Remove acentos e coloca em minúsculas para comparação robusta."""
    return "".join(
        c for c in unicodedata.normalize("NFD", texto.lower())
        if unicodedata.category(c) != "Mn"
    )

def contem(texto: str, frase: str) -> bool:
    """Verifica se a frase existe no texto com word boundaries (evita 'ola' dentro de 'bola')."""
    if " " in frase:
        return frase in texto  # frases compostas: substring simples chega
    return bool(re.search(r"(?<![a-z])" + re.escape(frase) + r"(?![a-z])", texto))

# Cada entrada: (lista de palavras-chave, ACTION, TARGET_override ou None)
# Ordem importa — mais específico primeiro
REGRAS = [
    # ── Buscar / Trazer / Agarrar ─────────────────────────────────────────────
    (["vai buscar", "ir buscar", "busca"],               "IR_BUSCAR",         None),
    (["traz", "traze", "traga"],                         "TRAZER",            None),
    (["agarra", "pega", "apanha"],                       "AGARRAR",           None),
    (["larga", "larga isso", "larga ai"],                "LARGAR",            None),
    # ── Movimento ─────────────────────────────────────────────────────────────
    (["anda", "avanca", "vai para a frente", "caminha"], "ANDAR",             "NENHUM"),
    (["para", "stop", "fica quieto"],                    "PARAR",             "NENHUM"),
    (["recua", "vai para tras"],                         "RECUAR",            "NENHUM"),
    (["vira a esquerda", "esquerda"],                    "VIRAR_ESQUERDA",    "NENHUM"),
    (["vira a direita", "direita"],                      "VIRAR_DIREITA",     "NENHUM"),
    (["levanta", "levanta-te", "levanta te"],            "LEVANTAR",          "NENHUM"),
    (["senta", "senta-te", "senta te"],                  "SENTAR",            "NENHUM"),
    # ── Olhar ─────────────────────────────────────────────────────────────────
    (["olha para mim", "olha para a pessoa"],            "OLHAR_INTERLOCUTOR","NENHUM"),
    (["olha em frente", "olha para a frente"],           "OLHAR_FRENTE",      "NENHUM"),
    # ── Social ────────────────────────────────────────────────────────────────
    (["cumprimenta", "cumprimento", "ola"],              "CUMPRIMENTAR",      "NENHUM"),
    (["quem es", "apresenta-te", "como te chamas"],      "APRESENTAR",        "NENHUM"),
    (["como estas", "estado atual"],                     "ESTADO_ATUAL",      "NENHUM"),
    (["repete", "diz outra vez"],                        "REPETIR",           "NENHUM"),
    # ── Confirmação / Cancelamento ────────────────────────────────────────────
    (["sim", "claro", "faz isso", "ok", "pode ser",
      "isso mesmo", "exato", "vai", "faz",
      "por favor", "faz favor", "se faz favor",
      "quero", "quero sim", "exatamente"],               "CONFIRMAR",         "NENHUM"),
    (["nao", "cancela", "esquece", "afinal nao"],        "CANCELAR",          "NENHUM"),
]

REGRAS_TARGET = [
    (["bola de tenis", "bola de ténis", "bola", "tenis", "boletenas"], "BOLA_DE_TENIS"),
    (["cubo de rubik", "cubo magico", "cubo",  "rubik"],               "CUBO_DE_RUBIK"),
    (["pasta de dentes", "pasta",      "dentes"],                      "PASTA_DE_DENTES"),
]

def classificar(texto: str) -> dict:
    t = normalizar(texto)
    action = "DESCONHECIDA"
    target = "NENHUM"

    # Detetar target (independente da ação)
    for palavras, tgt in REGRAS_TARGET:
        if any(contem(t, p) for p in palavras):
            target = tgt
            break

    # Detetar ação
    for palavras, act, tgt_override in REGRAS:
        if any(contem(t, p) for p in palavras):
            action = act
            if tgt_override is not None:
                target = tgt_override
            break

    # Se ação precisa de objeto mas não detetou nenhum
    if action in ACOES_COM_CONFIRMACAO and target == "NENHUM":
        target = "DESCONHECIDO"

    return {"action": action, "target": target}


# ══════════════════════════════════════════════════════════════════════════════
# 3. CONVERSA COM OLLAMA
# ══════════════════════════════════════════════════════════════════════════════
SYSTEM_PROMPT = (
    "Tu és o Johnny, um robô Unitree em Portugal. "
    "Fala sempre em Português de Portugal (pt-PT). Sê muito breve e simpático. "
    "Nunca uses mais de 2 frases seguidas. "
    "Se não perceberes o comando, diz que não percebeste e pede para repetir."
)

def conversar(historico: list) -> str:
    print("A pensar...", end="", flush=True)
    try:
        resposta = ollama_client.chat(model=OLLAMA_MODEL, messages=historico)
        texto = resposta["message"]["content"].strip()
        print(f"\r                    \r", end="")
        return texto
    except Exception as e:
        print(f"\n[ERRO Ollama] {e}")
        print("     Garante que o Ollama está a correr: ollama serve")
        return "Desculpa, tive um problema."


# ══════════════════════════════════════════════════════════════════════════════
# 4. ÁUDIO
# ══════════════════════════════════════════════════════════════════════════════
def gravar() -> Optional[str]:
    r = sr.Recognizer()
    r.dynamic_energy_threshold = False  # CRÍTICO: não deixar calibrar para ~6
    r.energy_threshold = MIC_THRESHOLD
    r.pause_threshold = 1.2

    with sr.Microphone(device_index=MIC_DEVICE_INDEX) as source:
        print(f"\nÀ escuta... fala agora!")
        try:
            audio = r.listen(source, timeout=10, phrase_time_limit=12)
            with open(AUDIO_TEMP, "wb") as f:
                f.write(audio.get_wav_data())
            return AUDIO_TEMP
        except sr.WaitTimeoutError:
            print("Timeout — ninguém falou.")
            return None


def falar(texto: str) -> None:
    async def _gerar():
        await edge_tts.Communicate(texto, "pt-PT-DuarteNeural").save(AUDIO_RESP)
    asyncio.run(_gerar())
    pygame.mixer.init()
    pygame.mixer.music.load(AUDIO_RESP)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    pygame.mixer.quit()


# ══════════════════════════════════════════════════════════════════════════════
# 5. DDS
# ══════════════════════════════════════════════════════════════════════════════
def publicar_dds(writer, texto: str, action: str, target: str) -> None:
    cmd = HRICommand(
        source="HRI",
        original_text=texto,
        action=action,
        target=target,
        confirmed=True,
        timestamp=datetime.now().isoformat(timespec="seconds"),
    )
    writer.write(cmd)
    print(f"[DDS] PUBLICADO → action={action}  target={target}")


# Nome legível de cada target para as frases do Johnny
NOME_TARGET = {
    "BOLA_DE_TENIS":   "a bola de ténis",
    "CUBO_DE_RUBIK":   "o cubo mágico",
    "PASTA_DE_DENTES": "a pasta de dentes",
    "DESCONHECIDO":    "o objeto",
    "NENHUM":          "isso",
}

# Frase de confirmação por ação
def frase_confirmacao(action: str, target: str) -> str:
    nome = NOME_TARGET.get(target, "o objeto")
    if action == "TRAZER":
        return f"Queres que eu traga {nome}?"
    elif action == "IR_BUSCAR":
        return f"Queres que eu vá buscar {nome}?"
    elif action == "AGARRAR":
        return f"Queres que eu agarre {nome}?"
    elif action == "LARGAR":
        return f"Queres que eu large {nome}?"
    return f"Confirmas que queres que eu execute esta ação com {nome}?"

# Frase de execução (depois do "sim")
def frase_execucao(action: str, target: str) -> str:
    nome = NOME_TARGET.get(target, "o objeto")
    if action == "TRAZER":
        return f"Combinado! Vou já trazer {nome}."
    elif action == "IR_BUSCAR":
        return f"Combinado! Vou já buscar {nome}."
    elif action == "AGARRAR":
        return f"Combinado! Vou agarrar {nome}."
    elif action == "LARGAR":
        return f"Combinado! Vou largar {nome}."
    return "Combinado, vou já!"

# Frase para ações imediatas
def frase_imediata(action: str) -> str:
    return {
        "ANDAR":             "Ok, a andar!",
        "PARAR":             "Ok, paro aqui.",
        "RECUAR":            "Ok, a recuar.",
        "LEVANTAR":          "Ok, a levantar!",
        "SENTAR":            "Ok, a sentar.",
        "VIRAR_ESQUERDA":    "Ok, a virar à esquerda.",
        "VIRAR_DIREITA":     "Ok, a virar à direita.",
        "OLHAR_INTERLOCUTOR":"Ok, a olhar para ti.",
        "OLHAR_FRENTE":      "Ok, a olhar em frente.",
        "CUMPRIMENTAR":      "Olá! Muito prazer!",
        "APRESENTAR":        "Olá! Sou o Johnny, um robô Unitree. É um prazer conhecer-te!",
        "ESTADO_ATUAL":      "Estou operacional e pronto para ajudar!",
        "REPETIR":           "Claro, repito!",
        "LARGAR":            "Ok, a largar!",
    }.get(action, "Ok!")


# ══════════════════════════════════════════════════════════════════════════════
# 6. LOOP PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("A carregar Whisper...")
    whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    print("Whisper pronto!")

    print(f"A verificar Ollama ({OLLAMA_MODEL})...")
    try:
        ollama_client.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": "ok"}])
        print("Ollama pronto!\n")
    except Exception:
        print(f"\n[ERRO] Não consegui ligar ao Ollama!")
        print(f"  1. Instala em: https://ollama.com")
        print(f"  2. Corre num terminal: ollama serve")
        print(f"  3. Descarrega o modelo: ollama pull {OLLAMA_MODEL}")
        sys.exit(1)

    participant = DomainParticipant()
    topic = Topic(participant, TOPIC_NAME, HRICommand)
    writer = DataWriter(participant, topic)

    historico = [{"role": "system", "content": SYSTEM_PROMPT}]
    pending = None  # type: Optional[dict]  -- ação à espera de confirmação

    print("═" * 52)
    print("   SISTEMA HRI — UNITREE G1  (Ctrl+C para sair)")
    print("═" * 52 + "\n")

    try:
        while True:
            # 1. Gravar
            ficheiro = gravar()
            if not ficheiro:
                continue

            try:
                # 2. Transcrever
                segs, _ = whisper.transcribe(ficheiro, language="pt")
                texto = "".join(s.text for s in segs).strip()

                if not texto:
                    print("[Whisper] Não percebi nada.")
                    continue

                print(f"[Utilizador]: {texto}")

                # 3. Classificar (instantâneo)
                result = classificar(texto)
                action, target = result["action"], result["target"]
                print(f"[Classificação]: {action} / {target}")

                # 4. State machine — respostas estruturadas hardcoded (não dependem do LLM)
                resposta = None

                if action == "CONFIRMAR" and pending:
                    publicar_dds(writer, pending["texto"], pending["action"], pending["target"])
                    resposta = frase_execucao(pending["action"], pending["target"])
                    pending = None

                elif action == "CONFIRMAR" and not pending:
                    # Confirmação sem nada pendente — ignorar
                    resposta = "Não há nenhuma ação pendente para confirmar."

                elif action == "CANCELAR" and pending:
                    print("[X] Ação cancelada.")
                    resposta = "Ok, fico aqui então."
                    pending = None

                elif action == "CANCELAR":
                    resposta = "Ok, sem problema."

                elif action in ACOES_COM_CONFIRMACAO:
                    pending = {"action": action, "target": target, "texto": texto}
                    print(f"Pendente: {action}/{target} — à espera de confirmação")
                    resposta = frase_confirmacao(action, target)

                elif action in ACOES_IMEDIATAS:
                    publicar_dds(writer, texto, action, target)
                    resposta = frase_imediata(action)

                else:
                    # DESCONHECIDA — conversa livre com o LLM
                    historico.append({"role": "user", "content": texto})
                    resposta = conversar(historico)
                    historico.append({"role": "assistant", "content": resposta})

                # 5. Falar
                print(f"[Johnny]: {resposta}")
                falar(resposta)

            finally:
                if os.path.exists(ficheiro):
                    os.remove(ficheiro)

    except KeyboardInterrupt:
        print("\n\nA desligar o Johnny... Até logo!")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()