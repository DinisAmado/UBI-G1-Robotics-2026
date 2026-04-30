#!/usr/bin/env python3
"""
hri_unitree.py -- Sistema HRI completo para o Robo Unitree G1
Classificacao: palavras-chave (instantanea, sem GPU)
Conversa:      Ollama local

Prerequisitos:
  pip install faster-whisper pygame edge-tts cyclonedds ollama pyzmq lz4
  ollama pull qwen2.5:1.5b
"""

import os
import re
import sys
import asyncio
import unicodedata
import zmq
import lz4.frame
import wave
import time
from datetime import datetime
from dataclasses import dataclass
from typing import Optional

import pygame
from faster_whisper import WhisperModel
import edge_tts
import ollama as ollama_client
from cyclonedds.idl import IdlStruct
from cyclonedds.domain import DomainParticipant
from cyclonedds.topic import Topic
from cyclonedds.pub import DataWriter


# ==============================================================================
# CONFIGURACAO
# ==============================================================================
WHISPER_MODEL  = "medium"
OLLAMA_MODEL   = "qwen2.5:1.5b"
TOPIC_NAME     = "HRICommands"
AUDIO_TEMP     = "temp_hri.wav"
AUDIO_RESP     = "resposta_hri.mp3"

AUDIO_TOPIC    = b"g1_audio"
G1_IP          = "192.168.123.164"
PORT           = 5556
ZMQ_TIMEOUT    = 5
RECORD_SECONDS = 4

ACOES_COM_CONFIRMACAO = {"IR_BUSCAR", "TRAZER", "AGARRAR"}

ACOES_IMEDIATAS = {
    "ANDAR", "PARAR", "RECUAR", "LEVANTAR", "SENTAR",
    "VIRAR_ESQUERDA", "VIRAR_DIREITA", "OLHAR_INTERLOCUTOR",
    "OLHAR_FRENTE", "CUMPRIMENTAR", "APRESENTAR", "ESTADO_ATUAL", "REPETIR",
    "LARGAR",
}


# ==============================================================================
# 1. TIPO DDS
# ==============================================================================
@dataclass
class HRICommand(IdlStruct):
    source: str
    original_text: str
    action: str
    target: str
    confirmed: bool
    timestamp: str


# ==============================================================================
# 2. CLASSIFICADOR DE PALAVRAS-CHAVE
# ==============================================================================
def normalizar(texto: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", texto.lower())
        if unicodedata.category(c) != "Mn"
    )

def contem(texto: str, frase: str) -> bool:
    if " " in frase:
        return frase in texto
    return bool(re.search(r"(?<![a-z])" + re.escape(frase) + r"(?![a-z])", texto))

REGRAS = [
    (["vai buscar", "ir buscar", "busca"],               "IR_BUSCAR",          None),
    (["traz", "traze", "traga"],                         "TRAZER",             None),
    (["agarra", "pega", "apanha"],                       "AGARRAR",            None),
    (["larga", "larga isso", "larga ai"],                "LARGAR",             None),
    (["anda", "avanca", "vai para a frente", "caminha"], "ANDAR",              "NENHUM"),
    (["para", "stop", "fica quieto"],                    "PARAR",              "NENHUM"),
    (["recua", "vai para tras"],                         "RECUAR",             "NENHUM"),
    (["vira a esquerda", "esquerda"],                    "VIRAR_ESQUERDA",     "NENHUM"),
    (["vira a direita", "direita"],                      "VIRAR_DIREITA",      "NENHUM"),
    (["levanta", "levanta-te", "levanta te"],            "LEVANTAR",           "NENHUM"),
    (["senta", "senta-te", "senta te"],                  "SENTAR",             "NENHUM"),
    (["olha para mim", "olha para a pessoa"],            "OLHAR_INTERLOCUTOR", "NENHUM"),
    (["olha em frente", "olha para a frente"],           "OLHAR_FRENTE",       "NENHUM"),
    (["cumprimenta", "cumprimento", "ola"],               "CUMPRIMENTAR",       "NENHUM"),
    (["quem es", "apresenta-te", "como te chamas"],      "APRESENTAR",         "NENHUM"),
    (["como estas", "estado atual"],                     "ESTADO_ATUAL",       "NENHUM"),
    (["repete", "diz outra vez"],                        "REPETIR",            "NENHUM"),
    (["sim", "claro", "faz isso", "ok", "pode ser",
      "isso mesmo", "exato", "vai", "faz",
      "por favor", "faz favor", "se faz favor",
      "quero", "quero sim", "exatamente"],               "CONFIRMAR",          "NENHUM"),
    (["nao", "cancela", "esquece", "afinal nao"],        "CANCELAR",           "NENHUM"),
]

REGRAS_TARGET = [
    (["bola de tenis", "bola de tenis", "bola", "tenis", "boletenas"], "BOLA_DE_TENIS"),
    (["cubo de rubik", "cubo magico", "cubo", "rubik"],                "CUBO_DE_RUBIK"),
    (["pasta de dentes", "pasta", "dentes"],                           "PASTA_DE_DENTES"),
]

def classificar(texto: str) -> dict:
    t = normalizar(texto)
    action = "DESCONHECIDA"
    target = "NENHUM"

    for palavras, tgt in REGRAS_TARGET:
        if any(contem(t, p) for p in palavras):
            target = tgt
            break

    for palavras, act, tgt_override in REGRAS:
        if any(contem(t, p) for p in palavras):
            action = act
            if tgt_override is not None:
                target = tgt_override
            break

    if action in ACOES_COM_CONFIRMACAO and target == "NENHUM":
        target = "DESCONHECIDO"

    return {"action": action, "target": target}


# ==============================================================================
# 3. CONVERSA COM OLLAMA
# ==============================================================================
SYSTEM_PROMPT = (
    "Tu es o Johnny, um robo Unitree em Portugal. "
    "Fala sempre em Portugues de Portugal (pt-PT). Se muito breve e simpatico. "
    "Nunca uses mais de 2 frases seguidas. "
    "Se nao perceberes o comando, diz que nao percebeste e pede para repetir."
)

def conversar(historico: list) -> str:
    print("A pensar...", end="", flush=True)
    try:
        resposta = ollama_client.chat(model=OLLAMA_MODEL, messages=historico)
        texto = resposta["message"]["content"].strip()
        print("\r                    \r", end="")
        return texto
    except Exception as e:
        print(f"\n[ERRO Ollama] {e}")
        return "Desculpa, tive um problema."


# ==============================================================================
# 4. AUDIO — ZMQ streaming do microfone do G1
# ==============================================================================
def gravar() -> Optional[str]:
    """
    Recebe audio do G1 via ZMQ durante RECORD_SECONDS e guarda em WAV.
    """
    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.connect(f"tcp://{G1_IP}:{PORT}")
    sock.setsockopt(zmq.SUBSCRIBE, AUDIO_TOPIC)
    sock.setsockopt(zmq.RCVTIMEO, ZMQ_TIMEOUT * 1000)

    print(f"\n[MIC] A escuta (microfone G1 -- {RECORD_SECONDS}s)...")

    audio_buffer = bytearray()
    last_sr = 48000
    last_ch = 1
    start_time = time.time()

    try:
        while time.time() - start_time < RECORD_SECONDS:
            try:
                parts = sock.recv_multipart()
            except zmq.Again:
                print("[ZMQ] Timeout -- sem audio do robo.")
                break

            if not parts or parts[0] != AUDIO_TOPIC:
                continue

            if len(parts) == 3:
                _, header, pcm_compressed = parts
            else:
                _, header, *rest = parts
                pcm_compressed = b"".join(rest)

            if len(header) >= 5:
                last_sr = int.from_bytes(header[:4], "little")
                last_ch = header[4]

            try:
                pcm = lz4.frame.decompress(pcm_compressed)
            except Exception:
                continue

            audio_buffer.extend(pcm)

        if not audio_buffer:
            print("[ZMQ] Sem audio recebido.")
            return None

        with wave.open(AUDIO_TEMP, "wb") as wf:
            wf.setnchannels(last_ch)
            wf.setsampwidth(2)
            wf.setframerate(last_sr)
            wf.writeframes(audio_buffer)

        print(f"[MIC] Gravacao concluida ({len(audio_buffer)//2} samples @ {last_sr}Hz)")
        return AUDIO_TEMP

    finally:
        sock.close()
        ctx.term()


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


# ==============================================================================
# 5. DDS
# ==============================================================================
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
    print(f"[DDS] PUBLICADO -- action={action}  target={target}")


NOME_TARGET = {
    "BOLA_DE_TENIS":   "a bola de tenis",
    "CUBO_DE_RUBIK":   "o cubo magico",
    "PASTA_DE_DENTES": "a pasta de dentes",
    "DESCONHECIDO":    "o objeto",
    "NENHUM":          "isso",
}

def frase_confirmacao(action: str, target: str) -> str:
    nome = NOME_TARGET.get(target, "o objeto")
    if action == "TRAZER":
        return f"Queres que eu traga {nome}?"
    elif action == "IR_BUSCAR":
        return f"Queres que eu va buscar {nome}?"
    elif action == "AGARRAR":
        return f"Queres que eu agarre {nome}?"
    return f"Confirmas a acao com {nome}?"

def frase_execucao(action: str, target: str) -> str:
    nome = NOME_TARGET.get(target, "o objeto")
    if action == "TRAZER":
        return f"Combinado! Vou ja trazer {nome}."
    elif action == "IR_BUSCAR":
        return f"Combinado! Vou ja buscar {nome}."
    elif action == "AGARRAR":
        return f"Combinado! Vou agarrar {nome}."
    return "Combinado, vou ja!"

def frase_imediata(action: str) -> str:
    return {
        "ANDAR":              "Ok, a andar!",
        "PARAR":              "Ok, paro aqui.",
        "RECUAR":             "Ok, a recuar.",
        "LEVANTAR":           "Ok, a levantar!",
        "SENTAR":             "Ok, a sentar.",
        "VIRAR_ESQUERDA":     "Ok, a virar a esquerda.",
        "VIRAR_DIREITA":      "Ok, a virar a direita.",
        "OLHAR_INTERLOCUTOR": "Ok, a olhar para ti.",
        "OLHAR_FRENTE":       "Ok, a olhar em frente.",
        "CUMPRIMENTAR":       "Ola! Muito prazer!",
        "APRESENTAR":         "Ola! Sou o Johnny, um robo Unitree. E um prazer conhecer-te!",
        "ESTADO_ATUAL":       "Estou operacional e pronto para ajudar!",
        "REPETIR":            "Claro, repito!",
        "LARGAR":             "Ok, a largar!",
    }.get(action, "Ok!")


# ==============================================================================
# 6. LOOP PRINCIPAL
# ==============================================================================
def main():
    print("A carregar Whisper...")
    whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    print("Whisper pronto!")

    print(f"A verificar Ollama ({OLLAMA_MODEL})...")
    try:
        ollama_client.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": "ok"}])
        print("Ollama pronto!\n")
    except Exception:
        print(f"\n[ERRO] Nao consegui ligar ao Ollama!")
        print(f"  1. Instala em: https://ollama.com")
        print(f"  2. Corre num terminal: ollama serve")
        print(f"  3. Descarrega o modelo: ollama pull {OLLAMA_MODEL}")
        sys.exit(1)

    participant = DomainParticipant()
    topic = Topic(participant, TOPIC_NAME, HRICommand)
    writer = DataWriter(participant, topic)

    historico = [{"role": "system", "content": SYSTEM_PROMPT}]
    pending = None  # type: Optional[dict]

    print("=" * 52)
    print("   SISTEMA HRI -- UNITREE G1  (Ctrl+C para sair)")
    print("=" * 52 + "\n")

    try:
        while True:
            ficheiro = gravar()
            if not ficheiro:
                continue

            try:
                segs, _ = whisper.transcribe(ficheiro, language="pt")
                texto = "".join(s.text for s in segs).strip()

                if not texto:
                    print("[Whisper] Nao percebi nada.")
                    continue

                print(f"[Utilizador]: {texto}")

                result = classificar(texto)
                action, target = result["action"], result["target"]
                print(f"[Classificacao]: {action} / {target}")

                resposta = None

                if action == "CONFIRMAR" and pending:
                    publicar_dds(writer, pending["texto"], pending["action"], pending["target"])
                    resposta = frase_execucao(pending["action"], pending["target"])
                    pending = None

                elif action == "CONFIRMAR" and not pending:
                    resposta = "Nao ha nenhuma acao pendente para confirmar."

                elif action == "CANCELAR" and pending:
                    print("[X] Acao cancelada.")
                    resposta = "Ok, fico aqui entao."
                    pending = None

                elif action == "CANCELAR":
                    resposta = "Ok, sem problema."

                elif action in ACOES_COM_CONFIRMACAO:
                    pending = {"action": action, "target": target, "texto": texto}
                    print(f"Pendente: {action}/{target} -- a espera de confirmacao")
                    resposta = frase_confirmacao(action, target)

                elif action in ACOES_IMEDIATAS:
                    publicar_dds(writer, texto, action, target)
                    resposta = frase_imediata(action)

                else:
                    historico.append({"role": "user", "content": texto})
                    resposta = conversar(historico)
                    historico.append({"role": "assistant", "content": resposta})

                print(f"[Johnny]: {resposta}")
                falar(resposta)

            finally:
                if os.path.exists(ficheiro):
                    os.remove(ficheiro)

    except KeyboardInterrupt:
        print("\n\nA desligar o Johnny... Ate logo!")


if __name__ == "__main__":
    main()
