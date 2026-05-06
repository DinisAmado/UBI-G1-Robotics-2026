#!/usr/bin/env python3

import os
import re
import math
import struct
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
from cyclonedds.idl import IdlStruct
from cyclonedds.domain import DomainParticipant
from cyclonedds.topic import Topic
from cyclonedds.pub import DataWriter

# ==============================================================================
# CONFIG
# ==============================================================================
WHISPER_MODEL = "medium"

AUDIO_TEMP = "temp_hri.wav"
AUDIO_RESP = "resposta_hri.mp3"

AUDIO_TOPIC = b"g1_audio"
G1_IP = "192.168.123.164"
PORT = 5556
ZMQ_TIMEOUT = 5

VAD_THRESHOLD = 3500
VAD_SILENCE_SECS = 1.2
VAD_MIN_SPEECH_SECS = 0.4

PENDING_TIMEOUT_SEC = 6

ACOES_COM_CONFIRMACAO = {"IR_BUSCAR", "TRAZER", "AGARRAR"}
ACOES_IMEDIATAS = {"ANDAR", "PARAR", "RECUAR"}

# ==============================================================================
# DDS
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
# NLP
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

def classificar(texto: str) -> dict:
    t = normalizar(texto)
    action = "DESCONHECIDA"
    target = "NENHUM"

    REGRAS_TARGET = [
        (["bola", "tenis"], "BOLA_DE_TENIS"),
        (["cubo", "rubik"], "CUBO_DE_RUBIK"),
        (["pasta", "dentes"], "PASTA_DE_DENTES"),
    ]

    REGRAS = [
        (["vai buscar", "ir buscar", "apanha"], "IR_BUSCAR", None),
        (["traz"], "TRAZER", None),
        (["agarra"], "AGARRAR", None),

        (["anda"], "ANDAR", "NENHUM"),
        (["para"], "PARAR", "NENHUM"),
        (["recua"], "RECUAR", "NENHUM"),

        (["sim", "ok", "claro"], "CONFIRMAR", "NENHUM"),
        (["nao", "cancela"], "CANCELAR", "NENHUM"),
    ]

    for palavras, tgt in REGRAS_TARGET:
        if any(contem(t, p) for p in palavras):
            target = tgt
            break

    for palavras, act, override in REGRAS:
        if any(contem(t, p) for p in palavras):
            action = act
            if override:
                target = override
            break

    return {"action": action, "target": target}

# ==============================================================================
# AUDIO (VAD)
# ==============================================================================
def calcular_rms(pcm_bytes: bytes) -> float:
    if len(pcm_bytes) < 2:
        return 0.0
    samples = struct.unpack(f"{len(pcm_bytes)//2}h", pcm_bytes)
    return math.sqrt(sum(s*s for s in samples)/len(samples))

def gravar() -> Optional[str]:
    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.connect(f"tcp://{G1_IP}:{PORT}")
    sock.setsockopt(zmq.SUBSCRIBE, AUDIO_TOPIC)
    sock.setsockopt(zmq.RCVTIMEO, ZMQ_TIMEOUT * 1000)

    audio_buffer = bytearray()
    last_sr = 48000
    last_ch = 1

    speech_started = False
    speech_duration = 0.0
    silence_duration = 0.0

    speech_frames = 0
    silence_frames = 0

    print("\n[MIC] A ouvir...")

    while True:
        try:
            parts = sock.recv_multipart()
        except zmq.Again:
            if speech_started and speech_duration >= VAD_MIN_SPEECH_SECS:
                break
            continue

        if not parts or parts[0] != AUDIO_TOPIC:
            continue

        # FIX ZMQ (variável)
        header = parts[1]
        pcm_compressed = b"".join(parts[2:])

        if header and len(header) >= 5:
            last_sr = int.from_bytes(header[:4], "little")
            last_ch = header[4]

        try:
            pcm = lz4.frame.decompress(pcm_compressed)
        except:
            continue

        chunk_secs = (len(pcm)//2) / last_sr
        rms = calcular_rms(pcm)

        if rms > VAD_THRESHOLD:
            speech_frames += 1
            silence_frames = 0
        else:
            silence_frames += 1
            speech_frames = 0

        is_speech = speech_frames >= 3
        is_silence = silence_frames >= 8

        if is_speech:
            if not speech_started:
                print("[MIC] Voz detetada")
                speech_started = True

            silence_duration = 0
            speech_duration += chunk_secs
            audio_buffer.extend(pcm)

        elif speech_started:
            silence_duration += chunk_secs
            audio_buffer.extend(pcm)

            if is_silence and silence_duration >= VAD_SILENCE_SECS:
                break

    if not audio_buffer:
        return None

    with wave.open(AUDIO_TEMP, "wb") as wf:
        wf.setnchannels(last_ch)
        wf.setsampwidth(2)
        wf.setframerate(last_sr)
        wf.writeframes(audio_buffer)

    return AUDIO_TEMP

# ==============================================================================
# TTS
# ==============================================================================
def falar(texto: str):
    async def _run():
        await edge_tts.Communicate(texto, "pt-PT-DuarteNeural").save(AUDIO_RESP)

    asyncio.run(_run())

    pygame.mixer.init()
    pygame.mixer.music.load(AUDIO_RESP)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    pygame.mixer.quit()

# ==============================================================================
# FRASES
# ==============================================================================
NOME_TARGET = {
    "BOLA_DE_TENIS": "a bola de ténis",
    "CUBO_DE_RUBIK": "o cubo mágico",
    "PASTA_DE_DENTES": "a pasta de dentes",
    "NENHUM": "isso",
}

def frase_confirmacao(action, target):
    nome = NOME_TARGET.get(target, "o objeto")
    return f"Queres que eu vá buscar {nome}?"

def frase_execucao(action, target):
    nome = NOME_TARGET.get(target, "o objeto")
    return f"Ok! Vou tratar de {nome}."

def frase_imediata(action):
    return {
        "ANDAR": "Ok, a andar!",
        "PARAR": "Ok, paro aqui.",
        "RECUAR": "Ok, a recuar.",
    }.get(action, "Ok!")

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print("Sistema pronto")

    whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")

    participant = DomainParticipant()
    topic = Topic(participant, "HRICommands", HRICommand)
    writer = DataWriter(participant, topic)

    pending = None
    pending_time = None
    last_text = ""

    while True:
        ficheiro = gravar()
        if not ficheiro:
            continue

        segs, _ = whisper.transcribe(ficheiro, language="pt")
        texto = "".join(s.text for s in segs).strip()

        if texto == last_text:
            continue
        last_text = texto

        if not texto:
            continue

        print("[USER]", texto)

        result = classificar(texto)
        action = result["action"]
        target = result["target"]

        resposta = None

        # FIX timeout seguro
        if pending and pending_time is not None and (time.time() - pending_time > PENDING_TIMEOUT_SEC):
            print("[PENDING] expirou")
            pending = None

        if action == "CONFIRMAR" and pending:
            cmd = HRICommand(
                source="HRI",
                original_text=texto,
                action=pending["action"],
                target=pending["target"],
                confirmed=True,
                timestamp=datetime.now().isoformat()
            )
            writer.write(cmd)

            resposta = frase_execucao(pending["action"], pending["target"])
            pending = None

        elif action == "CANCELAR":
            pending = None
            resposta = "Ok, cancelado."

        elif action in ACOES_COM_CONFIRMACAO:
            pending = {"action": action, "target": target}
            pending_time = time.time()
            resposta = frase_confirmacao(action, target)

        elif action in ACOES_IMEDIATAS:
            cmd = HRICommand(
                source="HRI",
                original_text=texto,
                action=action,
                target=target,
                confirmed=True,
                timestamp=datetime.now().isoformat()
            )
            writer.write(cmd)
            resposta = frase_imediata(action)

        else:
            resposta = "Não percebi, podes repetir?"

        print("[ROBOT]", resposta)
        falar(resposta)

if __name__ == "__main__":
    main()
