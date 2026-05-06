#!/usr/bin/env python3
"""
hri_unitree.py -- Sistema HRI completo para o Robo Unitree G1
"""

import os
import re
import sys
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
import ollama as ollama_client
from cyclonedds.idl import IdlStruct
from cyclonedds.domain import DomainParticipant
from cyclonedds.topic import Topic
from cyclonedds.pub import DataWriter


# ==============================================================================
# CONFIG
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

VAD_THRESHOLD       = 100  
VAD_SILENCE_SECS    = 1.2
VAD_MIN_SPEECH_SECS = 0.4


ACOES_COM_CONFIRMACAO = {"IR_BUSCAR", "TRAZER", "AGARRAR"}

ACOES_IMEDIATAS = {
    "ANDAR", "PARAR", "RECUAR", "LEVANTAR", "SENTAR",
    "VIRAR_ESQUERDA", "VIRAR_DIREITA", "OLHAR_INTERLOCUTOR",
    "OLHAR_FRENTE", "CUMPRIMENTAR", "APRESENTAR",
    "ESTADO_ATUAL", "REPETIR", "LARGAR",
}


# ==============================================================================
# DDS TYPE
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
# NLP CLASSIFIER
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
        (["vai buscar", "ir buscar"], "IR_BUSCAR", None),
        (["traz"], "TRAZER", None),
        (["agarra"], "AGARRAR", None),
        (["anda"], "ANDAR", "NENHUM"),
        (["para"], "PARAR", "NENHUM"),
        (["recua"], "RECUAR", "NENHUM"),
        (["levanta"], "LEVANTAR", "NENHUM"),
        (["senta"], "SENTAR", "NENHUM"),
        (["olha para mim"], "OLHAR_INTERLOCUTOR", "NENHUM"),
        (["em frente"], "OLHAR_FRENTE", "NENHUM"),
        (["ola"], "CUMPRIMENTAR", "NENHUM"),
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
# AUDIO HELPERS
# ==============================================================================
def calcular_rms(pcm_bytes: bytes) -> float:
    if len(pcm_bytes) < 2:
        return 0.0
    samples = struct.unpack(f"{len(pcm_bytes)//2}h", pcm_bytes)
    return math.sqrt(sum(s*s for s in samples)/len(samples))


def parse_timestamp(part):
    if len(part) != 8:
        return None
    return struct.unpack("d", part)[0]


def parse_audio_parts(parts):
    if len(parts) == 4:
        _, _, header, pcm = parts
        return None, header, pcm

    if len(parts) > 4:
        _, _, header, *rest = parts
        return None, header, b"".join(rest)

    if len(parts) == 3:
        _, header, pcm = parts
        return None, header, pcm

    if len(parts) > 3:
        _, header, *rest = parts
        return None, header, b"".join(rest)

    return None, None, None


# ==============================================================================
# AUDIO CAPTURE (FIX PRINCIPAL)
# ==============================================================================
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

    print("\n[MIC] A ouvir...")

    try:
        while True:
            try:
                parts = sock.recv_multipart()
            except zmq.Again:
                if speech_started and speech_duration >= VAD_MIN_SPEECH_SECS:
                    break
                continue

            if not parts or parts[0] != AUDIO_TOPIC:
                continue

            _, header, pcm = parse_audio_parts(parts)

            if header and len(header) >= 5:
                last_sr = int.from_bytes(header[:4], "little")
                last_ch = header[4]

            try:
                pcm = lz4.frame.decompress(pcm)
            except:
                continue

            chunk_secs = (len(pcm)//2) / last_sr
            rms = calcular_rms(pcm)

            is_speech = rms > VAD_THRESHOLD

            if is_speech:
                speech_started = True
                silence_duration = 0
                speech_duration += chunk_secs
                audio_buffer.extend(pcm)

            else:
                if speech_started:
                    silence_duration += chunk_secs
                    audio_buffer.extend(pcm)

                    if silence_duration > VAD_SILENCE_SECS:
                        if speech_duration >= VAD_MIN_SPEECH_SECS:
                            break
                        else:
                            audio_buffer.clear()
                            speech_started = False
                            speech_duration = 0
                            silence_duration = 0

        if not audio_buffer:
            return None

        with wave.open(AUDIO_TEMP, "wb") as wf:
            wf.setnchannels(last_ch)
            wf.setsampwidth(2)
            wf.setframerate(last_sr)
            wf.writeframes(audio_buffer)

        return AUDIO_TEMP

    finally:
        sock.close()
        ctx.term()


# ==============================================================================
# MAIN LOOP (igual ao teu)
# ==============================================================================
def main():
    print("Sistema pronto (versao corrigida)")
    whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")

    participant = DomainParticipant()
    topic = Topic(participant, TOPIC_NAME, HRICommand)
    writer = DataWriter(participant, topic)

    historico = []

    while True:
        ficheiro = gravar()
        if not ficheiro:
            continue

        segs, _ = whisper.transcribe(ficheiro, language="pt")
        texto = "".join(s.text for s in segs)

        print("[USER]", texto)

        result = classificar(texto)
        print("[CLASS]", result)


if __name__ == "__main__":
    main()
