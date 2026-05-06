#!/usr/bin/env python3

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


# ================= CONFIG =================
WHISPER_MODEL = "medium"
OLLAMA_MODEL = "qwen2.5:1.5b"

TOPIC_NAME = "HRICommands"
AUDIO_TEMP = "temp_hri.wav"
AUDIO_RESP = "resposta_hri.mp3"

AUDIO_TOPIC = b"g1_audio"
G1_IP = "192.168.123.164"
PORT = 5556

ZMQ_TIMEOUT = 5

# 🔥 AJUSTADO PARA ROBÔ (ruído alto)
VAD_THRESHOLD = 3500
VAD_SILENCE_SECS = 1.2
VAD_MIN_SPEECH_SECS = 0.4


ACOES_COM_CONFIRMACAO = {"IR_BUSCAR", "TRAZER", "AGARRAR"}

ACOES_IMEDIATAS = {
    "ANDAR", "PARAR", "RECUAR", "LEVANTAR", "SENTAR",
    "VIRAR_ESQUERDA", "VIRAR_DIREITA", "OLHAR_INTERLOCUTOR",
    "OLHAR_FRENTE", "CUMPRIMENTAR", "APRESENTAR",
    "ESTADO_ATUAL", "REPETIR", "LARGAR",
}


# ================= DDS =================
@dataclass
class HRICommand(IdlStruct):
    source: str
    original_text: str
    action: str
    target: str
    confirmed: bool
    timestamp: str


# ================= NLP =================
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
    (["vai buscar", "ir buscar"], "IR_BUSCAR", None),
    (["traz"], "TRAZER", None),
    (["agarra"], "AGARRAR", None),
    (["anda"], "ANDAR", "NENHUM"),
    (["para"], "PARAR", "NENHUM"),
    (["recua"], "RECUAR", "NENHUM"),
    (["levanta"], "LEVANTAR", "NENHUM"),
    (["senta"], "SENTAR", "NENHUM"),
    (["ola"], "CUMPRIMENTAR", "NENHUM"),

    # 🔥 IMPORTANTE (tinham perdido isto!)
    (["sim", "ok", "faz", "quero"], "CONFIRMAR", "NENHUM"),
    (["nao", "cancela"], "CANCELAR", "NENHUM"),
]

REGRAS_TARGET = [
    (["bola", "tenis"], "BOLA_DE_TENIS"),
    (["cubo", "rubik"], "CUBO_DE_RUBIK"),
]


def classificar(texto: str):
    t = normalizar(texto)
    action = "DESCONHECIDA"
    target = "NENHUM"

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

    if action in ACOES_COM_CONFIRMACAO and target == "NENHUM":
        target = "DESCONHECIDO"

    return action, target


# ================= AUDIO =================
def calcular_rms(pcm):
    samples = struct.unpack(f"{len(pcm)//2}h", pcm)
    return math.sqrt(sum(s*s for s in samples)/len(samples))


def parse_audio_parts(parts):
    if len(parts) >= 3:
        return parts[-2], parts[-1]
    return None, None


def gravar():
    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.connect(f"tcp://{G1_IP}:{PORT}")
    sock.setsockopt(zmq.SUBSCRIBE, AUDIO_TOPIC)
    sock.setsockopt(zmq.RCVTIMEO, ZMQ_TIMEOUT * 1000)

    audio_buffer = bytearray()

    speech_started = False
    speech_frames = 0
    silence_frames = 0

    print("\n[MIC] A ouvir...")

    try:
        while True:
            try:
                parts = sock.recv_multipart()
            except zmq.Again:
                continue

            if parts[0] != AUDIO_TOPIC:
                continue

            header, pcm = parse_audio_parts(parts)

            try:
                pcm = lz4.frame.decompress(pcm)
            except:
                continue

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
                    speech_started = True
                    print("[MIC] Voz detetada")
                audio_buffer.extend(pcm)

            elif speech_started:
                audio_buffer.extend(pcm)

                if is_silence:
                    print("[MIC] Silencio -> processar")
                    break

        if not audio_buffer:
            return None

        with wave.open(AUDIO_TEMP, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(48000)
            wf.writeframes(audio_buffer)

        return AUDIO_TEMP

    finally:
        sock.close()
        ctx.term()


# ================= TTS =================
def falar(texto):
    async def _run():
        await edge_tts.Communicate(texto, "pt-PT-DuarteNeural").save(AUDIO_RESP)

    asyncio.run(_run())

    pygame.mixer.init()
    pygame.mixer.music.load(AUDIO_RESP)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    pygame.mixer.quit()


# ================= FRASES =================
def frase_confirmacao(action, target):
    return f"Queres que eu execute {action} com {target}?"

def frase_execucao(action, target):
    return f"Ok! Vou executar {action}."

def frase_imediata(action):
    return f"Ok! {action}."


# ================= MAIN =================
def main():
    whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")

    participant = DomainParticipant()
    topic = Topic(participant, TOPIC_NAME, HRICommand)
    writer = DataWriter(participant, topic)

    pending = None

    print("Sistema pronto\n")

    while True:
        ficheiro = gravar()
        if not ficheiro:
            continue

        segs, _ = whisper.transcribe(ficheiro, language="pt")
        texto = "".join(s.text for s in segs).strip()

        print("[USER]", texto)

        action, target = classificar(texto)
        print("[CLASS]", action, target)

        if action == "CONFIRMAR" and pending:
            print("[EXECUTAR]")
            falar(frase_execucao(pending[0], pending[1]))
            pending = None

        elif action in ACOES_COM_CONFIRMACAO:
            pending = (action, target)
            falar(frase_confirmacao(action, target))

        elif action in ACOES_IMEDIATAS:
            falar(frase_imediata(action))

        else:
            falar("Não percebi, podes repetir?")


if __name__ == "__main__":
    main()
