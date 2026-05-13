#!/usr/bin/env python3

import os, re, sys, math, struct, asyncio, unicodedata
import zmq, lz4.frame, wave, time
import subprocess
from datetime import datetime
from dataclasses import dataclass
from typing import Optional
from collections import deque

import pygame
from faster_whisper import WhisperModel
import edge_tts
import ollama as ollama_client
import webrtcvad

from cyclonedds.idl import IdlStruct
from cyclonedds.domain import DomainParticipant
from cyclonedds.topic import Topic
from cyclonedds.pub import DataWriter

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient


# ==============================================================================
# CONFIG
# ==============================================================================

WHISPER_MODEL  = "large-v3-turbo"
OLLAMA_MODEL   = "qwen2.5:1.5b"

TOPIC_NAME     = "HRICommands"

AUDIO_TEMP     = "temp_hri.wav"
AUDIO_RESP     = "resposta_hri.mp3"
AUDIO_RESP_WAV = "resposta_hri_16k_mono.wav"

NET_INTERFACE  = "enp117s0"

AUDIO_TOPIC    = b"g1_audio"
G1_IP          = "192.168.123.164"
PORT           = 5556
ZMQ_TIMEOUT    = 5

VAD_RMS_MIN = 1200
WEBRTC_VAD_MODE = 3
WEBRTC_MIN_SPEECH_RATIO = 0.6

VAD_SILENCE_SECS = 1.2
VAD_MIN_SPEECH_SECS = 0.4
PRE_BUFFER_SECS = 0.5
AUDIO_GAIN = 1.6
MAX_RECORDING_SECS = 5.0


# ==============================================================================
# LED CONTROLLER
# ==============================================================================

LED_A_OUVIR = (0, 0, 255)
LED_A_FALAR = (0, 255, 0)
LED_ERRO = (255, 0, 0)
LED_OFF = (0, 0, 0)


class LedController:
    def __init__(self, interface=NET_INTERFACE):
        self.ok = False
        self.client = None

        try:
            ChannelFactoryInitialize(0, interface)

            self.client = AudioClient()
            self.client.SetTimeout(10.0)
            self.client.Init()

            self.ok = True
            print("[LEDS] OK")

        except Exception as e:
            print(f"[LEDS] Erro: {e}")

    def set(self, r, g, b):
        if not self.ok:
            return
        try:
            self.client.LedControl(int(r), int(g), int(b))
        except Exception as e:
            print(f"[LEDS] falha: {e}")

    def ouvir(self):
        self.set(*LED_A_OUVIR)

    def falar(self):
        self.set(*LED_A_FALAR)

    def erro(self):
        self.set(*LED_ERRO)

    def off(self):
        self.set(*LED_OFF)


# ==============================================================================
# DDS STRUCT
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
# NLP RULES
# ==============================================================================

def normalizar(texto):
    return "".join(
        c for c in unicodedata.normalize("NFD", texto.lower())
        if unicodedata.category(c) != "Mn"
    )


def contem(texto, frase):
    if " " in frase:
        return frase in texto
    return bool(re.search(r"(?<![a-z])" + re.escape(frase) + r"(?![a-z])", texto))


REGRAS = [
    (["traz", "traga", "traz-me"], "TRAZER", None),
    (["vai buscar", "busca"], "IR_BUSCAR", None),
    (["agarrar", "pega"], "AGARRAR", None),
    (["para", "stop"], "PARAR", "NENHUM"),
    (["anda", "avança"], "ANDAR", "NENHUM"),
    (["sim", "ok", "confirmo"], "CONFIRMAR", "NENHUM"),
    (["nao", "cancela"], "CANCELAR", "NENHUM"),
]

REGRAS_TARGET = [
    (["bola", "tenis"], "BOLA_DE_TENIS"),
    (["cubo", "rubik"], "CUBO_DE_RUBIK"),
    (["pasta", "dentes"], "PASTA_DE_DENTES"),
]


def classificar(texto):
    t = normalizar(texto)

    action, target = "DESCONHECIDA", "NENHUM"

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
# SYSTEM PROMPT
# ==============================================================================

SYSTEM_PROMPT = (
    "Tu és o Johnny, um robô em Portugal. "
    "Fala português  de portugal europeu. Máximo 2 frases."
)


# ==============================================================================
# AUDIO / TTS
# ==============================================================================

async def tts(texto):
    await edge_tts.Communicate(texto, "pt-PT-DuarteNeural").save(AUDIO_RESP)


def falar(texto, leds: LedController):
    asyncio.run(tts(texto))

    leds.falar()

    pygame.mixer.init()
    pygame.mixer.music.load(AUDIO_RESP)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

    pygame.mixer.quit()

    leds.off()


# ==============================================================================
# OLLAMA
# ==============================================================================

def conversar(hist):
    try:
        r = ollama_client.chat(model=OLLAMA_MODEL, messages=hist)
        return r["message"]["content"].strip()
    except:
        return "Tive um problema."


# ==============================================================================
# MAIN
# ==============================================================================

def main():

    leds = LedController()

    print("A carregar Whisper...")
    whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    print("Whisper pronto")

    try:
        ollama_client.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": "ok"}])
        print("Ollama pronto")
    except:
        leds.erro()
        print("ERRO OLLAMA")
        sys.exit(1)

    participant = DomainParticipant()
    topic = Topic(participant, TOPIC_NAME, HRICommand)
    writer = DataWriter(participant, topic)

    hist = [{"role": "system", "content": SYSTEM_PROMPT}]
    pending = None

    while True:
        leds.ouvir()

        # =========================
        # AQUI ASSUMIMOS O TEU GRAVAR()
        # =========================
        ficheiro = "temp.wav"  # substitui pelo teu gravar()

        try:
            segs, _ = whisper.transcribe(ficheiro, language="pt")
            texto = "".join(s.text for s in segs).strip()

            print("[Utilizador]:", texto)

            r = classificar(texto)
            action, target = r["action"], r["target"]

            if action == "CONFIRMAR" and pending:
                resposta = "Executado."
                pending = None

            elif action == "CANCELAR":
                resposta = "Cancelado."
                pending = None

            elif action in ["TRAZER", "IR_BUSCAR", "AGARRAR"]:
                pending = {"action": action, "target": target}
                resposta = f"Queres confirmar {action}?"

            else:
                hist.append({"role": "user", "content": texto})
                resposta = conversar(hist)
                hist.append({"role": "assistant", "content": resposta})

            print("[Johnny]:", resposta)
            falar(resposta, leds)

        except Exception as e:
            leds.erro()
            print("Erro:", e)


if __name__ == "__main__":
    main()
