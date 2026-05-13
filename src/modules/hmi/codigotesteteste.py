#!/usr/bin/env python3

import os
import re
import sys
import math
import struct
import asyncio
import unicodedata
import subprocess
import zmq
import lz4.frame
import wave
import time

from datetime import datetime
from dataclasses import dataclass
from typing import Optional
from collections import deque

import webrtcvad
import edge_tts
import ollama as ollama_client

from faster_whisper import WhisperModel

from cyclonedds.idl import IdlStruct
from cyclonedds.domain import DomainParticipant
from cyclonedds.topic import Topic
from cyclonedds.pub import DataWriter

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient


# =========================================================
# CONFIG
# =========================================================

WHISPER_MODEL = "large-v3-turbo"
OLLAMA_MODEL = "qwen2.5:3b"

TOPIC_NAME = "HRICommands"

AUDIO_TEMP = "temp_hri.wav"
AUDIO_RESP = "resposta_hri.mp3"
AUDIO_MONO = "resposta_mono.wav"

AUDIO_TOPIC = b"g1_audio"

G1_IP = "192.168.123.164"
PORT = 5556

NET_INTERFACE = "enp117s0"

ZMQ_TIMEOUT = 5

VAD_RMS_MIN = 1200
WEBRTC_VAD_MODE = 3
WEBRTC_FRAME_MS = 30
WEBRTC_MIN_SPEECH_RATIO = 0.6

VAD_SILENCE_SECS = 1.2
VAD_MIN_SPEECH_SECS = 0.4

PRE_BUFFER_SECS = 0.5

AUDIO_GAIN = 1.6

MAX_RECORDING_SECS = 7.0


# =========================================================
# AÇÕES
# =========================================================

ACOES_COM_CONFIRMACAO = {
    "IR_BUSCAR",
    "TRAZER",
    "AGARRAR"
}

ACOES_IMEDIATAS = {
    "ANDAR",
    "PARAR",
    "RECUAR",
    "LEVANTAR",
    "SENTAR",
    "VIRAR_ESQUERDA",
    "VIRAR_DIREITA",
    "OLHAR_INTERLOCUTOR",
    "OLHAR_FRENTE",
    "CUMPRIMENTAR",
    "APRESENTAR",
    "ESTADO_ATUAL",
    "REPETIR",
    "LARGAR",
}


# =========================================================
# DDS STRUCT
# =========================================================

@dataclass
class HRICommand(IdlStruct):
    source: str
    original_text: str
    action: str
    target: str
    confirmed: bool
    timestamp: str


# =========================================================
# NLP
# =========================================================

def normalizar(texto):
    return "".join(
        c for c in unicodedata.normalize("NFD", texto.lower())
        if unicodedata.category(c) != "Mn"
    )


def contem(texto, frase):
    if " " in frase:
        return frase in texto

    return bool(
        re.search(
            r"(?<![a-z])" + re.escape(frase) + r"(?![a-z])",
            texto
        )
    )


REGRAS = [

    (["traz", "traze", "traga", "traz-me", "traz me"], "TRAZER", None),

    (["vai buscar", "ir buscar", "busca", "procura"],
     "IR_BUSCAR", None),

    (["agarra", "pega", "apanha"], "AGARRAR", None),

    (["larga"], "LARGAR", None),

    (["anda", "avanca", "vai para a frente"],
     "ANDAR", "NENHUM"),

    (["para", "stop"], "PARAR", "NENHUM"),

    (["recua", "vai para tras"],
     "RECUAR", "NENHUM"),

    (["vira a esquerda", "esquerda"],
     "VIRAR_ESQUERDA", "NENHUM"),

    (["vira a direita", "direita"],
     "VIRAR_DIREITA", "NENHUM"),

    (["levanta"], "LEVANTAR", "NENHUM"),

    (["senta"], "SENTAR", "NENHUM"),

    (["olha para mim"],
     "OLHAR_INTERLOCUTOR", "NENHUM"),

    (["olha em frente", "olha para a frente"],
     "OLHAR_FRENTE", "NENHUM"),

    (["cumprimenta", "ola"],
     "CUMPRIMENTAR", "NENHUM"),

    (["quem es", "apresenta-te", "como te chamas"],
     "APRESENTAR", "NENHUM"),

    (["como estas", "estado atual"],
     "ESTADO_ATUAL", "NENHUM"),

    (["repete", "diz outra vez"],
     "REPETIR", "NENHUM"),

    (["sim", "claro", "faz isso", "ok",
      "pode ser", "exato", "por favor",
      "faz favor", "confirmo"],
     "CONFIRMAR", "NENHUM"),

    (["nao", "cancela", "esquece", "afinal nao"],
     "CANCELAR", "NENHUM"),
]


REGRAS_TARGET = [

    (["bola de tenis", "bola", "tenis"],
     "BOLA_DE_TENIS"),

    (["cubo de rubik", "cubo magico",
      "cubo", "rubik"],
     "CUBO_DE_RUBIK"),

    (["pasta de dentes", "pasta", "dentes"],
     "PASTA_DE_DENTES"),
]


def classificar(texto):

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

            if override is not None:
                target = override

            break

    if action in ACOES_COM_CONFIRMACAO and target == "NENHUM":
        target = "DESCONHECIDO"

    return {
        "action": action,
        "target": target
    }


# =========================================================
# LLM
# =========================================================

SYSTEM_PROMPT = (
    "Tu es o Johnny, um robo Unitree em Portugal. "
    "Fala sempre em Portugues de Portugal. "
    "Se simpatico e muito breve. "
    "Nunca uses mais de 2 frases."
)


def conversar(historico):

    try:

        r = ollama_client.chat(
            model=OLLAMA_MODEL,
            messages=historico
        )

        return r["message"]["content"].strip()

    except Exception as e:

        print(f"[ERRO OLLAMA] {e}")

        return "Desculpa, tive um problema."


# =========================================================
# AUDIO HELPERS
# =========================================================

def calcular_rms(pcm_bytes):

    if len(pcm_bytes) < 2:
        return 0.0

    samples = struct.unpack(
        f"{len(pcm_bytes)//2}h",
        pcm_bytes
    )

    return math.sqrt(
        sum(s*s for s in samples) / len(samples)
    )


def aplicar_ganho_pcm16(pcm_bytes, ganho=AUDIO_GAIN):

    if not pcm_bytes or ganho == 1.0:
        return pcm_bytes

    samples = struct.unpack(
        f"{len(pcm_bytes)//2}h",
        pcm_bytes
    )

    out = []

    for s in samples:

        v = int(s * ganho)

        if v > 32767:
            v = 32767

        elif v < -32768:
            v = -32768

        out.append(v)

    return struct.pack(f"{len(out)}h", *out)


def parse_audio_parts(parts):

    if len(parts) == 4:
        return None, parts[2], parts[3]

    if len(parts) > 4:
        return None, parts[2], b"".join(parts[3:])

    if len(parts) == 3:
        return None, parts[1], parts[2]

    if len(parts) > 3:
        return None, parts[1], b"".join(parts[2:])

    return None, None, None


def pcm_to_mono(pcm_bytes, channels):

    if channels <= 1:
        return pcm_bytes

    samples = struct.unpack(
        f"{len(pcm_bytes)//2}h",
        pcm_bytes
    )

    mono = samples[::channels]

    return struct.pack(f"{len(mono)}h", *mono)


def gerar_frames_webrtc(
    pcm_bytes,
    sample_rate,
    frame_ms=WEBRTC_FRAME_MS
):

    samples_por_frame = int(
        sample_rate * frame_ms / 1000
    )

    bytes_por_frame = samples_por_frame * 2

    for i in range(
        0,
        len(pcm_bytes) - bytes_por_frame + 1,
        bytes_por_frame
    ):

        yield pcm_bytes[i:i+bytes_por_frame]


def webrtc_tem_fala(vad, pcm_bytes, sample_rate, channels):

    if sample_rate not in (8000, 16000, 32000, 48000):
        return False

    pcm_mono = pcm_to_mono(pcm_bytes, channels)

    frames = list(
        gerar_frames_webrtc(
            pcm_mono,
            sample_rate
        )
    )

    if not frames:
        return False

    frames_com_fala = 0

    for frame in frames:

        try:

            if vad.is_speech(frame, sample_rate):
                frames_com_fala += 1

        except Exception:
            return False

    ratio = frames_com_fala / len(frames)

    return ratio >= WEBRTC_MIN_SPEECH_RATIO


# =========================================================
# GRAVAR
# =========================================================

def gravar():

    ctx = zmq.Context()

    sock = ctx.socket(zmq.SUB)

    sock.connect(f"tcp://{G1_IP}:{PORT}")

    sock.setsockopt(zmq.SUBSCRIBE, AUDIO_TOPIC)

    sock.setsockopt(
        zmq.RCVTIMEO,
        ZMQ_TIMEOUT * 1000
    )

    audio_buffer = bytearray()

    pre_buffer = deque()

    pre_buffer_duration = 0.0

    last_sr = 48000
    last_ch = 1

    speech_started = False

    speech_duration = 0.0
    silence_duration = 0.0

    speech_frames = 0
    silence_frames = 0

    recording_duration = 0.0

    vad = webrtcvad.Vad(WEBRTC_VAD_MODE)

    print("\n[MIC] A ouvir...")

    try:

        while True:

            try:

                parts = sock.recv_multipart()

            except zmq.Again:

                if speech_started:
                    break

                continue

            if not parts or parts[0] != AUDIO_TOPIC:
                continue

            _, header, pcm_compressed = parse_audio_parts(parts)

            if header and len(header) >= 5:

                last_sr = int.from_bytes(
                    header[:4],
                    "little"
                )

                last_ch = header[4]

            try:

                pcm = lz4.frame.decompress(
                    pcm_compressed
                )

            except:
                continue

            if not pcm:
                continue

            chunk_secs = (
                (len(pcm)//2)
                / (last_sr * max(last_ch, 1))
            )

            rms = calcular_rms(pcm)

            vad_ok = webrtc_tem_fala(
                vad,
                pcm,
                last_sr,
                last_ch
            )

            if rms > VAD_RMS_MIN and vad_ok:

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

                    for old_pcm, _ in pre_buffer:
                        audio_buffer.extend(old_pcm)

                    pre_buffer.clear()

                silence_duration = 0.0

                speech_duration += chunk_secs
                recording_duration += chunk_secs

                audio_buffer.extend(pcm)

            elif not speech_started:

                pre_buffer.append((pcm, chunk_secs))

                pre_buffer_duration += chunk_secs

                while (
                    pre_buffer_duration > PRE_BUFFER_SECS
                    and pre_buffer
                ):

                    _, dur = pre_buffer.popleft()

                    pre_buffer_duration -= dur

            else:

                silence_duration += chunk_secs
                recording_duration += chunk_secs

                audio_buffer.extend(pcm)

                if (
                    is_silence
                    and silence_duration >= VAD_SILENCE_SECS
                ):

                    if speech_duration >= VAD_MIN_SPEECH_SECS:

                        print("[MIC] Silencio -> processar")

                        break

            if (
                speech_started
                and recording_duration >= MAX_RECORDING_SECS
            ):

                print("[MIC] Tempo maximo -> processar")

                break

        if not audio_buffer:
            return None

        audio_final = aplicar_ganho_pcm16(
            bytes(audio_buffer)
        )

        with wave.open(AUDIO_TEMP, "wb") as wf:

            wf.setnchannels(last_ch)

            wf.setsampwidth(2)

            wf.setframerate(last_sr)

            wf.writeframes(audio_final)

        return AUDIO_TEMP

    finally:

        sock.close()

        ctx.term()


# =========================================================
# TTS UNITREE
# =========================================================

def ler_wav_mono16k(path):

    with wave.open(path, "rb") as wf:

        assert wf.getframerate() == 16000
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2

        return wf.readframes(wf.getnframes())


def falar(texto, audio_client):

    async def _g():

        await edge_tts.Communicate(
            texto,
            "pt-PT-DuarteNeural"
        ).save(AUDIO_RESP)

    asyncio.run(_g())

    # Converter MP3 -> WAV PCM mono 16kHz
    subprocess.run([

        "ffmpeg",
        "-y",

        "-i", AUDIO_RESP,

        "-ar", "16000",

        "-ac", "1",

        "-sample_fmt", "s16",

        AUDIO_MONO

    ],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL)

    try:

        pcm_bytes = ler_wav_mono16k(AUDIO_MONO)

        # 100 ms chunks
        CHUNK = 3200

        chunks = [

            pcm_bytes[i:i+CHUNK]

            for i in range(
                0,
                len(pcm_bytes),
                CHUNK
            )
        ]

        audio_client.SetVolume(100)

        print("[TTS] A reproduzir...")

        for chunk in chunks:

            try:

                audio_client.PlayStream(
                    "hri_app",
                    "stream_001",
                    chunk
                )

            except Exception as e:

                print(f"[ERRO STREAM] {e}")

                break

            # IMPORTANTÍSSIMO
            time.sleep(0.1)

        try:

            audio_client.PlayStop("hri_app")

        except Exception as e:

            print(f"[ERRO STOP] {e}")

        print("[TTS] Concluido")

    except Exception as e:

        print(f"[ERRO AudioClient] {e}")

    finally:

        for f in [AUDIO_RESP, AUDIO_MONO]:

            if os.path.exists(f):

                os.remove(f)



# =========================================================
# DDS
# =========================================================

def publicar_dds(writer, texto, action, target):

    cmd = HRICommand(

        source="HRI",

        original_text=texto,

        action=action,

        target=target,

        confirmed=True,

        timestamp=datetime.now().isoformat(
            timespec="seconds"
        )
    )

    writer.write(cmd)

    print(
        f"[DDS] PUBLICADO -- "
        f"action={action} target={target}"
    )


# =========================================================
# FRASES
# =========================================================

NOME_TARGET = {

    "BOLA_DE_TENIS": "a bola de tenis",

    "CUBO_DE_RUBIK": "o cubo magico",

    "PASTA_DE_DENTES": "a pasta de dentes",

    "DESCONHECIDO": "o objeto",

    "NENHUM": "isso",
}


def frase_confirmacao(action, target):

    nome = NOME_TARGET.get(target, "o objeto")

    if action == "TRAZER":
        return f"Queres que eu traga {nome}?"

    if action == "IR_BUSCAR":
        return f"Queres que eu va buscar {nome}?"

    if action == "AGARRAR":
        return f"Queres que eu agarre {nome}?"

    return "Confirmas?"


def frase_execucao(action, target):

    nome = NOME_TARGET.get(target, "o objeto")

    if action == "TRAZER":
        return f"Vou trazer {nome}."

    if action == "IR_BUSCAR":
        return f"Vou buscar {nome}."

    if action == "AGARRAR":
        return f"Vou agarrar {nome}."

    return "Ok!"


def frase_imediata(action):

    return {

        "ANDAR": "Ok, a andar!",

        "PARAR": "Ok, paro aqui.",

        "RECUAR": "Ok, a recuar.",

        "LEVANTAR": "Ok, a levantar!",

        "SENTAR": "Ok, a sentar.",

        "VIRAR_ESQUERDA":
            "Ok, a virar a esquerda.",

        "VIRAR_DIREITA":
            "Ok, a virar a direita.",

        "OLHAR_INTERLOCUTOR":
            "Ok, a olhar para ti.",

        "OLHAR_FRENTE":
            "Ok, a olhar em frente.",

        "CUMPRIMENTAR":
            "Ola! Muito prazer!",

        "APRESENTAR":
            "Ola! Sou o Johnny.",

        "ESTADO_ATUAL":
            "Estou operacional!",

        "REPETIR":
            "Claro, repito!",

        "LARGAR":
            "Ok, a largar!",

    }.get(action, "Ok!")


# =========================================================
# MAIN
# =========================================================

def main():

    print("A carregar Whisper...")

    whisper = WhisperModel(
        WHISPER_MODEL,
        device="cpu",
        compute_type="int8"
    )

    print("Whisper pronto!")

    try:

        ollama_client.chat(

            model=OLLAMA_MODEL,

            messages=[
                {
                    "role": "user",
                    "content": "ok"
                }
            ]
        )

        print("Ollama pronto!")

    except Exception as e:

        print(f"Erro Ollama: {e}")

        sys.exit(1)

    print("A inicializar AudioClient...")

    ChannelFactoryInitialize(0, NET_INTERFACE)

    audio_client = AudioClient()

    audio_client.SetTimeout(10.0)

    audio_client.Init()

    print("AudioClient pronto!\n")

    participant = DomainParticipant()

    topic = Topic(
        participant,
        TOPIC_NAME,
        HRICommand
    )

    writer = DataWriter(
        participant,
        topic
    )

    historico = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]

    pending = None

    print("=" * 52)
    print("   SISTEMA HRI -- UNITREE G1")
    print("=" * 52)

    try:

        while True:

            ficheiro = gravar()

            if not ficheiro:
                continue

            try:

                segs, _ = whisper.transcribe(

                    ficheiro,

                    language="pt",

                    beam_size=5,

                    best_of=5,

                    temperature=0.0,

                    condition_on_previous_text=False
                )

                texto = "".join(
                    s.text for s in segs
                ).strip()

                if not texto:

                    print("[Whisper] Nada.")

                    continue

                print(f"[Utilizador]: {texto}")

                result = classificar(texto)

                action = result["action"]

                target = result["target"]

                print(
                    f"[Classificacao]: "
                    f"{action} / {target}"
                )

                if action == "CONFIRMAR" and pending:

                    publicar_dds(

                        writer,

                        pending["texto"],

                        pending["action"],

                        pending["target"]
                    )

                    resposta = frase_execucao(
                        pending["action"],
                        pending["target"]
                    )

                    pending = None

                elif action == "CANCELAR" and pending:

                    resposta = "Ok, cancelado."

                    pending = None

                elif action in ACOES_COM_CONFIRMACAO:

                    pending = {

                        "action": action,

                        "target": target,

                        "texto": texto
                    }

                    resposta = frase_confirmacao(
                        action,
                        target
                    )

                elif action in ACOES_IMEDIATAS:

                    publicar_dds(
                        writer,
                        texto,
                        action,
                        target
                    )

                    resposta = frase_imediata(action)

                else:

                    historico.append({
                        "role": "user",
                        "content": texto
                    })

                    resposta = conversar(historico)

                    historico.append({
                        "role": "assistant",
                        "content": resposta
                    })

                print(f"[Johnny]: {resposta}")

                falar(
                    resposta,
                    audio_client
                )

                time.sleep(0.3)

            finally:

                if os.path.exists(ficheiro):
                    os.remove(ficheiro)

    except KeyboardInterrupt:

        print("\nA desligar o Johnny...")


if __name__ == "__main__":
    main()
