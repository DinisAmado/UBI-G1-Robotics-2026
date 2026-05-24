#!/usr/bin/env python3
"""
hri_funciona.py — Sistema HRI completo + integração com Orquestração
Grupo 5 — Robótica Inteligente 2025/2026

Fluxo: microfone ZMQ → VAD → Whisper → Classificador → FSM → Intent DDS + TTS + LEDs
Publica:   rt/hmi/intent    (Intent)   — intenção do operador após confirmação
Subscreve: rt/hmi/feedback  (Feedback) — estado da orquestração → TTS + LEDs
"""

import os, re, sys, math, struct, asyncio, unicodedata
import zmq, lz4.frame, wave, time, subprocess, logging, threading
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
from collections import deque

import pygame
from faster_whisper import WhisperModel
import edge_tts
import webrtcvad

# ── CycloneDDS (orquestração + Unitree SDK) ───────────────────────────────────
from cyclonedds.domain import DomainParticipant
from cyclonedds.topic import Topic
from cyclonedds.pub import Publisher, DataWriter
from cyclonedds.sub import Subscriber, DataReader
from cyclonedds.idl import IdlStruct, IdlEnum
from cyclonedds.idl.types import sequence, uint8
from cyclonedds.idl.annotations import key
from enum import auto

# ── Unitree SDK ───────────────────────────────────────────────────────────────
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient

# ── QoS e IDL da orquestração ─────────────────────────────────────────────────
from qos_profiles import QOS_HMI
from idl_ri import (
    Header, Intent, Acao, Feedback, Status, OrchestrationState,
)

log = logging.getLogger("hmi")

# ==============================================================================
# CONFIGURAÇÃO
# ==============================================================================
WHISPER_MODEL  = "large-v3"
TOPIC_NAME     = "HRICommands"
AUDIO_TEMP     = "temp_hri.wav"
AUDIO_RESP     = "resposta_hri.mp3"
AUDIO_RESP_WAV = "resposta_hri_16k_mono.wav"

NET_INTERFACE = "enp117s0"
ROBOT_VOLUME  = 100
DOMAIN_ID     = 0

AUDIO_TOPIC = b"g1_audio"
G1_IP       = "192.168.123.164"
PORT        = 5556
ZMQ_TIMEOUT = 5

VAD_RMS_MIN             = 1200
WEBRTC_VAD_MODE         = 3
WEBRTC_FRAME_MS         = 30
WEBRTC_MIN_SPEECH_RATIO = 0.6
VAD_SILENCE_SECS        = 1.2
VAD_MIN_SPEECH_SECS     = 0.4
PRE_BUFFER_SECS         = 0.5
AUDIO_GAIN              = 1.6
MAX_RECORDING_SECS      = 3.0

LED_A_OUVIR   = (0, 0, 255)
LED_A_FALAR   = (0, 255, 0)
LED_CANCELADO = (255, 0, 0)
LED_OFF       = (0, 0, 0)

ACOES_COM_CONFIRMACAO = {"TRAZER", "AGARRAR"}
ACOES_IMEDIATAS = {
    "ANDAR", "PARAR", "RECUAR", "LEVANTAR", "SENTAR",
    "VIRAR_ESQUERDA", "VIRAR_DIREITA", "OLHAR_INTERLOCUTOR",
    "OLHAR_FRENTE", "CUMPRIMENTAR", "APRESENTAR",
    "ESTADO_ATUAL", "REPETIR", "LARGAR",
}

# Mapeamento HRI → Acao da orquestração
MAPA_ACAO = {
    "TRAZER":   Acao.RECOLHER,
    "AGARRAR":  Acao.RECOLHER,
    "ENTREGAR": Acao.ENTREGAR,
    "SEGUIR":   Acao.SEGUIR,
    "PARAR":    Acao.PARAR,
    "LARGAR":   Acao.LARGA,
    "ANDAR":    Acao.PARAR,
    "RECUAR":   Acao.PARAR,
    "LEVANTAR": Acao.PARAR,
    "SENTAR":   Acao.PARAR,
}

MAPA_TARGET = {
    "BOLA_DE_TENIS":   "bola",   # nome que o grupo visão usa
    "CUBO_DE_RUBIK":   "cubo",   # nome que o grupo visão usa
    "PASTA_DE_DENTES": "pasta",  # nome que o grupo visão usa
    "NENHUM":          "",
    "DESCONHECIDO":    "",
}

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
# CLASSIFICADOR
# ==============================================================================
def normalizar(texto):
    return "".join(c for c in unicodedata.normalize("NFD", texto.lower())
                   if unicodedata.category(c) != "Mn")

def contem(texto, frase):
    if " " in frase:
        return frase in texto
    return bool(re.search(r"(?<![a-z])" + re.escape(frase) + r"(?![a-z])", texto))

REGRAS = [
    (["traz", "traze", "traga", "traz-me", "traz me",
      "trav", "tras", "trage", "trag", "faz", "faz-me", "fas",
      "vai buscar", "ir buscar", "busca", "procura",
      "e ai buscar", "la buscar", "vai la buscar",
      "leva", "leva-me", "traga-me"],               "TRAZER",             None),
    (["agarra", "pega", "apanha"],                  "AGARRAR",            None),
    (["larga"],                                     "LARGAR",             None),
    (["anda", "avanca", "vai para a frente"],        "ANDAR",              "NENHUM"),
    (["para", "stop"],                              "PARAR",              "NENHUM"),
    (["recua", "vai para tras"],                    "RECUAR",             "NENHUM"),
    (["vira a esquerda", "esquerda"],               "VIRAR_ESQUERDA",     "NENHUM"),
    (["vira a direita", "direita"],                 "VIRAR_DIREITA",      "NENHUM"),
    (["levanta"],                                   "LEVANTAR",           "NENHUM"),
    (["senta"],                                     "SENTAR",             "NENHUM"),
    (["olha para mim"],                             "OLHAR_INTERLOCUTOR", "NENHUM"),
    (["olha em frente", "olha para a frente"],      "OLHAR_FRENTE",       "NENHUM"),
    (["cumprimenta", "ola"],                        "CUMPRIMENTAR",       "NENHUM"),
    (["quem es", "apresenta-te", "como te chamas"], "APRESENTAR",         "NENHUM"),
    (["como estas", "estado atual"],                "ESTADO_ATUAL",       "NENHUM"),
    (["repete", "diz outra vez"],                   "REPETIR",            "NENHUM"),
    (["nao", "cancela", "esquece", "afinal nao",
      "nao quero", "nao obrigado", "deixa estar",
      "esquece isso"],                              "CANCELAR",           "NENHUM"),
    (["sim", "claro", "faz isso", "pode ser",
      "exato", "por favor", "faz favor", "confirmo",
      "quero sim", "sim por favor", "vai",
      "faz isso sim"],                              "CONFIRMAR",          "NENHUM"),
]

REGRAS_TARGET = [
    (["bola de tenis", "bola", "tenis", "boletenas"], "BOLA_DE_TENIS"),
    (["cubo de rubik", "cubo magico", "cubo", "rubik"], "CUBO_DE_RUBIK"),
    (["pasta de dentes", "pasta", "dentes"], "PASTA_DE_DENTES"),
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
            if override is not None:
                target = override
            break
    if action in ACOES_COM_CONFIRMACAO and target == "NENHUM":
        target = "DESCONHECIDO"
    # Segurança: "não" sobrepõe sempre CONFIRMAR
    if action == "CONFIRMAR" and contem(t, "nao"):
        action = "CANCELAR"
    return {"action": action, "target": target}

# ==============================================================================
# AUDIO — VAD + ZMQ
# ==============================================================================
def calcular_rms(pcm_bytes):
    if len(pcm_bytes) < 2: return 0.0
    samples = struct.unpack(f"{len(pcm_bytes)//2}h", pcm_bytes)
    return math.sqrt(sum(s*s for s in samples) / len(samples))

def aplicar_ganho_pcm16(pcm_bytes, ganho=AUDIO_GAIN):
    if not pcm_bytes or ganho == 1.0: return pcm_bytes
    samples = struct.unpack(f"{len(pcm_bytes)//2}h", pcm_bytes)
    result = [max(-32768, min(32767, int(s * ganho))) for s in samples]
    return struct.pack(f"{len(result)}h", *result)

def parse_audio_parts(parts):
    if len(parts) == 4: return None, parts[2], parts[3]
    if len(parts) > 4:  return None, parts[2], b"".join(parts[3:])
    if len(parts) == 3: return None, parts[1], parts[2]
    if len(parts) > 3:  return None, parts[1], b"".join(parts[2:])
    return None, None, None

def pcm_to_mono(pcm_bytes, channels):
    if channels <= 1: return pcm_bytes
    samples = struct.unpack(f"{len(pcm_bytes)//2}h", pcm_bytes)
    mono = samples[::channels]
    return struct.pack(f"{len(mono)}h", *mono)

def gerar_frames_webrtc(pcm_bytes, sample_rate, frame_ms=WEBRTC_FRAME_MS):
    bytes_por_frame = int(sample_rate * frame_ms / 1000) * 2
    for i in range(0, len(pcm_bytes) - bytes_por_frame + 1, bytes_por_frame):
        yield pcm_bytes[i:i + bytes_por_frame]

def webrtc_tem_fala(vad, pcm_bytes, sample_rate, channels):
    if sample_rate not in (8000, 16000, 32000, 48000): return False
    pcm_mono = pcm_to_mono(pcm_bytes, channels)
    frames = list(gerar_frames_webrtc(pcm_mono, sample_rate))
    if not frames: return False
    fala = sum(1 for f in frames if vad.is_speech(f, sample_rate))
    return (fala / len(frames)) >= WEBRTC_MIN_SPEECH_RATIO

def gravar():
    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.RCVTIMEO, ZMQ_TIMEOUT * 1000)
    sock.setsockopt(zmq.SUBSCRIBE, AUDIO_TOPIC)
    sock.connect(f"tcp://{G1_IP}:{PORT}")

    audio_buffer = bytearray()
    pre_buffer = deque()
    pre_buffer_duration = 0.0
    last_sr, last_ch = 48000, 1
    speech_started = False
    speech_duration = silence_duration = recording_duration = 0.0
    speech_frames = silence_frames = 0
    vad = webrtcvad.Vad(WEBRTC_VAD_MODE)

    print("\n[MIC] A ouvir...")
    try:
        while True:
            try:
                parts = sock.recv_multipart()
            except zmq.Again:
                if speech_started and speech_duration >= VAD_MIN_SPEECH_SECS:
                    print("[MIC] Timeout ZMQ após fala -> processar")
                    break
                continue

            if not parts or parts[0] != AUDIO_TOPIC:
                continue

            _, header, pcm_compressed = parse_audio_parts(parts)
            if header and len(header) >= 5:
                last_sr = int.from_bytes(header[:4], "little")
                last_ch = header[4]

            try:
                pcm = lz4.frame.decompress(pcm_compressed)
            except:
                continue
            if not pcm:
                continue

            chunk_secs = (len(pcm) // 2) / (last_sr * max(last_ch, 1))
            rms = calcular_rms(pcm)
            vad_ok = webrtc_tem_fala(vad, pcm, last_sr, last_ch)

            if rms > VAD_RMS_MIN and vad_ok:
                speech_frames += 1; silence_frames = 0
            else:
                silence_frames += 1; speech_frames = 0

            is_speech  = speech_frames >= 3
            is_silence = silence_frames >= 8

            if is_speech:
                if not speech_started:
                    speech_started = True
                    print("[MIC] Voz detetada")
                    for old_pcm, _ in pre_buffer:
                        audio_buffer.extend(old_pcm)
                    pre_buffer.clear()
                    pre_buffer_duration = 0.0
                silence_duration = 0.0
                speech_duration += chunk_secs
                recording_duration += chunk_secs
                audio_buffer.extend(pcm)
            elif not speech_started:
                pre_buffer.append((pcm, chunk_secs))
                pre_buffer_duration += chunk_secs
                while pre_buffer_duration > PRE_BUFFER_SECS and pre_buffer:
                    _, dur = pre_buffer.popleft()
                    pre_buffer_duration -= dur
            else:
                silence_duration += chunk_secs
                recording_duration += chunk_secs
                audio_buffer.extend(pcm)
                if is_silence and silence_duration >= VAD_SILENCE_SECS:
                    if speech_duration >= VAD_MIN_SPEECH_SECS:
                        print("[MIC] Silêncio -> processar")
                        break
                    else:
                        audio_buffer.clear(); pre_buffer.clear()
                        speech_started = False
                        speech_duration = silence_duration = recording_duration = 0.0
                        speech_frames = silence_frames = 0
                        pre_buffer_duration = 0.0

            if speech_started and recording_duration >= MAX_RECORDING_SECS:
                print("[MIC] Tempo máximo atingido -> processar")
                break

        if not audio_buffer:
            return None

        audio_final = aplicar_ganho_pcm16(bytes(audio_buffer))
        with wave.open(AUDIO_TEMP, "wb") as wf:
            wf.setnchannels(last_ch); wf.setsampwidth(2)
            wf.setframerate(last_sr); wf.writeframes(audio_final)
        return AUDIO_TEMP
    finally:
        sock.close(); ctx.term()

# ==============================================================================
# LEDS
# ==============================================================================
class LedController:
    """
    Controla os LEDs do G1 com um thread dedicado que repete
    continuamente a cor desejada — necessário porque o firmware
    do robô (sport_mode) repõe a cor padrão se não houver sinal.
    """
    def __init__(self, audio_client):
        self.audio_client = audio_client
        self.disponivel = audio_client is not None
        self._cor = LED_A_OUVIR
        self._lock = threading.Lock()
        self._stop = False

        if self.disponivel:
            t = threading.Thread(target=self._loop, daemon=True)
            t.start()

    def _loop(self):
        """Thread que envia a cor atual ao robô a cada 200ms."""
        while not self._stop:
            try:
                with self._lock:
                    r, g, b = self._cor
                self.audio_client.LedControl(int(r), int(g), int(b))
            except Exception as e:
                print(f"[LEDS] Erro no thread: {e}")
            time.sleep(0.2)

    def set_color(self, r, g, b):
        if not self.disponivel: return
        with self._lock:
            self._cor = (r, g, b)

    def ouvir(self):
        print("[LEDS] Azul: a ouvir")
        self.set_color(*LED_A_OUVIR)

    def falar(self):
        print("[LEDS] Verde: a falar")
        self.set_color(*LED_A_FALAR)

    def pendente(self):
        print("[LEDS] Laranja: a processar / pendente")
        self.set_color(255, 165, 0)

    def cancelar(self):
        print("[LEDS] Vermelho: cancelado")
        self.set_color(*LED_CANCELADO)

    def nao_percebeu(self):
        print("[LEDS] Vermelho: não percebeu")
        self.set_color(*LED_CANCELADO)

    def desligar(self):
        self._stop = True
        print("[LEDS] Desligado")
        self.set_color(*LED_OFF)

# ==============================================================================
# ALTIFALANTES DO ROBÔ
# ==============================================================================
def converter_mp3_para_wav_16k_mono(mp3_path, wav_path):
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", mp3_path, "-ac", "1", "-ar", "16000",
        "-acodec", "pcm_s16le", wav_path
    ], check=True)

def read_wav_local(wav_path):
    try:
        with wave.open(wav_path, "rb") as wf:
            sr = wf.getframerate(); ch = wf.getnchannels()
            sw = wf.getsampwidth(); pcm = wf.readframes(wf.getnframes())
        return pcm, sr, ch, (sw == 2 and len(pcm) > 0)
    except Exception as e:
        print(f"[ROBOT AUDIO] Erro ao ler WAV: {e}")
        return b"", 0, 0, False

class RobotSpeaker:
    def __init__(self):
        self.audio_client = None
        self.disponivel = False
        try:
            print("[ROBOT AUDIO] A inicializar AudioClient...")
            self.audio_client = AudioClient()
            self.audio_client.SetTimeout(10.0)
            self.audio_client.Init()
            self.audio_client.SetVolume(ROBOT_VOLUME)
            self.disponivel = True
            print("[ROBOT AUDIO] Altifalante do robô pronto.")
        except Exception as e:
            print(f"[ROBOT AUDIO] Erro: {type(e).__name__}: {e}")

    def falar_mp3(self, mp3_path):
        if not self.disponivel: return False
        try:
            converter_mp3_para_wav_16k_mono(mp3_path, AUDIO_RESP_WAV)
            pcm_data, sr, ch, is_ok = read_wav_local(AUDIO_RESP_WAV)
            if not is_ok or sr != 16000 or ch != 1:
                print("[ROBOT AUDIO] Formato WAV inválido.")
                return False
            stream_id = f"response_{int(time.time() * 1000)}"
            self.audio_client.PlayStream("hri_response", stream_id, pcm_data)
            duracao = len(pcm_data) / (sr * 2 * ch)
            time.sleep(duracao + 0.2)
            self.audio_client.PlayStop("hri_response")
            return True
        except Exception as e:
            print(f"[ROBOT AUDIO] Erro ao reproduzir: {e}")
            return False

def falar(texto, speaker, leds):
    async def _g():
        await edge_tts.Communicate(texto, "pt-PT-DuarteNeural").save(AUDIO_RESP)
    asyncio.run(_g())

    leds.falar()  # verde durante o audio

    if speaker.disponivel:
        ok = speaker.falar_mp3(AUDIO_RESP)
        # NAO chama leds.ouvir() aqui — o loop principal decide a cor seguinte
        if ok:
            for f in [AUDIO_RESP, AUDIO_RESP_WAV]:
                if os.path.exists(f): os.remove(f)
            return

    # Fallback pygame
    print("[ROBOT AUDIO] Fallback: a reproduzir no PC.")
    pygame.mixer.init()
    pygame.mixer.music.load(AUDIO_RESP)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    pygame.mixer.quit()
    # NAO chama leds.ouvir() aqui — o loop principal decide a cor seguinte
    if os.path.exists(AUDIO_RESP): os.remove(AUDIO_RESP)

# ==============================================================================
# FRASES
# ==============================================================================
NOME_TARGET = {
    "BOLA_DE_TENIS":   "a bola de ténis",
    "CUBO_DE_RUBIK":   "o cubo mágico",
    "PASTA_DE_DENTES": "a pasta de dentes",
    "DESCONHECIDO":    "o objeto",
    "NENHUM":          "isso",
}

def frase_confirmacao(action, target):
    nome = NOME_TARGET.get(target, "o objeto")
    if action == "TRAZER":  return f"Queres que eu traga {nome}?"
    if action == "AGARRAR": return f"Queres que eu agarre {nome}?"
    return f"Confirmas a ação com {nome}?"

def frase_execucao(action, target):
    nome = NOME_TARGET.get(target, "o objeto")
    if action == "TRAZER":  return f"Combinado! Vou já trazer {nome}."
    if action == "AGARRAR": return f"Combinado! Vou agarrar {nome}."
    return "Combinado, vou já!"

def frase_imediata(action):
    return {
        "ANDAR":              "Ok, a andar!",
        "PARAR":              "Ok, paro aqui.",
        "RECUAR":             "Ok, a recuar.",
        "LEVANTAR":           "Ok, a levantar!",
        "SENTAR":             "Ok, a sentar.",
        "VIRAR_ESQUERDA":     "Ok, a virar à esquerda.",
        "VIRAR_DIREITA":      "Ok, a virar à direita.",
        "OLHAR_INTERLOCUTOR": "Ok, a olhar para ti.",
        "OLHAR_FRENTE":       "Ok, a olhar em frente.",
        "CUMPRIMENTAR":       "Olá! Muito prazer!",
        "APRESENTAR":         "Olá! Sou o Johnny, um robô Unitree.",
        "ESTADO_ATUAL":       "Estou operacional!",
        "REPETIR":            "Claro, repito!",
        "LARGAR":             "Ok, a largar!",
    }.get(action, "Ok!")

# ==============================================================================
# HMI NODE — DDS da orquestração
# ==============================================================================
class HmiNode:
    def __init__(self):
        self._seq = 0
        self._dp  = DomainParticipant(DOMAIN_ID)
        pub = Publisher(self._dp)
        sub = Subscriber(self._dp)
        t_intent   = Topic(self._dp, "rt/hmi/intent",   Intent,   qos=QOS_HMI)
        t_feedback = Topic(self._dp, "rt/hmi/feedback", Feedback, qos=QOS_HMI)
        self._w_intent   = DataWriter(pub, t_intent)
        self._r_feedback = DataReader(sub, t_feedback)
        log.info("HmiNode inicializado no domínio %d", DOMAIN_ID)

    def _header(self):
        self._seq += 1
        return Header(timestamp_ns=time.time_ns(), frame_id="hmi", seq=self._seq)

    def send_intent(self, acao: Acao, alvo: str, comando_grasping: str = "") -> None:
        self._w_intent.write(Intent(
            header=self._header(),
            acao=acao, alvo=alvo,
            comando_grasping=comando_grasping,
        ))
        log.info("[INTENT] acao=%s  alvo=%s  grasping=%s", acao.name, alvo, comando_grasping)

    def poll_feedback(self) -> Optional[Feedback]:
        samples = self._r_feedback.take(1)
        return samples[0] if samples else None

    def publicar_intent_hri(self, action: str, target: str) -> bool:
        acao_orq = MAPA_ACAO.get(action)
        if acao_orq is None:
            log.warning("[HMI] Ação '%s' sem mapeamento", action)
            return False
        if target not in MAPA_TARGET:
            log.warning("[HMI] Target '%s' não reconhecido", target)
        alvo_orq = MAPA_TARGET.get(target, "")
        self.send_intent(acao_orq, alvo=alvo_orq, comando_grasping=alvo_orq)
        return True

def processar_feedback(fb: Feedback, speaker, leds) -> None:
    log.info("[FEEDBACK] status=%s  estado=%s  msg=%s",
             fb.status.name, fb.state.name, fb.message)
    msg = fb.message if fb.message else FEEDBACK_ESTADO.get(fb.state, "")
    if not msg: return
    if fb.status == Status.DONE:
        leds.falar()
    elif fb.status == Status.FAILED:
        leds.nao_percebeu()
    elif fb.state in (OrchestrationState.NAVIGATING_TO_TABLE,
                      OrchestrationState.NAVIGATING_TO_PERSON,
                      OrchestrationState.GRASPING_OBJECT,
                      OrchestrationState.LOCATING_OBJECT):
        leds.pendente()
    falar(msg, speaker, leds)

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    logging.basicConfig(level=logging.INFO)

    # ChannelFactoryInitialize TEM de ser o PRIMEIRO — antes de qualquer DomainParticipant
    print("A inicializar SDK Unitree...")
    ChannelFactoryInitialize(0, NET_INTERFACE)

    # DDS orquestração (usa o mesmo domínio CycloneDDS já inicializado)
    hmi = HmiNode()

    print("A carregar Whisper...")
    whisper = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="float16")
    print("Whisper pronto!")

    # AudioClient reutiliza o canal já inicializado pelo ChannelFactoryInitialize
    speaker = RobotSpeaker()
    leds    = LedController(speaker.audio_client)

    pending = None

    print("=" * 52)
    print("   SISTEMA HMI -- UNITREE G1  (Ctrl+C para sair)")
    print("=" * 52 + "\n")

    falar("Olá! Eu sou o Johnny. Em que posso ajudá-lo?", speaker, leds)

    try:
        while True:
            # Feedback do orquestrador
            fb = hmi.poll_feedback()
            if fb:
                processar_feedback(fb, speaker, leds)

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
                        "sim, não, cancela."
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
