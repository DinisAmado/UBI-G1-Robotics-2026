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
# CONFIGURACAO
# ==============================================================================
WHISPER_MODEL  = "large-v3-turbo"
OLLAMA_MODEL   = "qwen2.5:1.5b"
TOPIC_NAME     = "HRICommands"
AUDIO_TEMP     = "temp_hri.wav"
AUDIO_RESP     = "resposta_hri.mp3"
AUDIO_RESP_WAV = "resposta_hri_16k_mono.wav"

NET_INTERFACE = "enp117s0"
ROBOT_VOLUME  = 100

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
MAX_RECORDING_SECS      = 5.0

# Cores LEDs
LED_A_OUVIR  = (0, 0, 255)   # azul
LED_A_FALAR  = (0, 255, 0)   # verde
LED_CANCELADO = (255, 0, 0)  # vermelho
LED_OFF      = (0, 0, 0)     # desligado

ACOES_COM_CONFIRMACAO = {"IR_BUSCAR", "TRAZER", "AGARRAR"}
ACOES_IMEDIATAS = {
    "ANDAR", "PARAR", "RECUAR", "LEVANTAR", "SENTAR",
    "VIRAR_ESQUERDA", "VIRAR_DIREITA", "OLHAR_INTERLOCUTOR",
    "OLHAR_FRENTE", "CUMPRIMENTAR", "APRESENTAR",
    "ESTADO_ATUAL", "REPETIR", "LARGAR",
}

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
# CLASSIFICADOR
# ==============================================================================
def normalizar(texto):
    return "".join(c for c in unicodedata.normalize("NFD", texto.lower()) if unicodedata.category(c) != "Mn")

def contem(texto, frase):
    if " " in frase:
        return frase in texto
    return bool(re.search(r"(?<![a-z])" + re.escape(frase) + r"(?![a-z])", texto))

REGRAS = [
    (["traz", "traze", "traga", "traz-me", "traz me"], "TRAZER",   None),
    (["vai buscar", "ir buscar", "busca", "procura"],  "IR_BUSCAR", None),
    (["agarra", "pega", "apanha"],                     "AGARRAR",  None),
    (["larga"],                                        "LARGAR",   None),
    (["anda", "avanca", "vai para a frente"],          "ANDAR",    "NENHUM"),
    (["para", "stop"],                                 "PARAR",    "NENHUM"),
    (["recua", "vai para tras"],                       "RECUAR",   "NENHUM"),
    (["vira a esquerda", "esquerda"],                  "VIRAR_ESQUERDA", "NENHUM"),
    (["vira a direita", "direita"],                    "VIRAR_DIREITA",  "NENHUM"),
    (["levanta"],                                      "LEVANTAR", "NENHUM"),
    (["senta"],                                        "SENTAR",   "NENHUM"),
    (["olha para mim"],                                "OLHAR_INTERLOCUTOR", "NENHUM"),
    (["olha em frente", "olha para a frente"],         "OLHAR_FRENTE",   "NENHUM"),
    (["cumprimenta", "ola"],                           "CUMPRIMENTAR",   "NENHUM"),
    (["quem es", "apresenta-te", "como te chamas"],    "APRESENTAR",     "NENHUM"),
    (["como estas", "estado atual"],                   "ESTADO_ATUAL",   "NENHUM"),
    (["repete", "diz outra vez"],                      "REPETIR",        "NENHUM"),
    (["sim", "claro", "faz isso", "ok", "pode ser",
      "exato", "por favor", "faz favor", "confirmo"],  "CONFIRMAR",      "NENHUM"),
    (["nao", "cancela", "esquece", "afinal nao"],      "CANCELAR",       "NENHUM"),
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
    return {"action": action, "target": target}

# ==============================================================================
# OLLAMA
# ==============================================================================
SYSTEM_PROMPT = (
    "Tu es o Johnny, um robo Unitree em Portugal. "
    "Fala sempre em Portugues de Portugal (pt-PT). Se muito breve e simpatico. "
    "Nunca uses mais de 2 frases."
)

def conversar(historico):
    try:
        r = ollama_client.chat(model=OLLAMA_MODEL, messages=historico)
        return r["message"]["content"].strip()
    except Exception as e:
        return "Desculpa, tive um problema."

# ==============================================================================
# AUDIO — VAD + ZMQ
# ==============================================================================
def calcular_rms(pcm_bytes):
    if len(pcm_bytes) < 2: return 0.0
    samples = struct.unpack(f"{len(pcm_bytes)//2}h", pcm_bytes)
    return math.sqrt(sum(s*s for s in samples) / len(samples))

def aplicar_ganho_pcm16(pcm_bytes, ganho=AUDIO_GAIN):
    if not pcm_bytes or ganho == 1.0:
        return pcm_bytes
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
    if channels <= 1:
        return pcm_bytes
    samples = struct.unpack(f"{len(pcm_bytes)//2}h", pcm_bytes)
    mono = samples[::channels]
    return struct.pack(f"{len(mono)}h", *mono)

def gerar_frames_webrtc(pcm_bytes, sample_rate, frame_ms=WEBRTC_FRAME_MS):
    bytes_por_frame = int(sample_rate * frame_ms / 1000) * 2
    for i in range(0, len(pcm_bytes) - bytes_por_frame + 1, bytes_por_frame):
        yield pcm_bytes[i:i + bytes_por_frame]

def webrtc_tem_fala(vad, pcm_bytes, sample_rate, channels):
    if sample_rate not in (8000, 16000, 32000, 48000):
        return False
    pcm_mono = pcm_to_mono(pcm_bytes, channels)
    frames = list(gerar_frames_webrtc(pcm_mono, sample_rate))
    if not frames:
        return False
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
                    print("[MIC] Timeout ZMQ apos fala -> processar")
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
                        print("[MIC] Silencio -> processar")
                        break
                    else:
                        audio_buffer.clear(); pre_buffer.clear()
                        speech_started = False
                        speech_duration = silence_duration = recording_duration = 0.0
                        speech_frames = silence_frames = 0
                        pre_buffer_duration = 0.0

            if speech_started and recording_duration >= MAX_RECORDING_SECS:
                print("[MIC] Tempo maximo atingido -> processar")
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
    def __init__(self, audio_client):
        # Reutiliza o AudioClient ja inicializado pelo RobotSpeaker
        self.audio_client = audio_client
        self.disponivel = audio_client is not None

    def set_color(self, r, g, b):
        if not self.disponivel:
            return
        try:
            self.audio_client.LedControl(int(r), int(g), int(b))
        except Exception as e:
            print(f"[LEDS] Erro: {e}")

    def ouvir(self):
        print("[LEDS] Azul: a ouvir")
        self.set_color(*LED_A_OUVIR)

    def falar(self):
        print("[LEDS] Verde: a falar")
        self.set_color(*LED_A_FALAR)

    def cancelar(self):
        print("[LEDS] Vermelho: cancelado")
        self.set_color(*LED_CANCELADO)

    def desligar(self):
        print("[LEDS] Desligado")
        self.set_color(*LED_OFF)

# ==============================================================================
# ALTIFALANTES DO ROBO
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
            # ChannelFactoryInitialize chamado UMA UNICA VEZ aqui
            ChannelFactoryInitialize(0, NET_INTERFACE)
            self.audio_client = AudioClient()
            self.audio_client.SetTimeout(10.0)
            self.audio_client.Init()
            self.audio_client.SetVolume(ROBOT_VOLUME)
            self.disponivel = True
            print("[ROBOT AUDIO] Altifalante do robo pronto.")
        except Exception as e:
            print(f"[ROBOT AUDIO] Erro: {e}")

    def falar_mp3(self, mp3_path):
        if not self.disponivel:
            return False
        try:
            converter_mp3_para_wav_16k_mono(mp3_path, AUDIO_RESP_WAV)
            pcm_data, sr, ch, is_ok = read_wav_local(AUDIO_RESP_WAV)
            if not is_ok or sr != 16000 or ch != 1:
                print("[ROBOT AUDIO] Formato WAV invalido.")
                return False
            app_name = "hri_response"
            stream_id = f"response_{int(time.time() * 1000)}"
            self.audio_client.PlayStream(app_name, stream_id, pcm_data)
            duracao = len(pcm_data) / (sr * 2 * ch)
            time.sleep(duracao + 0.2)
            self.audio_client.PlayStop(app_name)
            return True
        except Exception as e:
            print(f"[ROBOT AUDIO] Erro ao reproduzir: {e}")
            return False

def falar(texto, speaker, leds):
    async def _g():
        await edge_tts.Communicate(texto, "pt-PT-DuarteNeural").save(AUDIO_RESP)
    asyncio.run(_g())

    leds.falar()

    if speaker.disponivel:
        ok = speaker.falar_mp3(AUDIO_RESP)
        if ok:
            return

    # Fallback no PC
    print("[ROBOT AUDIO] Fallback: a reproduzir no PC.")
    pygame.mixer.init()
    pygame.mixer.music.load(AUDIO_RESP)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    pygame.mixer.quit()

# ==============================================================================
# DDS
# ==============================================================================
def publicar_dds(writer, texto, action, target):
    cmd = HRICommand(source="HRI", original_text=texto, action=action, target=target,
                     confirmed=True, timestamp=datetime.now().isoformat(timespec="seconds"))
    writer.write(cmd)
    print(f"[DDS] PUBLICADO -- action={action}  target={target}")

NOME_TARGET = {
    "BOLA_DE_TENIS": "a bola de ténis", "CUBO_DE_RUBIK": "o cubo mágico",
    "PASTA_DE_DENTES": "a pasta de dentes", "DESCONHECIDO": "o objeto", "NENHUM": "isso",
}

def frase_confirmacao(action, target):
    nome = NOME_TARGET.get(target, "o objeto")
    if action == "TRAZER":    return f"Queres que eu traga {nome}?"
    if action == "IR_BUSCAR": return f"Queres que eu vá buscar {nome}?"
    if action == "AGARRAR":   return f"Queres que eu agarre {nome}?"
    return f"Confirmas a ação com {nome}?"

def frase_execucao(action, target):
    nome = NOME_TARGET.get(target, "o objeto")
    if action == "TRAZER":    return f"Combinado! Vou já trazer {nome}."
    if action == "IR_BUSCAR": return f"Combinado! Vou já buscar {nome}."
    if action == "AGARRAR":   return f"Combinado! Vou agarrar {nome}."
    return "Combinado, vou já!"

def frase_imediata(action):
    return {
        "ANDAR": "Ok, a andar!", "PARAR": "Ok, paro aqui.", "RECUAR": "Ok, a recuar.",
        "LEVANTAR": "Ok, a levantar!", "SENTAR": "Ok, a sentar.",
        "VIRAR_ESQUERDA": "Ok, a virar à esquerda.", "VIRAR_DIREITA": "Ok, a virar à direita.",
        "OLHAR_INTERLOCUTOR": "Ok, a olhar para ti.", "OLHAR_FRENTE": "Ok, a olhar em frente.",
        "CUMPRIMENTAR": "Olá! Muito prazer!",
        "APRESENTAR": "Olá! Sou o Johnny, um robô Unitree.",
        "ESTADO_ATUAL": "Estou operacional!", "REPETIR": "Claro, repito!",
        "LARGAR": "Ok, a largar!",
    }.get(action, "Ok!")

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print("A carregar Whisper...")
    whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    print("Whisper pronto!")

    try:
        ollama_client.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": "ok"}])
        print("Ollama pronto!")
    except:
        print("ERRO: ollama serve + ollama pull qwen2.5:1.5b"); sys.exit(1)

    # RobotSpeaker inicializa o ChannelFactory e o AudioClient
    speaker = RobotSpeaker()

    # LedController reutiliza o mesmo AudioClient — sem double init
    leds = LedController(speaker.audio_client)

    participant = DomainParticipant()
    topic = Topic(participant, TOPIC_NAME, HRICommand)
    writer = DataWriter(participant, topic)
    historico = [{"role": "system", "content": SYSTEM_PROMPT}]
    pending = None

    print("=" * 52)
    print("   SISTEMA HRI -- UNITREE G1  (Ctrl+C para sair)")
    print("=" * 52 + "\n")

    try:
        while True:
            leds.ouvir()
            ficheiro = gravar()
            if not ficheiro:
                continue
            try:
                segs, _ = whisper.transcribe(
                    ficheiro, language="pt",
                    beam_size=5, best_of=5,
                    temperature=0.0,
                    condition_on_previous_text=False,
                    initial_prompt=(
                        "Comandos possiveis em portugues: anda, para, recua, levanta-te, senta-te, "
                        "vira a esquerda, vira a direita, olha para mim, olha em frente, "
                        "vai buscar a bola de tenis, traz a bola de tenis, agarra a bola de tenis, "
                        "vai buscar o cubo de Rubik, traz o cubo de Rubik, agarra o cubo de Rubik, "
                        "vai buscar a pasta de dentes, traz a pasta de dentes, agarra a pasta de dentes, "
                        "sim, nao, cancela, ok."
                    )
                )
                texto = "".join(s.text for s in segs).strip()
                if not texto:
                    print("[Whisper] Nao percebi nada.")
                    continue

                print(f"[Utilizador]: {texto}")
                result = classificar(texto)
                action, target = result["action"], result["target"]
                print(f"[Classificacao]: {action} / {target}")

                if action == "CONFIRMAR" and pending:
                    publicar_dds(writer, pending["texto"], pending["action"], pending["target"])
                    resposta = frase_execucao(pending["action"], pending["target"])
                    pending = None
                elif action == "CONFIRMAR":
                    resposta = "Não há nenhuma ação pendente."
                elif action == "CANCELAR" and pending:
                    leds.cancelar(); time.sleep(1)
                    resposta = "Ok, fico aqui então."; pending = None
                elif action == "CANCELAR":
                    leds.cancelar(); time.sleep(1)
                    resposta = "Ok, sem problema."
                elif action in ACOES_COM_CONFIRMACAO:
                    pending = {"action": action, "target": target, "texto": texto}
                    resposta = frase_confirmacao(action, target)
                elif action in ACOES_IMEDIATAS:
                    publicar_dds(writer, texto, action, target)
                    resposta = frase_imediata(action)
                else:
                    historico.append({"role": "user", "content": texto})
                    resposta = conversar(historico)
                    historico.append({"role": "assistant", "content": resposta})

                print(f"[Johnny]: {resposta}")
                falar(resposta, speaker, leds)
                time.sleep(0.4)

            finally:
                if os.path.exists(ficheiro):
                    os.remove(ficheiro)

    except KeyboardInterrupt:
        leds.desligar()
        print("\nA desligar o Johnny... Ate logo!")

if __name__ == "__main__":
    main()
