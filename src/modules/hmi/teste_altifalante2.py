#!/usr/bin/env python3
import os, re, sys, math, struct, asyncio, unicodedata, subprocess
import zmq, lz4.frame, wave, time
from datetime import datetime
from dataclasses import dataclass
from typing import Optional

from faster_whisper import WhisperModel
import edge_tts
import ollama as ollama_client
from cyclonedds.idl import IdlStruct
from cyclonedds.domain import DomainParticipant
from cyclonedds.topic import Topic
from cyclonedds.pub import DataWriter
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient

WHISPER_MODEL  = "large-v3-turbo"
OLLAMA_MODEL   = "qwen2.5:1.5b"
TOPIC_NAME     = "HRICommands"
AUDIO_TEMP     = "temp_hri.wav"
AUDIO_RESP     = "resposta_hri.mp3"
AUDIO_MONO     = "resposta_mono.wav"
AUDIO_TOPIC    = b"g1_audio"
G1_IP          = "192.168.123.164"
NET_INTERFACE  = "enp117s0"
PORT           = 5556
ZMQ_TIMEOUT    = 5
VAD_THRESHOLD       = 4500
VAD_SILENCE_SECS    = 1.2
VAD_MIN_SPEECH_SECS = 0.4

ACOES_COM_CONFIRMACAO = {"IR_BUSCAR", "TRAZER", "AGARRAR"}
ACOES_IMEDIATAS = {
    "ANDAR", "PARAR", "RECUAR", "LEVANTAR", "SENTAR",
    "VIRAR_ESQUERDA", "VIRAR_DIREITA", "OLHAR_INTERLOCUTOR",
    "OLHAR_FRENTE", "CUMPRIMENTAR", "APRESENTAR",
    "ESTADO_ATUAL", "REPETIR", "LARGAR",
}

@dataclass
class HRICommand(IdlStruct):
    source: str
    original_text: str
    action: str
    target: str
    confirmed: bool
    timestamp: str

def normalizar(texto):
    return "".join(c for c in unicodedata.normalize("NFD", texto.lower()) if unicodedata.category(c) != "Mn")

def contem(texto, frase):
    if " " in frase:
        return frase in texto
    return bool(re.search(r"(?<![a-z])" + re.escape(frase) + r"(?![a-z])", texto))

REGRAS = [
    (["vai buscar", "ir buscar", "busca", "vai ate", "atras da"], "IR_BUSCAR", None),
    (["traz", "traze", "traga"],                                  "TRAZER",    None),
    (["agarra", "pega", "apanha"],                                "AGARRAR",   None),
    (["larga"],                                                   "LARGAR",    None),
    (["anda", "avanca", "vai para a frente"],                     "ANDAR",     "NENHUM"),
    (["para", "stop"],                                            "PARAR",     "NENHUM"),
    (["recua", "vai para tras"],                                  "RECUAR",    "NENHUM"),
    (["vira a esquerda", "esquerda"],                             "VIRAR_ESQUERDA", "NENHUM"),
    (["vira a direita", "direita"],                               "VIRAR_DIREITA",  "NENHUM"),
    (["levanta"],                                                 "LEVANTAR",  "NENHUM"),
    (["senta"],                                                   "SENTAR",    "NENHUM"),
    (["olha para mim"],                                           "OLHAR_INTERLOCUTOR", "NENHUM"),
    (["olha em frente", "olha para a frente"],                    "OLHAR_FRENTE", "NENHUM"),
    (["cumprimenta", "ola"],                                      "CUMPRIMENTAR", "NENHUM"),
    (["quem es", "apresenta-te", "como te chamas"],               "APRESENTAR",   "NENHUM"),
    (["como estas", "estado atual"],                              "ESTADO_ATUAL", "NENHUM"),
    (["repete", "diz outra vez"],                                 "REPETIR",      "NENHUM"),
    (["sim", "claro", "faz isso", "ok", "pode ser",
      "exato", "por favor", "faz favor", "confirmo"],             "CONFIRMAR",    "NENHUM"),
    (["nao", "cancela", "esquece", "afinal nao"],                 "CANCELAR",     "NENHUM"),
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

def calcular_rms(pcm_bytes):
    if len(pcm_bytes) < 2: return 0.0
    samples = struct.unpack(f"{len(pcm_bytes)//2}h", pcm_bytes)
    return math.sqrt(sum(s*s for s in samples) / len(samples))

def parse_audio_parts(parts):
    if len(parts) == 4: return None, parts[2], parts[3]
    if len(parts) > 4:  return None, parts[2], b"".join(parts[3:])
    if len(parts) == 3: return None, parts[1], parts[2]
    if len(parts) > 3:  return None, parts[1], b"".join(parts[2:])
    return None, None, None

def gravar():
    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.connect(f"tcp://{G1_IP}:{PORT}")
    sock.setsockopt(zmq.SUBSCRIBE, AUDIO_TOPIC)
    sock.setsockopt(zmq.RCVTIMEO, ZMQ_TIMEOUT * 1000)

    audio_buffer = bytearray()
    last_sr, last_ch = 48000, 1
    speech_started = False
    speech_duration = silence_duration = 0.0
    speech_frames = silence_frames = 0

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

            chunk_secs = (len(pcm) // 2) / last_sr
            rms = calcular_rms(pcm)

            if rms > VAD_THRESHOLD:
                speech_frames += 1; silence_frames = 0
            else:
                silence_frames += 1; speech_frames = 0

            is_speech  = speech_frames >= 3
            is_silence = silence_frames >= 8

            if is_speech:
                if not speech_started:
                    speech_started = True
                    print("[MIC] Voz detetada")
                silence_duration = 0.0
                speech_duration += chunk_secs
                audio_buffer.extend(pcm)
            elif speech_started:
                silence_duration += chunk_secs
                audio_buffer.extend(pcm)
                if is_silence and silence_duration >= VAD_SILENCE_SECS:
                    if speech_duration >= VAD_MIN_SPEECH_SECS:
                        print("[MIC] Silencio -> processar")
                        break
                    else:
                        audio_buffer.clear()
                        speech_started = False
                        speech_duration = silence_duration = 0.0
                        speech_frames = silence_frames = 0

        if not audio_buffer:
            return None
        with wave.open(AUDIO_TEMP, "wb") as wf:
            wf.setnchannels(last_ch); wf.setsampwidth(2)
            wf.setframerate(last_sr); wf.writeframes(audio_buffer)
        return AUDIO_TEMP
    finally:
        sock.close(); ctx.term()

def ler_wav_mono16k(path):
    """Le um WAV 16kHz mono e devolve os bytes PCM raw."""
    with wave.open(path, "rb") as wf:
        assert wf.getframerate() == 16000, "Sample rate deve ser 16000"
        assert wf.getnchannels() == 1,     "Deve ser mono"
        assert wf.getsampwidth() == 2,     "Deve ser 16-bit"
        return wf.readframes(wf.getnframes())

def falar(texto, audio_client):

    async def _g():
        await edge_tts.Communicate(
            texto,
            "pt-PT-DuarteNeural"
        ).save(AUDIO_RESP)

    asyncio.run(_g())

    subprocess.run([
        "ffmpeg", "-y",
        "-i", AUDIO_RESP,
        "-ar", "16000",
        "-ac", "1",
        "-sample_fmt", "s16",
        AUDIO_MONO
    ], stdout=subprocess.DEVNULL,
       stderr=subprocess.DEVNULL)

    try:
        pcm_bytes = ler_wav_mono16k(AUDIO_MONO)

        CHUNK = 3200

        chunks = [
            pcm_bytes[i:i+CHUNK]
            for i in range(0, len(pcm_bytes), CHUNK)
        ]

        audio_client.SetVolume(100)

        print("[TTS] StartPlay")
        audio_client.StartPlay()

        for chunk in chunks:

            audio_client.PlayStream(
                "hri_response",
                16000,
                chunk
            )

            time.sleep(0.1)

        print("[TTS] StopPlay")
        audio_client.StopPlay()

        print("[TTS] Audio reproduzido")

    except Exception as e:
        print(f"[ERRO AudioClient] {e}")

    finally:

        try:
            audio_client.StopPlay()
        except:
            pass

        for f in [AUDIO_RESP, AUDIO_MONO]:
            if os.path.exists(f):
                os.remove(f)
                
def publicar_dds(writer, texto, action, target):
    cmd = HRICommand(source="HRI", original_text=texto, action=action, target=target,
                     confirmed=True, timestamp=datetime.now().isoformat(timespec="seconds"))
    writer.write(cmd)
    print(f"[DDS] PUBLICADO -- action={action}  target={target}")

NOME_TARGET = {
    "BOLA_DE_TENIS": "a bola de tenis", "CUBO_DE_RUBIK": "o cubo magico",
    "PASTA_DE_DENTES": "a pasta de dentes", "DESCONHECIDO": "o objeto", "NENHUM": "isso",
}

def frase_confirmacao(action, target):
    nome = NOME_TARGET.get(target, "o objeto")
    if action == "TRAZER":    return f"Queres que eu traga {nome}?"
    if action == "IR_BUSCAR": return f"Queres que eu va buscar {nome}?"
    if action == "AGARRAR":   return f"Queres que eu agarre {nome}?"
    return f"Confirmas a acao com {nome}?"

def frase_execucao(action, target):
    nome = NOME_TARGET.get(target, "o objeto")
    if action == "TRAZER":    return f"Combinado! Vou ja trazer {nome}."
    if action == "IR_BUSCAR": return f"Combinado! Vou ja buscar {nome}."
    if action == "AGARRAR":   return f"Combinado! Vou agarrar {nome}."
    return "Combinado, vou ja!"

def frase_imediata(action):
    return {
        "ANDAR": "Ok, a andar!", "PARAR": "Ok, paro aqui.", "RECUAR": "Ok, a recuar.",
        "LEVANTAR": "Ok, a levantar!", "SENTAR": "Ok, a sentar.",
        "VIRAR_ESQUERDA": "Ok, a virar a esquerda.", "VIRAR_DIREITA": "Ok, a virar a direita.",
        "OLHAR_INTERLOCUTOR": "Ok, a olhar para ti.", "OLHAR_FRENTE": "Ok, a olhar em frente.",
        "CUMPRIMENTAR": "Ola! Muito prazer!",
        "APRESENTAR": "Ola! Sou o Johnny, um robo Unitree.",
        "ESTADO_ATUAL": "Estou operacional!", "REPETIR": "Claro, repito!",
        "LARGAR": "Ok, a largar!",
    }.get(action, "Ok!")

def main():
    print("A carregar Whisper...")
    whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    print("Whisper pronto!")

    try:
        ollama_client.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": "ok"}])
        print("Ollama pronto!")
    except:
        print("ERRO: ollama serve + ollama pull qwen2.5:1.5b"); sys.exit(1)

    # Inicializar AudioClient do SDK Unitree
    print("A inicializar AudioClient...")
    ChannelFactoryInitialize(0, NET_INTERFACE)
    audio_client = AudioClient()
    audio_client.SetTimeout(10.0)
    audio_client.Init()
    print("AudioClient pronto!\n")

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
            ficheiro = gravar()
            if not ficheiro:
                continue
            try:
                segs, _ = whisper.transcribe(ficheiro, language="pt",
                                             beam_size=5, best_of=5,
                                             temperature=0.0,
                                             condition_on_previous_text=False)
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
                    resposta = "Nao ha nenhuma acao pendente."
                elif action == "CANCELAR" and pending:
                    resposta = "Ok, fico aqui entao."; pending = None
                elif action == "CANCELAR":
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
                falar(resposta, audio_client)
            finally:
                if os.path.exists(ficheiro):
                    os.remove(ficheiro)

    except KeyboardInterrupt:
        print("\nA desligar o Johnny... Ate logo!")

if __name__ == "__main__":
    main()
