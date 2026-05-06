#!/usr/bin/env python3
import os, re, sys, math, struct, asyncio, unicodedata
import zmq, lz4.frame, wave, time
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

WHISPER_MODEL  = "medium"
OLLAMA_MODEL   = "qwen2.5:1.5b"
TOPIC_NAME     = "HRICommands"
AUDIO_TEMP     = "temp_hri.wav"
AUDIO_RESP     = "resposta_hri.mp3"
AUDIO_TOPIC    = b"g1_audio"
G1_IP          = "192.168.123.164"
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

def falar(texto):
    async def _g():
        await edge_tts.Communicate(texto, "pt-PT-DuarteNeural").save(AUDIO_RESP)
    asyncio.run(_g())
    pygame.mixer.init()
    pygame.mixer.music.load(AUDIO_RESP)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    pygame.mixer.quit()

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
        print("Ollama pronto!\n")
    except:
        print("ERRO: ollama serve + ollama pull qwen2.5:1.5b"); sys.exit(1)

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
                falar(resposta)
            finally:
                if os.path.exists(ficheiro):
                    os.remove(ficheiro)

    except KeyboardInterrupt:
        print("\nA desligar o Johnny... Ate logo!")

if __name__ == "__main__":
    main()
