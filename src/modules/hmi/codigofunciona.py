#!/usr/bin/env python3

# python -c "import webrtcvad; print('webrtcvad OK')"
# python -c "import zmq, lz4.frame; print('ZMQ e LZ4 OK')"
# python -c "import cyclonedds; print('CycloneDDS OK')"
# python -c "from faster_whisper import WhisperModel; print('Whisper OK')"
# python -c "import ollama; print('Ollama OK')"

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


WHISPER_MODEL  = "large-v3-turbo"
OLLAMA_MODEL   = "qwen2.5:1.5b"
TOPIC_NAME     = "HRICommands"
AUDIO_TEMP     = "temp_hri.wav"
AUDIO_RESP     = "resposta_hri.mp3"
AUDIO_RESP_WAV = "resposta_hri_16k_mono.wav"

NET_INTERFACE = "enp117s0"
ROBOT_VOLUME = 100
AUDIO_TOPIC    = b"g1_audio"
G1_IP          = "192.168.123.164"
PORT           = 5556
ZMQ_TIMEOUT    = 5
# VAD_THRESHOLD       = 4500
# VAD_SILENCE_SECS    = 1.2
# VAD_MIN_SPEECH_SECS = 0.4

# Volume mínimo (RMS) para aceitar que existe som relevante no áudio.
# Mesmo usando WebRTC VAD, mantemos este filtro para ignorar silêncio absoluto
# ou ruídos muito fracos.
# Se o robô não detetar bem a fala, baixa este valor.
# Se o robô apanhar demasiado ruído de fundo, aumenta este valor.
VAD_RMS_MIN = 1200


# Grau de agressividade do WebRTC VAD.
# Este valor controla o quão seletivo o detetor é ao decidir se há voz humana.
#
# 0 = pouco agressivo: aceita mais áudio como fala, mas pode apanhar mais ruído.
# 1 = ligeiramente mais seletivo.
# 2 = equilibrado; boa opção inicial.
# 3 = muito agressivo: corta mais ruído, mas pode falhar fala baixa ou distante.
WEBRTC_VAD_MODE = 3


# Tamanho de cada frame enviada para o WebRTC VAD, em milissegundos.
# O WebRTC VAD só aceita frames de 10, 20 ou 30 ms.
#
# 30 ms costuma ser uma escolha estável porque dá ao VAD mais contexto
# para decidir se o áudio contém voz.
WEBRTC_FRAME_MS = 30


# Percentagem mínima de frames classificadas como voz dentro de um chunk.
#
# Exemplo:
# Se um chunk for dividido em 10 frames e este valor for 0.5,
# pelo menos 5 dessas frames têm de ser consideradas voz para o chunk
# ser aceite como fala.
#
# Valor mais baixo = mais permissivo.
# Valor mais alto = mais exigente.
WEBRTC_MIN_SPEECH_RATIO = 0.6


# Tempo de silêncio, em segundos, necessário para considerar que o utilizador
# acabou de falar e que o áudio já pode ser enviado para o Whisper.
#
# Se estiver muito baixo, pode cortar frases antes do fim.
# Se estiver muito alto, o sistema demora mais a responder depois da fala.
VAD_SILENCE_SECS = 1.2


# Duração mínima, em segundos, que a fala tem de ter para ser aceite.
# Serve para ignorar ruídos curtos, estalos, cliques ou pequenas interferências.
#
# Se o valor for muito baixo, pode aceitar ruídos como comandos.
# Se for muito alto, pode ignorar comandos curtos como "para" ou "sim".
VAD_MIN_SPEECH_SECS = 0.4


# Tempo de áudio guardado antes de o sistema detetar oficialmente a voz.
# Isto evita cortar o início de comandos curtos, como "para", "sim" ou "não".
PRE_BUFFER_SECS = 0.5

# Ganho aplicado ao áudio antes de enviar para o Whisper.
# Ajuda quando a voz vem baixa ou distante.
# 1.0 = sem ganho
# 1.5 = aumento moderado
# 2.0 = aumento forte, mas pode distorcer
AUDIO_GAIN = 1.6

# Tempo máximo que o robô pode ficar a gravar uma frase.
# Isto impede que o sistema fique preso em "Voz detetada" para sempre.
MAX_RECORDING_SECS = 5.0


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
    (["traz", "traze", "traga", "traz-me", "traz me"], "TRAZER", None),
    (["vai buscar", "ir buscar", "busca", "procura"], "IR_BUSCAR", None),
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

def aplicar_ganho_pcm16(pcm_bytes: bytes, ganho: float = AUDIO_GAIN) -> bytes:
    """
    Aplica ganho ao áudio PCM 16-bit.

    Serve para aumentar ligeiramente o volume antes de enviar o áudio para o Whisper.
    Usa clipping para evitar ultrapassar os limites de int16.
    """
    if not pcm_bytes or ganho == 1.0:
        return pcm_bytes

    samples = struct.unpack(f"{len(pcm_bytes) // 2}h", pcm_bytes)
    samples_com_ganho = []

    for s in samples:
        valor = int(s * ganho)

        # Limites do formato int16
        if valor > 32767:
            valor = 32767
        elif valor < -32768:
            valor = -32768

        samples_com_ganho.append(valor)

    return struct.pack(f"{len(samples_com_ganho)}h", *samples_com_ganho)

def parse_audio_parts(parts):
    if len(parts) == 4: return None, parts[2], parts[3]
    if len(parts) > 4:  return None, parts[2], b"".join(parts[3:])
    if len(parts) == 3: return None, parts[1], parts[2]
    if len(parts) > 3:  return None, parts[1], b"".join(parts[2:])
    return None, None, None


def pcm_to_mono(pcm_bytes: bytes, channels: int) -> bytes:
    """
    Garante que o áudio está em mono PCM 16-bit.

    Se já vier mono, devolve igual.
    Se vier com mais canais, fica apenas com o primeiro canal.
    """
    if channels <= 1:
        return pcm_bytes

    samples = struct.unpack(f"{len(pcm_bytes) // 2}h", pcm_bytes)

    mono_samples = samples[::channels]

    return struct.pack(f"{len(mono_samples)}h", *mono_samples)


def gerar_frames_webrtc(pcm_bytes: bytes, sample_rate: int, frame_ms: int = WEBRTC_FRAME_MS):
    """
    Divide o áudio em frames compatíveis com WebRTC VAD.

    O WebRTC VAD exige frames de 10, 20 ou 30 ms.
    """
    bytes_por_sample = 2
    samples_por_frame = int(sample_rate * frame_ms / 1000)
    bytes_por_frame = samples_por_frame * bytes_por_sample

    for i in range(0, len(pcm_bytes) - bytes_por_frame + 1, bytes_por_frame):
        yield pcm_bytes[i:i + bytes_por_frame]


def webrtc_tem_fala(vad, pcm_bytes: bytes, sample_rate: int, channels: int) -> bool:
    """
    Usa WebRTC VAD para decidir se um chunk contém voz.

    Devolve True se a percentagem de frames com fala for suficiente.
    """
    if sample_rate not in (8000, 16000, 32000, 48000):
        return False

    pcm_mono = pcm_to_mono(pcm_bytes, channels)

    frames = list(gerar_frames_webrtc(pcm_mono, sample_rate, WEBRTC_FRAME_MS))

    if not frames:
        return False

    frames_com_fala = 0

    for frame in frames:
        try:
            if vad.is_speech(frame, sample_rate):
                frames_com_fala += 1
        except Exception:
            return False

    ratio_fala = frames_com_fala / len(frames)

    return ratio_fala >= WEBRTC_MIN_SPEECH_RATIO

def gravar():
    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.connect(f"tcp://{G1_IP}:{PORT}")
    sock.setsockopt(zmq.SUBSCRIBE, AUDIO_TOPIC)
    sock.setsockopt(zmq.RCVTIMEO, ZMQ_TIMEOUT * 1000)

    audio_buffer = bytearray()

    # Pré-buffer: guarda áudio imediatamente anterior à deteção de voz.
    # Isto ajuda a não perder o início da frase.
    pre_buffer = deque()
    pre_buffer_duration = 0.0

    last_sr, last_ch = 48000, 1
    speech_started = False
    speech_duration = 0.0
    silence_duration = 0.0
    speech_frames = 0
    silence_frames = 0

    # Conta o tempo total desde que a voz foi detetada.
    # Serve como segurança caso o VAD nunca encontre silêncio.
    recording_duration = 0.0

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

            # Duração do chunk.
            # Divide por last_ch para ficar correto caso o áudio venha com mais de 1 canal.
            chunk_secs = (len(pcm) // 2) / (last_sr * max(last_ch, 1))

            rms = calcular_rms(pcm)

            # WebRTC VAD tenta perceber se há voz humana.
            vad_ok = webrtc_tem_fala(vad, pcm, last_sr, last_ch)

            # Decisão híbrida:
            # - RMS evita processar silêncio absoluto
            # - WebRTC VAD evita confundir ruído com fala
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

                    # Quando a voz começa, acrescentamos o áudio guardado antes.
                    # Isto evita perder o início da frase.
                    for old_pcm, _ in pre_buffer:
                        audio_buffer.extend(old_pcm)

                    pre_buffer.clear()
                    pre_buffer_duration = 0.0

                silence_duration = 0.0
                speech_duration += chunk_secs
                recording_duration += chunk_secs
                audio_buffer.extend(pcm)

            elif not speech_started:
                # Enquanto ainda não começou a fala, guardamos um pequeno histórico.
                pre_buffer.append((pcm, chunk_secs))
                pre_buffer_duration += chunk_secs

                # Mantém apenas os últimos PRE_BUFFER_SECS segundos.
                while pre_buffer_duration > PRE_BUFFER_SECS and pre_buffer:
                    _, dur = pre_buffer.popleft()
                    pre_buffer_duration -= dur

            else:
                # Já começou fala, mas agora este chunk foi considerado silêncio.
                silence_duration += chunk_secs
                recording_duration += chunk_secs
                audio_buffer.extend(pcm)

                if is_silence and silence_duration >= VAD_SILENCE_SECS:
                    if speech_duration >= VAD_MIN_SPEECH_SECS:
                        print("[MIC] Silêncio -> processar")
                        break
                    else:
                        # Fala demasiado curta: provavelmente ruído.
                        audio_buffer.clear()
                        pre_buffer.clear()

                        speech_started = False
                        speech_duration = 0.0
                        silence_duration = 0.0
                        speech_frames = 0
                        silence_frames = 0
                        pre_buffer_duration = 0.0
                        recording_duration = 0.0

            # Segurança: se depois de detetar voz o sistema nunca encontrar silêncio,
            # processa na mesma após alguns segundos.
            if speech_started and recording_duration >= MAX_RECORDING_SECS:
                print("[MIC] Tempo máximo de gravação atingido -> processar")
                break

        if not audio_buffer:
            return None

        # Aplica ganho antes de guardar o ficheiro para o Whisper.
        audio_final = aplicar_ganho_pcm16(bytes(audio_buffer), ganho=AUDIO_GAIN)

        with wave.open(AUDIO_TEMP, "wb") as wf:
            wf.setnchannels(last_ch)
            wf.setsampwidth(2)
            wf.setframerate(last_sr)
            wf.writeframes(audio_final)

        return AUDIO_TEMP

    finally:
        sock.close()
        ctx.term()


def read_wav_local(wav_path: str):
    """
    Lê um ficheiro WAV e devolve:
    - pcm_data: bytes PCM
    - sample_rate: frequência de amostragem
    - num_channels: número de canais
    - is_ok: True se conseguiu ler corretamente
    """
    try:
        with wave.open(wav_path, "rb") as wf:
            sample_rate = wf.getframerate()
            num_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            num_frames = wf.getnframes()
            pcm_data = wf.readframes(num_frames)

        # O áudio do professor exige PCM 16-bit.
        is_ok = sample_width == 2 and len(pcm_data) > 0

        return pcm_data, sample_rate, num_channels, is_ok

    except Exception as e:
        print(f"[ROBOT AUDIO] Erro ao ler WAV: {e}")
        return b"", 0, 0, False


def converter_mp3_para_wav_16k_mono(mp3_path: str, wav_path: str):
    """
    Converte o MP3 gerado pelo edge-tts para WAV 16 kHz mono.
    Este é o formato exigido pelo código de áudio do professor.
    """
    comando = [
        "ffmpeg",
        "-y",
        "-loglevel", "error",
        "-i", mp3_path,
        "-ac", "1",
        "-ar", "16000",
        "-acodec", "pcm_s16le",
        wav_path
    ]

    subprocess.run(comando, check=True)


class RobotSpeaker:
    """
    Classe responsável por enviar a voz do Johnny para o altifalante do Unitree G1.

    Fluxo:
    1. Recebe o MP3 gerado pelo edge-tts
    2. Converte para WAV 16 kHz mono
    3. Lê o WAV como PCM
    4. Envia o PCM para o robô através do AudioClient
    """

    def __init__(self, net_interface: str = NET_INTERFACE, volume: int = ROBOT_VOLUME):
        self.net_interface = net_interface
        self.volume = volume
        self.audio_client = None
        self.disponivel = False

        try:
            print("[ROBOT AUDIO] A inicializar AudioClient...")

            ChannelFactoryInitialize(0, self.net_interface)

            self.audio_client = AudioClient()
            self.audio_client.SetTimeout(10.0)
            self.audio_client.Init()
            self.audio_client.SetVolume(self.volume)

            self.disponivel = True
            print("[ROBOT AUDIO] Altifalante do robô pronto.")

        except Exception as e:
            print(f"[ROBOT AUDIO] Erro ao inicializar áudio do robô: {e}")
            self.disponivel = False

    def falar_mp3(self, mp3_path: str) -> bool:
        """
        Converte o MP3 para WAV 16 kHz mono e reproduz no robô.
        """
        if not self.disponivel:
            return False

        try:
            converter_mp3_para_wav_16k_mono(mp3_path, AUDIO_RESP_WAV)

            pcm_data, sample_rate, num_channels, is_ok = read_wav_local(AUDIO_RESP_WAV)

            print(f"[ROBOT AUDIO] Read success: {is_ok}")
            print(f"[ROBOT AUDIO] Sample rate: {sample_rate} Hz")
            print(f"[ROBOT AUDIO] Channels: {num_channels}")
            print(f"[ROBOT AUDIO] PCM byte length: {len(pcm_data)}")

            if not is_ok or sample_rate != 16000 or num_channels != 1:
                print("[ROBOT AUDIO] Erro: WAV tem de estar em 16 kHz mono.")
                return False

            app_name = "hri_response"
            stream_id = f"response_{int(time.time() * 1000)}"

            # Envia o áudio PCM para o robô.
            self.audio_client.PlayStream(app_name, stream_id, pcm_data)

            # Espera aproximadamente o tempo de duração do áudio.
            duracao_audio = len(pcm_data) / (sample_rate * 2 * num_channels)
            time.sleep(duracao_audio + 0.2)

            self.audio_client.PlayStop(app_name)

            return True

        except Exception as e:
            print(f"[ROBOT AUDIO] Erro ao reproduzir no robô: {e}")
            return False

def falar(texto, speaker: Optional[RobotSpeaker] = None):
    """
    Gera a voz do Johnny e tenta reproduzir pelo altifalante do robô.
    Se falhar, toca no PC como fallback.
    """

    async def _g():
        await edge_tts.Communicate(texto, "pt-PT-DuarteNeural").save(AUDIO_RESP)

    # 1. Gerar MP3 com edge-tts
    asyncio.run(_g())

    # 2. Tentar reproduzir pelo altifalante do robô
    if speaker is not None and speaker.disponivel:
        ok = speaker.falar_mp3(AUDIO_RESP)

        if ok:
            return

    # 3. Fallback no PC
    print("[ROBOT AUDIO] Fallback: a reproduzir no PC.")

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
        print("ERRO: ollama serve + ollama pull qwen2.5:1.5b")
        sys.exit(1)

    speaker = RobotSpeaker()

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
                # segs, _ = whisper.transcribe(ficheiro, language="pt",
                #                              beam_size=5, best_of=5,
                #                              temperature=0.0,
                #                              condition_on_previous_text=False)
                segs, _ = whisper.transcribe(
                    ficheiro,
                    language="pt",
                    beam_size=5,
                    best_of=5,
                    temperature=0.0,
                    condition_on_previous_text=False,
                    initial_prompt=(
                        "Comandos possíveis em português: anda, para, recua, levanta-te, senta-te, "
                        "vira à esquerda, vira à direita, olha para mim, olha em frente, "
                        "vai buscar a bola de ténis, traz a bola de ténis, agarra a bola de ténis, "
                        "vai buscar o cubo de Rubik, traz o cubo de Rubik, agarra o cubo de Rubik, "
                        "vai buscar a pasta de dentes, traz a pasta de dentes, agarra a pasta de dentes, "
                        "sim, não, cancela, ok."
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
                falar(resposta, speaker)
                time.sleep(0.4)
            finally:
                if os.path.exists(ficheiro):
                    os.remove(ficheiro)

    except KeyboardInterrupt:
        print("\nA desligar o Johnny... Ate logo!")

if __name__ == "__main__":
    main()
