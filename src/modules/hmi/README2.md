# Módulo HMI — Unitree G1 (com LEDs + Voz + DDS)

Interação humano-robô por voz:
microfone → VAD → Whisper → classificação → confirmação → DDS → resposta TTS → áudio no robô + LEDs.

---

## Dependências de sistema (Linux)

```bash
sudo apt update
sudo apt install -y portaudio19-dev python3-pyaudio ffmpeg
Instalar Ollama (modelo de linguagem local)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:1.5b

Depois correr o servidor:

ollama serve
Instalar dependências Python
pip install -r requirements.txt
requirements.txt (com instalação explicada)
# =========================
# CORE AUDIO / ML
# =========================

faster-whisper
# Whisper optimizado para transcrição local
# instalação:
pip install faster-whisper


webrtcvad
# Voice Activity Detection (detetar fala vs silêncio)
# instalação:
pip install webrtcvad


numpy
# operações numéricas base
# instalação:
pip install numpy


# =========================
# AUDIO / TTS / PLAYBACK
# =========================

edge-tts
# TTS (voz neural da Microsoft)
# instalação:
pip install edge-tts


pygame
# playback de áudio no PC (fallback)
# instalação:
pip install pygame


# =========================
# COMMUNICATION (STREAM AUDIO)
# =========================

pyzmq
# comunicação ZeroMQ (stream áudio do robô)
# instalação:
pip install pyzmq


lz4
# compressão/descompressão áudio
# instalação:
pip install lz4


# =========================
# DDS (UNITREE / ROS-LIKE MIDDLEWARE)
# =========================

cyclonedds
# middleware DDS
# instalação:
pip install cyclonedds


# =========================
# UNITREE SDK
# =========================

unitree-sdk2py
# SDK oficial Unitree G1
# instalação:
pip install unitree-sdk2py


# =========================
# LLM LOCAL
# =========================

ollama
# cliente Python para Ollama
# instalação:
pip install ollama


# =========================
# SYSTEM / UTILITIES
# =========================

dataclasses; python_version < "3.7"
# já incluído no Python moderno
# instalação:
# não precisa instalar
Correr o sistema

Dois terminais:

Terminal 1 (LLM)
ollama serve
Terminal 2 (HMI + robô)
python hmi.py
LEDs (novo módulo integrado)

Estados:

AZUL → a ouvir
VERDE → a falar
VERMELHO → comando cancelado
OFF → desligado

Exemplo de uso no código:

leds.ouvir()
leds.falar()
leds.cancelar()
leds.desligar()
DDS publicado

Topic: HRICommands

source: "HRI"
original_text: texto original do utilizador
action: ação classificada
target: objeto alvo
confirmed: True
timestamp: ISO datetime
Ações suportadas
Movimento

ANDAR, PARAR, RECUAR, VIRAR_ESQUERDA, VIRAR_DIREITA

Interação

OLHAR_INTERLOCUTOR, OLHAR_FRENTE, CUMPRIMENTAR, APRESENTAR

Estado

ESTADO_ATUAL, REPETIR

Manipulação

IR_BUSCAR, TRAZER, AGARRAR, LARGAR

Confirmação

CONFIRMAR, CANCELAR

Targets suportados

BOLA_DE_TENIS
CUBO_DE_RUBIK
PASTA_DE_DENTES
NENHUM
DESCONHECIDO

Lógica do sistema
Voz detetada via WebRTC VAD + RMS
Transcrição com Whisper local
Classificação por regras (rápida e determinística)
LLM usado apenas para fallback conversacional
Ações com objetos exigem confirmação
DDS publica ações para o robô
TTS gera resposta e envia áudio para o G1
LEDs indicam estado do sistema em tempo real
Fluxo completo

Utilizador fala
→ VAD detecta voz
→ Whisper transcreve
→ classificador interpreta intenção
→ (opcional) confirmação
→ DDS publica ação
→ LLM responde (se necessário)
→ edge-tts gera voz
→ robô reproduz áudio
→ LEDs atualizam estado
