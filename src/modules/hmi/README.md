# Módulo HMI — Unitree G1

Sistema de Interação Humano-Robô por voz para o robô Unitree G1.

**Fluxo:** microfone do robô (ZMQ) → Whisper (transcrição) → classificador → confirmação → DDS + voz nos altifalantes + LEDs

---

## Dependências de sistema (Linux)

```bash
sudo apt install ffmpeg
```

## Instalar Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:1.5b
```

## Instalar dependências Python

```bash
pip install -r requirements.txt
```

## Correr

```bash
# Terminal 1 — manter aberto
ollama serve

# Terminal 2
python hri_ana.py
```

---

## Configuração (topo do ficheiro)

| Variável | Valor padrão | Descrição |
|---|---|---|
| `WHISPER_MODEL` | `large-v3-turbo` | Modelo Whisper — `medium` é mais rápido, `large-v3-turbo` é mais preciso para PT |
| `OLLAMA_MODEL` | `qwen2.5:1.5b` | Modelo de conversa livre |
| `NET_INTERFACE` | `enp117s0` | Interface de rede ligada ao robô — verificar com `ip link show` |
| `G1_IP` | `192.168.123.164` | IP do robô |
| `PORT` | `5556` | Porto ZMQ do stream de áudio |
| `ROBOT_VOLUME` | `100` | Volume dos altifalantes do robô (0–100) |
| `VAD_RMS_MIN` | `1200` | Volume mínimo (RMS) para aceitar áudio — aumenta se apanhar ruído, baixa se não detetar voz |
| `WEBRTC_VAD_MODE` | `3` | Agressividade do WebRTC VAD: 0 (permissivo) a 3 (muito seletivo) |
| `VAD_SILENCE_SECS` | `1.2` | Segundos de silêncio para terminar a gravação |
| `VAD_MIN_SPEECH_SECS` | `0.4` | Duração mínima de fala para não ignorar |
| `PRE_BUFFER_SECS` | `0.5` | Áudio guardado antes da deteção — evita perder o início do comando |
| `AUDIO_GAIN` | `1.6` | Ganho aplicado ao áudio antes do Whisper — aumenta se a voz vier baixa |
| `MAX_RECORDING_SECS` | `5.0` | Tempo máximo de gravação por comando (segurança) |

---

## Deteção de voz (VAD)

O sistema usa uma abordagem híbrida em dois níveis:

1. **RMS mínimo** (`VAD_RMS_MIN`) — filtra silêncio absoluto e ruídos muito fracos
2. **WebRTC VAD** (`webrtcvad`) — detetor de voz humana desenvolvido pela Google, robusto ao ruído dos motores do G1

O WebRTC VAD divide cada chunk de áudio em frames de 30ms e exige que pelo menos 60% sejam classificadas como voz. Um pré-buffer de 0.5s é mantido para não perder o início de comandos curtos como "para" ou "sim".

---

## LEDs

Os LEDs do G1 indicam o estado do sistema em tempo real:

| Cor | Estado |
|---|---|
| 🔵 Azul | A ouvir — à espera de comando |
| 🟢 Verde | A falar — Johnny está a responder |
| 🔴 Vermelho | Cancelado — utilizador disse não |
| ⚫ Desligado | Sistema encerrado |

O `LedController` reutiliza o mesmo `AudioClient` do `RobotSpeaker` — o `ChannelFactoryInitialize` é chamado **uma única vez** para evitar conflitos.

---

## Voz nos altifalantes

O `RobotSpeaker` gere a reprodução de voz no G1:

1. Edge TTS gera MP3 em `pt-PT-DuarteNeural`
2. `ffmpeg` converte para WAV 16kHz mono (formato exigido pelo `AudioClient`)
3. `AudioClient` do SDK Unitree envia o PCM para os altifalantes do robô

Se o `AudioClient` não estiver disponível, o áudio é reproduzido no PC como fallback (via `pygame`).

---

## Tópico DDS publicado

**Nome:** `HRICommands`

| Campo | Tipo | Exemplo |
|---|---|---|
| `source` | str | `"HRI"` |
| `original_text` | str | `"traz-me a bola"` |
| `action` | str | `"TRAZER"` |
| `target` | str | `"BOLA_DE_TENIS"` |
| `confirmed` | bool | `True` |
| `timestamp` | str | `"2026-05-13T14:32:00"` |

### Actions possíveis
`ANDAR` `PARAR` `RECUAR` `LEVANTAR` `SENTAR` `VIRAR_ESQUERDA` `VIRAR_DIREITA`
`OLHAR_INTERLOCUTOR` `OLHAR_FRENTE` `CUMPRIMENTAR` `APRESENTAR` `ESTADO_ATUAL`
`REPETIR` `IR_BUSCAR` `TRAZER` `AGARRAR` `LARGAR` `CONFIRMAR` `CANCELAR` `DESCONHECIDA`

### Targets possíveis
`BOLA_DE_TENIS` `CUBO_DE_RUBIK` `PASTA_DE_DENTES` `NENHUM` `DESCONHECIDO`

### Lógica de confirmação
`TRAZER`, `IR_BUSCAR` e `AGARRAR` só são publicados **após confirmação verbal** do utilizador.
Todas as outras ações são publicadas **imediatamente**.

---

## Requisitos Python

```
faster-whisper
pygame
edge-tts
cyclonedds
ollama
pyzmq
lz4
webrtcvad
```
