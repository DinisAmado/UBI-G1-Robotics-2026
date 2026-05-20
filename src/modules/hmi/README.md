# Módulo HMI — Unitree G1

Sistema de Interação Humano-Robô por voz para o robô Unitree G1.

**Ficheiro principal:** `hri_funciona.py`

**Fluxo:** microfone do robô (ZMQ) → VAD híbrido → Whisper (transcrição) → classificador → FSM confirmação → DDS + voz nos altifalantes + LEDs

---

## Dependências de sistema

```bash
sudo apt install ffmpeg
```

## Instalar dependências Python

```bash
pip install -r requirements.txt
```

## Correr

```bash
python hri_funciona.py
```

---

## Configuração (topo do ficheiro)

| Variável | Valor | Descrição |
|---|---|---|
| `WHISPER_MODEL` | `large-v3` | Modelo Whisper — melhor qualidade, requer GPU |
| `NET_INTERFACE` | `enp117s0` | Interface de rede ligada ao robô — verificar com `ip link show` |
| `G1_IP` | `192.168.123.164` | IP do robô |
| `PORT` | `5556` | Porto ZMQ do stream de áudio |
| `ROBOT_VOLUME` | `100` | Volume dos altifalantes do robô (0–100) |
| `VAD_RMS_MIN` | `1200` | Volume mínimo (RMS) para aceitar áudio — aumenta se apanhar ruído, baixa se não detetar voz |
| `WEBRTC_VAD_MODE` | `3` | Agressividade do WebRTC VAD: 0 (permissivo) a 3 (muito seletivo) |
| `VAD_SILENCE_SECS` | `1.2` | Segundos de silêncio para terminar gravação |
| `VAD_MIN_SPEECH_SECS` | `0.4` | Duração mínima de fala para não ignorar |
| `PRE_BUFFER_SECS` | `0.5` | Áudio guardado antes da deteção — evita perder o início do comando |
| `AUDIO_GAIN` | `1.6` | Ganho aplicado ao áudio antes do Whisper |
| `MAX_RECORDING_SECS` | `3.0` | Tempo máximo de gravação por comando |

---

## Requisitos de hardware

O Whisper `large-v3` com `compute_type="float16"` requer GPU NVIDIA com CUDA. Verificar com:

```bash
nvidia-smi
```

Se não houver GPU, alterar no código:
```python
whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
```

---

## Deteção de voz (VAD)

O sistema usa uma abordagem híbrida em dois níveis:

1. **RMS mínimo** (`VAD_RMS_MIN`) — filtra silêncio absoluto e ruídos muito fracos
2. **WebRTC VAD** (`webrtcvad`) — detetor de voz humana desenvolvido pela Google, robusto ao ruído dos motores do G1

O WebRTC VAD divide cada chunk de áudio em frames de 30ms e exige que pelo menos 60% sejam classificadas como voz. Um pré-buffer de 0.5s evita perder o início de comandos curtos como "para" ou "sim".

---

## Classificador de comandos

O classificador usa correspondência por palavras-chave com normalização de texto (minúsculas + remoção de acentos) e fronteiras de palavra para evitar falsas correspondências (ex: "ola" dentro de "bola").

Inclui verificação de segurança: se o texto contiver "não" e a ação classificada for CONFIRMAR, corrige automaticamente para CANCELAR.

Quando deteta um alvo (objeto) mas não reconhece a ação, assume TRAZER e entra em modo de confirmação — comportamento mais inteligente do que pedir para repetir.

---

## Ações possíveis

**Precisam de confirmação:**
`TRAZER` `AGARRAR`

**Publicam no DDS imediatamente:**
`ANDAR` `PARAR` `RECUAR` `LEVANTAR` `SENTAR` `VIRAR_ESQUERDA` `VIRAR_DIREITA` `OLHAR_INTERLOCUTOR` `OLHAR_FRENTE` `CUMPRIMENTAR` `APRESENTAR` `ESTADO_ATUAL` `REPETIR` `LARGAR`

**Controlo de fluxo:**
`CONFIRMAR` `CANCELAR`

## Alvos possíveis

`BOLA_DE_TENIS` `CUBO_DE_RUBIK` `PASTA_DE_DENTES` `NENHUM` `DESCONHECIDO`

---

## LEDs

| Cor | Estado |
|---|---|
| 🔵 Azul | A ouvir — à espera de comando |
| 🟠 Laranja | A processar (Whisper) ou ação pendente à espera de confirmação |
| 🟢 Verde | A falar — Johnny está a responder |
| 🔴 Vermelho | Cancelado ou comando não reconhecido |
| ⚫ Desligado | Sistema encerrado |

O `LedController` reutiliza o mesmo `AudioClient` do `RobotSpeaker` — o `ChannelFactoryInitialize` é chamado uma única vez.

---

## Voz nos altifalantes

1. Edge TTS gera MP3 em `pt-PT-DuarteNeural`
2. `ffmpeg` converte para WAV 16kHz mono (formato exigido pelo `AudioClient`)
3. `AudioClient` do SDK Unitree envia o PCM para os altifalantes do robô

Fallback automático para `pygame` se o `AudioClient` não estiver disponível.

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
| `timestamp` | str | `"2026-05-14T15:32:00"` |

O campo `confirmed` é sempre `True` — o sistema só publica após validação pela máquina de estados.
