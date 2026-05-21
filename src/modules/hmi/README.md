# Módulo HMI — Unitree G1

Sistema de Interação Humano-Robô por voz para o robô Unitree G1, com integração na orquestração do projeto.

**Ficheiro principal:** `hmi_main.py`

**Fluxo:** microfone do robô (ZMQ) → VAD híbrido → Whisper (transcrição) → classificador → FSM confirmação → Intent DDS + voz nos altifalantes + LEDs

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
python hmi_main.py
```

---

## Ficheiros necessários na mesma pasta

| Ficheiro | Origem |
|---|---|
| `hmi_main.py` | Este módulo |
| `idl_ri.py` | Partilhado pela orquestração |
| `qos_profiles.py` | Partilhado pela orquestração |

---

## Configuração (topo do ficheiro)

| Variável | Valor | Descrição |
|---|---|---|
| `WHISPER_MODEL` | `large-v3` | Modelo Whisper — requer GPU NVIDIA com CUDA |
| `NET_INTERFACE` | `enp117s0` | Interface de rede ligada ao robô — verificar com `ip link show` |
| `G1_IP` | `192.168.123.164` | IP do robô |
| `PORT` | `5556` | Porto ZMQ do stream de áudio |
| `DOMAIN_ID` | `0` | Domínio DDS partilhado com a orquestração |
| `ROBOT_VOLUME` | `100` | Volume dos altifalantes do robô (0–100) |
| `VAD_RMS_MIN` | `1200` | Volume mínimo para aceitar áudio — aumenta se apanhar ruído |
| `WEBRTC_VAD_MODE` | `3` | Agressividade do VAD: 0 (permissivo) a 3 (seletivo) |
| `VAD_SILENCE_SECS` | `1.2` | Segundos de silêncio para terminar gravação |
| `VAD_MIN_SPEECH_SECS` | `0.4` | Duração mínima de fala para não ignorar |
| `PRE_BUFFER_SECS` | `0.5` | Áudio guardado antes da deteção — evita perder início do comando |
| `AUDIO_GAIN` | `1.6` | Ganho aplicado ao áudio antes do Whisper |
| `MAX_RECORDING_SECS` | `3.0` | Tempo máximo de gravação por comando |

---

## Requisitos de hardware

O Whisper `large-v3` com `compute_type="float16"` requer GPU NVIDIA com CUDA. Verificar disponibilidade:

```bash
nvidia-smi
```

Se não houver GPU, alterar no código:
```python
whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
```

---

## Ordem de inicialização (crítica)

O `ChannelFactoryInitialize` do SDK Unitree **tem de ser chamado antes** de qualquer `DomainParticipant`. O código garante esta ordem:

```
ChannelFactoryInitialize(0, NET_INTERFACE)  ← 1º
HmiNode()           → DomainParticipant    ← 2º
RobotSpeaker()      → AudioClient          ← 3º
LedController()     → reutiliza AudioClient ← 4º
```

---

## Deteção de voz (VAD)

Abordagem híbrida em dois níveis:

1. **RMS mínimo** (`VAD_RMS_MIN`) — filtra silêncio absoluto e ruídos muito fracos
2. **WebRTC VAD** (`webrtcvad`) — detetor de voz humana da Google, robusto ao ruído dos motores do G1

Um pré-buffer de 0.5s evita perder o início de comandos curtos como "para" ou "sim".

---

## Classificador de comandos

Usa correspondência por palavras-chave com normalização (minúsculas + remoção de acentos) e fronteiras de palavra para evitar falsas correspondências.

Inclui verificação de segurança: se o texto contiver "não" e a ação classificada for CONFIRMAR, corrige automaticamente para CANCELAR.

Quando deteta um alvo (objeto) mas não reconhece a ação, assume TRAZER e entra em modo de confirmação.

---

## Ações possíveis

**Precisam de confirmação:**
`TRAZER` `AGARRAR`

**Publicam Intent imediatamente:**
`ANDAR` `PARAR` `RECUAR` `LEVANTAR` `SENTAR` `VIRAR_ESQUERDA` `VIRAR_DIREITA` `OLHAR_INTERLOCUTOR` `OLHAR_FRENTE` `CUMPRIMENTAR` `APRESENTAR` `ESTADO_ATUAL` `REPETIR` `LARGAR`

**Controlo de fluxo:**
`CONFIRMAR` `CANCELAR`

## Alvos possíveis

`BOLA_DE_TENIS` → `"bola"` · `CUBO_DE_RUBIK` → `"cubo"` · `PASTA_DE_DENTES` → `"pasta"`

---

## Integração com a Orquestração

### Tópicos DDS

| Tópico | Direção | Tipo | QoS |
|---|---|---|---|
| `rt/hmi/intent` | Publica | `Intent` | `QOS_HMI` |
| `rt/hmi/feedback` | Subscreve | `Feedback` | `QOS_HMI` |

### Mapeamento de ações HRI → Acao

| Ação HRI | Acao orquestração | Alvo |
|---|---|---|
| `TRAZER` / `AGARRAR` | `Acao.RECOLHER` | `"bola"` / `"cubo"` / `"pasta"` |
| `PARAR` / `ANDAR` / `RECUAR` | `Acao.PARAR` | `""` |
| `LARGAR` | `Acao.LARGA` | `""` |

### Feedback da orquestração

Quando chega `Feedback`, o HMI apresenta o estado ao utilizador via TTS e ajusta os LEDs:

| `fb.status` | LED | Exemplo de mensagem |
|---|---|---|
| `Status.DONE` | 🟢 Verde | mensagem do orquestrador |
| `Status.FAILED` | 🔴 Vermelho | mensagem do orquestrador |
| `OrchestrationState.NAVIGATING_*` / `GRASPING_*` | 🟠 Laranja | "A navegar até à mesa." |

---

## LEDs

O firmware do G1 (sport_mode) repõe continuamente a cor azul. Para competir com esse comportamento, o `LedController` usa um **thread dedicado** que reenvia a cor desejada a cada 200ms.

| Cor | Estado |
|---|---|
| 🔵 Azul | A ouvir — à espera de comando |
| 🟠 Laranja | A processar (Whisper) ou ação pendente |
| 🟢 Verde | A falar — Johnny está a responder |
| 🔴 Vermelho | Cancelado ou não percebeu |
| ⚫ Desligado | Sistema encerrado |

---

## Voz nos altifalantes

1. Edge TTS gera MP3 em `pt-PT-DuarteNeural`
2. `ffmpeg` converte para WAV 16kHz mono
3. `AudioClient` do SDK Unitree envia PCM para os altifalantes do robô

Fallback automático para `pygame` se o `AudioClient` não estiver disponível.

---

## Requisitos Python

```
faster-whisper
edge-tts
pygame
webrtcvad
pyzmq
lz4
cyclonedds
# unitree_sdk2py — instalar do repositório SDK
```
