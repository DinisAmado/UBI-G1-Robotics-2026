# Módulo HMI — Unitree G1

Interação humano-robô por voz: transcrição → classificação → confirmação → DDS.

## Dependências de sistema (Linux)

```bash
sudo apt install portaudio19-dev python3-pyaudio ffmpeg
```

## Instalar Ollama (modelo de conversa)

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:1.5b
```

## Instalar dependências Python

```bash
pip install -r requirements.txt
```

## Correr

Dois terminais:

```bash
# Terminal 1 — manter aberto
ollama serve

# Terminal 2
python hmi.py
```

> Na primeira execução o Whisper medium (~1.5 GB) é descarregado automaticamente.

## Listar microfones disponíveis

```bash
python hmi.py --list-mics
```

Se o microfone padrão não funcionar, edita a variável `MIC_DEVICE_INDEX` no topo do ficheiro com o índice correto.

## Tópico DDS publicado

**Nome:** `HRICommands`

| Campo | Tipo | Exemplo |
|---|---|---|
| `source` | str | `"HRI"` |
| `original_text` | str | `"traz-me a bola"` |
| `action` | str | `"TRAZER"` |
| `target` | str | `"BOLA_DE_TENIS"` |
| `confirmed` | bool | `True` |
| `timestamp` | str | `"2026-04-28T14:32:00"` |

### Actions possíveis
`ANDAR` `PARAR` `RECUAR` `LEVANTAR` `SENTAR` `VIRAR_ESQUERDA` `VIRAR_DIREITA`
`OLHAR_INTERLOCUTOR` `OLHAR_FRENTE` `CUMPRIMENTAR` `APRESENTAR` `ESTADO_ATUAL`
`REPETIR` `IR_BUSCAR` `TRAZER` `AGARRAR` `LARGAR` `CONFIRMAR` `CANCELAR` `DESCONHECIDA`

### Targets possíveis
`BOLA_DE_TENIS` `CUBO_DE_RUBIK` `PASTA_DE_DENTES` `NENHUM` `DESCONHECIDO`

### Lógica de confirmação
Ações com objetos (`TRAZER`, `IR_BUSCAR`, `AGARRAR`) só são publicadas **após confirmação verbal** do utilizador.
`LARGAR` e ações de movimento são publicadas **imediatamente**.
