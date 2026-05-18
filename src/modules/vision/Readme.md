# UBI-G1-ROBOTICS-2026

Arquitetura modular para controlo do robô G1 usando **CycloneDDS**.

## Estrutura

```bash
src/
├── modules/
│   ├── vision/
│   │   ├── obj/
│   │   │   ├── best.pt
│   │   │   └── main_objetos.py
│   │   │
│   │   ├── requirements_vision.txt
│   │   │
│   │   │
│   │   └── pes/
│   │       ├── face_landmarker.task
│   │       └── main_pessoas.py
│   │
│   ├── movement/
│   ├── navigation/
│   ├── grasping/
│   └── hmi/
│
├── idl_ri.py
├── qos_profiles.py
├── launch.py
├── main.py
└── requirements.txt
```

---

# CycloneDDS

Todos os módulos comunicam via DDS.

## Topics principais

| Topic                    | Tipo                | Uso                           |
| ------------------------ | ------------------- | ----------------------------- |
| `rt/vision/objects`      | `Objects`           | Objetos detetados             |
| `rt/grasp/command`       | `GraspCommand`      | Comandos para grasping        |
| `rt/orchestration/state` | `OrchestratorState` | Estado global do orquestrador |

---

# Vision

## `main_objetos.py`

Pipeline de visão com:

* YOLO
* RealSense RGB + Depth
* Estimativa 6-DOF
* Publicação DDS

### Subscreve

| Topic                    | Tipo                | Uso                                      |
| ------------------------ | ------------------- | ---------------------------------------- |
| `rt/orchestration/state` | `OrchestratorState` | Recebe o objeto alvo (`current_target_object`) |

### Publica

| Topic               | Tipo           | Dados                                          |
| ------------------- | -------------- | ---------------------------------------------- |
| `rt/vision/objects` | `Objects`      | Todos os objetos detetados com nome, confiança e pose 6-DOF |
| `rt/grasp/command`  | `GraspCommand` | Pose 6-DOF do objeto alvo (x, y, z, roll, pitch, yaw) |

### Funções principais

* deteção de objetos
* pose 6-DOF
* tracking EMA
* crop RGB do objeto
* integração ZMQ + DDS

---

## `main_pessoas.py`

Pipeline de deteção facial e pessoas usando:

* MediaPipe
* InsightFace
* RealSense

### Objetivo

* deteção de pessoas
* reconhecimento facial
* tracking distancia labial
* tracking humano
* publicação DDS

### Subscreve

| Topic                    | Tipo                |
| ------------------------ | ------------------- |
| `rt/orchestration/state` | `OrchestratorState` |

### Publica

| Topic               | Tipo      |
| ------------------- | --------- |
| `rt/vision/persons` | `Persons` |

---

# QoS

## `qos_profiles.py`

Perfis DDS centralizados:

* `QOS_VISION`
* `QOS_GRASP`
* `QOS_ORCHESTRATION`

---

# IDL

## `idl_ri.py`

Definição das mensagens DDS:

* `Header`
* `Image`
* `ObjectDetection`
* `Objects`
* `PersonDetection`
* `Persons`
* `GraspCommand`
* `OrchestratorState`

---

# Execução

## Instalar dependências

```bash
pip install -r requirements.txt
```

## Configurar CycloneDDS

```bash
export CYCLONEDDS_URI=file://$(pwd)/cyclonedds.xml
```

## Executar

Na raiz do repositório:

```bash
python src/main.py
```

ou diretamente:

```bash
python src/modules/vision/obj/main_objetos.py
```

---

# Convenções

## Imports

Os scripts assumem execução a partir da raiz do repositório.

Exemplo:

```python
from idl_ri import *
from qos_profiles import *
```

## Recomendado

Executar sempre:

```bash
cd UBI-G1-ROBOTICS-2026
```

antes de correr qualquer módulo.
