# Guia de Funcionamento do Módulo de Movimentação (Grupo 6)

Este módulo atua como a interface de baixo nivel (Low-Level / High Level Control) entre a rede de orquestração do robô (CycloneDDS) e o hardware físico do Unitree G1

Ele é respondsável por atuar como um "joystick virtual": recebe comandos de velocidade pura da Navegação, traduz esses comandos para o modelo de equilíbrio nativo do G1 (via Unitree SDK2), e devolve a odometria real dos motores para a rede.

## Interface DDS

O nosso módulo respeita estritamente os ficheiros `idl_ri.py` e `qos_profiles.py` localizados na diretoria `src/`.

## Subscrições (O que o módulo recebe)

* **`rt/motion/cmd_vel`** (`CmdVel` | QoS: `MOTION`): Recebe comandos de velocidade linear e angular (50Hz) da Navegação.
* **`rt/orchestration/state`** (`OrchestratorState` | QoS: `ORCHESTRATION`): Ouve o estado global da missão. Atua como **Kill Switch**: se `active_modules.motion` for `False`, o módulo corta imediatamente a velocidade dos motores para `0.0` por segurança.

## Publicações (O que o módulo entrega)

* **`rt/motion/odometry`** (`OdometryMsg` | QoS: `ODOMETRY`): Publica a odometria real lida diretamente dos encoders/IMU do robô a 50Hz. **Nota para o SLAM:** O `yaw` é extraído do IMU e convertido automaticamente para Quaternião (`x, y, z, w`).
* **`rt/orchestration/heartbeat`** (`Heartbeat` | QoS: `HEARTBEAT`): Sinal de vida emitido a 1Hz para a Orquestração saber que a ligação ao hardware está estável.

## Dependências

Não existe ficheiro `requirements.txt` local nesta pasta. O módulo assum que o ambiente virtual global do laboratório já possui:
* `cyclonedds`
* `unitree_sdk2py`

## Notas finais

Deve-se garantir que a porta de rede definida no código é a correta.