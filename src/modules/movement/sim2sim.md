# Guia de Exibição: Unitree G1 no MuJoCo (Sim2Sim)

Este guia é focado **exclusivamente na exibição do robô no simulador MuJoCo**. Não inclui ferramentas de treino (Isaac Gym), tornando a instalação muito mais leve e garantindo que funciona em qualquer computador (usa apenas o CPU).

## 1. Preparação Básica
Garantir que o sistema tem as ferramentas base de compilação:
```bash
sudo apt update
sudo apt install build-essential git curl -y
```

*(Nota: Instala o Miniconda caso o computador ainda não o tenha).*

## 2. Criação do Ambiente Python
Mantemos o Python 3.8 por ser a versão onde o RSL_RL v1.0.2 e o PyTorch antigo são 100% estáveis juntos.
```bash
conda create -n g1_mujoco python=3.8.10 -y
conda activate g1_mujoco
```

## 3. Instalação das Dependências
Faz o clone e instala apenas as três bibliotecas estritamente necessárias para a simulação.

**A. RSL_RL (Estrutura da Inteligência)**
```bash
cd ~
git clone https://github.com/leggedrobotics/rsl_rl.git
cd rsl_rl
git checkout v1.0.2
pip install -e . --no-build-isolation
```

**B. Unitree SDK2 Python (Ferramentas de Comunicação)**
```bash
cd ~
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python
pip install -e .
```

**C. Unitree RL Gym (Física do G1 e Scripts do MuJoCo)**
```bash
cd ~
git clone https://github.com/unitreerobotics/unitree_rl_gym.git
cd unitree_rl_gym
pip install -e .
```

## 4. Instalar o Simulador Visual
O MuJoCo é a única biblioteca extra necessária para gerar o mundo 3D (em principio ja foi instalado no passo acima)
```bash
pip install mujoco
```

## 5. Iniciar a Simulação
Navega para a pasta principal e lança o robô. Ele vai carregar automaticamente o cérebro `motion.pt` oficial da Unitree.
```bash
cd ~/unitree_rl_gym
python deploy/deploy_mujoco/deploy_mujoco.py g1.yaml
```
