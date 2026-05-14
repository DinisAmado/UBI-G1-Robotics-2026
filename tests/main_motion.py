import sys
import time
import logging

# 1. Imports EXCLUSIVOS do G1 (LocoClient)
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [HARDWARE TEST] %(message)s")

def test_hardware_direct(network_interface="enp117s0"):
    logging.info(f"A tentar ligar diretamente ao G1 na interface: {network_interface}...")
    
    try:
        # Inicializa a placa de rede
        ChannelFactoryInitialize(0, network_interface)  
        
        # Inicia o LocoClient (O "comando" do G1)
        client = LocoClient()
        client.SetTimeout(10.0)
        client.Init()
        logging.info("Ligação ao LocoClient estabelecida!")
    except Exception as e:
        logging.error(f"Erro fatal ao ligar ao SDK: {e}")
        sys.exit(1)

    logging.info("A iniciar movimento: 0.1 m/s para a frente durante 3 segundos.")
    
    try:
        # Loop de movimento de 3 segundos
        for _ in range(60):  # 60 iterações * 0.05s = 3 segundos
            # Função nativa do LocoClient para o G1
            client.Move(0.1, 0.0, 0.0) 
            time.sleep(0.05)
            
        logging.info("Tempo concluído. A travar o robô.")
        client.Move(0.0, 0.0, 0.0)
        
    except KeyboardInterrupt:
        logging.info("Interrompido manualmente! A travar...")
        client.Move(0.0, 0.0, 0.0)

if __name__ == "__main__":
    # Garante que enp117s0 é mesmo a placa onde o cabo de rede está ligado
    test_hardware_direct("enp117s0")