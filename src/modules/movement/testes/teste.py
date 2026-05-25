import sys
import time
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

def run_raw_test(interface="enp117s0"):
    print("--- TESTE RAW G1 MOVE ---")
    print(f"A inicializar SDK na interface: {interface}")
    
    ChannelFactoryInitialize(0, interface)
    loco_client = LocoClient()
    loco_client.SetTimeout(10.0)
    loco_client.Init()
    
    print("\nConectado ao hardware com sucesso.")
    print("ATENÇÃO: O robô vai começar a andar para a frente a 0.3 m/s.")
    print("Certifica-te que ele JÁ ESTÁ EM HIGHSTAND!")
    print("O teste arranca em 5 segundos...")
    time.sleep(5)
    
    print("\nA INICIAR MOVIMENTO... (Pressiona Ctrl+C para travar)")
    
    try:
        while True:
            loco_client.Move(0.4, 0.0, 0.0)
            time.sleep(0.02)
            
    except KeyboardInterrupt:
        print("\nSinal de paragem recebido!")
        loco_client.Move(0.0, 0.0, 0.0)
        print("Velocidade a zero. Robô imobilizado mantendo os motores ativos.")

if __name__ == "__main__":
    net_iface = sys.argv[1] if len(sys.argv) > 1 else "enp117s0"
    run_raw_test(net_iface)