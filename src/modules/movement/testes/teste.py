import sys
import time
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

def run_raw_test(interface="enp117s0"):
    print(f"--- TESTE RAW G1 MOVE ---")
    print(f"A inicializar SDK na interface: {interface}")
    
    # 1. Inicialização exata como no exemplo oficial
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
        # Loop infinito a 50Hz (como o vosso módulo real)
        while True:
            # Enviamos a velocidade linear de 0.3 m/s com o continuous_move = True
            loco_client.Move(0.4, 0.0, 0.0, True)
            time.sleep(0.02)
            
    except KeyboardInterrupt:
        print("\nSinal de paragem recebido!")
        # Força o cancelamento do movimento contínuo
        loco_client.Move(0.0, 0.0, 0.0, False)
        time.sleep(0.5)
        # Relaxa os motores
        loco_client.Damp()
        print("Motores em Damp. Teste concluído em segurança.")

if __name__ == "__main__":
    # Usa enp117s0 por defeito, mas permite passar eth0 ou outra placa
    net_iface = sys.argv[1] if len(sys.argv) > 1 else "enp117s0"
    run_raw_test(net_iface)