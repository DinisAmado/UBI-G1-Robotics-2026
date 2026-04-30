import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient


# ==============================================================================
# CONFIGURAÇÃO DOS LEDS
# ==============================================================================

# Interface de rede usada no exemplo do professor.
# Se no robô for outra, muda aqui.
UNITREE_NET_IFACE = "enp117s0"

# Cores RGB
LED_A_OUVIR = (0, 0, 255)       # azul
LED_A_FALAR = (0, 255, 0)       # verde
LED_CANCELADO = (255, 0, 0)     # vermelho
LED_OFF = (0, 0, 0)             # desligado


# ==============================================================================
# CONTROLADOR DOS LEDS
# ==============================================================================

class LedController:
    """
    Controlador simples dos LEDs do Unitree G1.

    Este ficheiro serve apenas para controlar os LEDs.
    Não tem Whisper, não tem DDS, não tem Ollama e não tem lógica HRI.

    Estados:
    - ouvir()     -> azul
    - falar()     -> verde
    - cancelar()  -> vermelho
    - desligar()  -> LEDs apagados
    """

    def __init__(self, interface_rede: str = UNITREE_NET_IFACE):
        self.interface_rede = interface_rede
        self.audio_client = None
        self.disponivel = False

        try:
            print("[LEDS] A inicializar LEDs...")

            # Inicialização da comunicação com o robô
            ChannelFactoryInitialize(0, self.interface_rede)

            # Cliente de áudio do G1, usado também para controlar LEDs
            self.audio_client = AudioClient()
            self.audio_client.SetTimeout(10.0)
            self.audio_client.Init()

            self.disponivel = True

            print("[LEDS] LEDs prontos.")

        except Exception as e:
            print(f"[LEDS] Erro ao inicializar LEDs: {e}")
            self.disponivel = False

    def set_color(self, r: int, g: int, b: int):
        """
        Muda a cor dos LEDs para o valor RGB indicado.
        """
        if not self.disponivel:
            print("[LEDS] Não disponível.")
            return

        try:
            self.audio_client.LedControl(int(r), int(g), int(b))
        except Exception as e:
            print(f"[LEDS] Erro ao mudar cor: {e}")

    def ouvir(self):
        """
        Azul: o robô está a ouvir.
        """
        print("[LEDS] Azul: a ouvir")
        self.set_color(*LED_A_OUVIR)

    def falar(self):
        """
        Verde: o robô está a falar.
        """
        print("[LEDS] Verde: a falar")
        self.set_color(*LED_A_FALAR)

    def cancelar(self):
        """
        Vermelho: o utilizador disse que não / cancelou.
        """
        print("[LEDS] Vermelho: cancelado")
        self.set_color(*LED_CANCELADO)

    def desligar(self):
        """
        Desliga os LEDs.
        """
        print("[LEDS] Desligado")
        self.set_color(*LED_OFF)


# ==============================================================================
# TESTE DIRETO DOS LEDS
# ==============================================================================

if __name__ == "__main__":
    leds = LedController()

    if not leds.disponivel:
        print("Não foi possível usar os LEDs.")
        exit(1)

    print("Teste 1: robô a ouvir")
    leds.ouvir()
    time.sleep(4)

    print("Teste 2: robô a falar")
    leds.falar()
    time.sleep(4)

    print("Teste 3: comando cancelado")
    leds.cancelar()
    time.sleep(4)

    print("Teste 4: desligar")
    leds.desligar()
