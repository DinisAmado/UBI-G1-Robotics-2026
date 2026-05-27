"""
odom_walk_test.py — Script de teste de odometria para o G1
Corre em paralelo com o main_config_test.py (módulo motion já em execução).

Fluxo:
  1. Liga ao mesmo barramento DDS do módulo motion
  2. Publica CmdVel → robot anda durante DURACAO_SEGUNDOS
  3. Lê rt/motion/odometry e mostra posição + orientação no terminal
  4. Ao fim do tempo, para o robot e imprime resumo

Uso:
    # Terminal 1 — módulo motion a correr:
    python3 main_config_test.py enp117s0

    # Terminal 2 — este script:
    python3 odom_walk_test.py
"""

import sys
import os
import time
import math
import signal

# ── Ajusta o path para encontrar idl_ri e qos_profiles ──────────────────────
pasta_atual = os.path.dirname(os.path.abspath(__file__))
pasta_src   = os.path.abspath(os.path.join(pasta_atual, '../..'))
if pasta_src not in sys.path:
    sys.path.insert(0, pasta_src)

from cyclonedds.domain import DomainParticipant
from cyclonedds.topic import Topic
from cyclonedds.pub import Publisher, DataWriter
from cyclonedds.sub import Subscriber, DataReader

from idl_ri import (
    Header, CmdVel, OdometryMsg
)
from qos_profiles import QOS_MOTION, QOS_ODOMETRY

# ──────────────────────────────────────────────────────────────────────────────
# PARÂMETROS — edita aqui
# ──────────────────────────────────────────────────────────────────────────────
DURACAO_SEGUNDOS = 5.0      # quanto tempo o robot anda
VX               = 0.3      # velocidade frontal (m/s)  — positivo = frente
VY               = 0.0      # velocidade lateral (m/s)
WZ               = 0.0      # rotação (rad/s)
DOMINIO_DDS      = 0        # mesmo domínio do módulo motion
# ──────────────────────────────────────────────────────────────────────────────

# Cores ANSI
R = "\033[0m";  BOLD = "\033[1m";  GREEN = "\033[92m"
YELLOW = "\033[93m";  CYAN = "\033[96m";  RED = "\033[91m"
GREY = "\033[90m";  WHITE = "\033[97m"

SEQ = 0
def get_header(frame="base_link") -> Header:
    global SEQ
    SEQ += 1
    return Header(timestamp_ns=time.time_ns(), frame_id=frame, seq=SEQ)

class OdomWalkTest:
    def __init__(self):
        self.running       = True
        self.ultima_odom   = None
        self.historico     = []          # lista de (t, px, py, yaw)
        self.fase          = "espera"    # espera → a_andar → parado → fim

        signal.signal(signal.SIGINT, self._shutdown)

        print(f"\n{CYAN}▶ A ligar ao DDS (domínio {DOMINIO_DDS})...{R}")
        self.dp  = DomainParticipant(DOMINIO_DDS)
        pub      = Publisher(self.dp)
        sub      = Subscriber(self.dp)

        self.w_cmd  = DataWriter(pub,  Topic(self.dp, "rt/motion/cmd_vel",    CmdVel,      qos=QOS_MOTION))
        self.r_odom = DataReader(sub,  Topic(self.dp, "rt/motion/odometry",   OdometryMsg, qos=QOS_ODOMETRY))

        print(f"{GREEN}✅ DDS pronto. Publisher e Reader configurados.{R}")

    def _shutdown(self, *_):
        self.running = False

    def _yaw_de_quaternion(self, q) -> float:
        """Extrai yaw de um Quaternion (x, y, z, w)."""
        return math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

    def _ler_odom(self):
        samples = self.r_odom.take()
        if samples:
            self.ultima_odom = samples[-1]

    def _publicar_cmd(self, vx, vy, wz):
        cmd = CmdVel(header=get_header(), vx=vx, vy=vy, wz=wz)
        self.w_cmd.write(cmd)

    def _render(self, elapsed_total, tempo_restante):
        o = self.ultima_odom

        if o is None:
            print(f"\r{YELLOW}⏳ A aguardar odometria de rt/motion/odometry...{R}  ", end="", flush=True)
            return

        px   = o.pose.position.x
        py   = o.pose.position.y
        pz   = o.pose.position.z
        yaw  = self._yaw_de_quaternion(o.pose.orientation)
        yaw_deg = math.degrees(yaw)
        dist = math.sqrt(px**2 + py**2)

        # cor do estado
        if self.fase == "a_andar":
            estado_cor  = GREEN
            estado_txt  = f"A ANDAR  [{tempo_restante:.1f}s restantes]"
        elif self.fase == "parado":
            estado_cor  = CYAN
            estado_txt  = "PARADO ✓"
        else:
            estado_cor  = YELLOW
            estado_txt  = "ESPERA"

        linhas = [
            f"{BOLD}{WHITE}┌─────────────────────────────────────────────┐{R}",
            f"{BOLD}{WHITE}│  ODOMETRY MONITOR          t={elapsed_total:6.1f}s         │{R}",
            f"{BOLD}{WHITE}├─────────────────────────────────────────────┤{R}",
            f"{WHITE}│ {CYAN}POSIÇÃO   {R}  x={GREEN}{px:+8.4f}{R}m   y={GREEN}{py:+8.4f}{R}m    {WHITE}│{R}",
            f"{WHITE}│            z={GREEN}{pz:+8.4f}{R}m   dist={YELLOW}{dist:6.3f}{R}m       {WHITE}│{R}",
            f"{WHITE}│ {CYAN}ORIENTAÇÃO{R}  yaw={GREEN}{yaw_deg:+8.3f}{R}°  ({yaw:+.4f} rad)   {WHITE}│{R}",
            f"{WHITE}│ {CYAN}VELOCIDADE{R}  vx={YELLOW}{o.vx:+7.3f}{R}  vy={YELLOW}{o.vy:+7.3f}{R}  wz={YELLOW}{o.wz:+6.3f}{R} {WHITE}│{R}",
            f"{WHITE}│ {CYAN}ESTADO    {R}  {estado_cor}{BOLD}{estado_txt:<37}{R}{WHITE}│{R}",
            f"{BOLD}{WHITE}└─────────────────────────────────────────────┘{R}",
        ]

        N = len(linhas)
        if hasattr(self, '_rendered_once') and self._rendered_once:
            print(f"\033[{N}A", end="")   # sobe N linhas

        for l in linhas:
            print(l)
        self._rendered_once = True

    def _resumo(self):
        if not self.historico:
            print(f"\n{RED}Sem dados de odometria para resumo.{R}")
            return

        t0, px0, py0, yaw0 = self.historico[0]
        tf, pxf, pyf, yawf = self.historico[-1]

        delta_x   = pxf - px0
        delta_y   = pyf - py0
        delta_yaw = math.degrees(yawf - yaw0)
        distancia = math.sqrt(delta_x**2 + delta_y**2)

        print(f"\n{BOLD}{WHITE}━━━━━━━━━━━━━━━ RESUMO DO TESTE ━━━━━━━━━━━━━━━{R}")
        print(f"  Duração total  : {BOLD}{tf - t0:.2f}s{R}")
        print(f"  Δ posição X    : {GREEN}{delta_x:+.4f}{R} m")
        print(f"  Δ posição Y    : {GREEN}{delta_y:+.4f}{R} m")
        print(f"  Distância total: {YELLOW}{distancia:.4f}{R} m")
        print(f"  Δ orientação   : {GREEN}{delta_yaw:+.2f}{R}°")
        print(f"{BOLD}{WHITE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{R}\n")

    def run(self):
        print(f"\n{BOLD}Configuração:{R}")
        print(f"  Velocidade : vx={VX}  vy={VY}  wz={WZ}")
        print(f"  Duração    : {DURACAO_SEGUNDOS}s")
        print(f"\n{YELLOW}Aguarda 2s para o módulo motion estabilizar...{R}")
        time.sleep(2.0)

        t_inicio     = time.time()
        t_inicio_cmd = None
        self._rendered_once = False

        while self.running:
            agora   = time.time()
            elapsed = agora - t_inicio

            self._ler_odom()

            # — Máquina de estados simples —
            if self.fase == "espera":
                if self.ultima_odom is not None:
                    print(f"\n{GREEN}✅ Odometria recebida! A iniciar movimento...{R}\n")
                    t_inicio_cmd = agora
                    self.fase = "a_andar"

            elif self.fase == "a_andar":
                tempo_cmd     = agora - t_inicio_cmd
                tempo_restante = max(0.0, DURACAO_SEGUNDOS - tempo_cmd)
                self._publicar_cmd(VX, VY, WZ)

                # guarda histórico
                if self.ultima_odom:
                    o = self.ultima_odom
                    yaw = self._yaw_de_quaternion(o.pose.orientation)
                    self.historico.append((agora, o.pose.position.x, o.pose.position.y, yaw))

                self._render(elapsed, tempo_restante)

                if tempo_cmd >= DURACAO_SEGUNDOS:
                    self._publicar_cmd(0.0, 0.0, 0.0)   # para o robot
                    self.fase = "parado"
                    print(f"\n\n{GREEN}✅ Tempo atingido — robot parado.{R}")

            elif self.fase == "parado":
                self._publicar_cmd(0.0, 0.0, 0.0)       # mantém parado (safety)
                self._render(elapsed, 0.0)
                self.fase = "fim"

            elif self.fase == "fim":
                self._resumo()
                break

            time.sleep(0.02)   # 50Hz


if __name__ == "__main__":
    OdomWalkTest().run()