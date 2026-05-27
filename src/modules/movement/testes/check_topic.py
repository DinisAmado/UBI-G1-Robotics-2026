# descobrir_topicos.py
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
import time

ChannelFactoryInitialize(0, "enp117s0")

candidatos = [
    "rt/sportmodestate",
    "rt/lf/sportmodestate",
    "rt/loco/sportmodestate",
    "rt/sportmode/state",
]

def make_cb(nome):
    def cb(msg):
        print(f"✅ TÓPICO ATIVO: {nome}")
        print(f"   pos=({msg.position[0]:.3f}, {msg.position[1]:.3f})")
        print(f"   vel=({msg.velocity[0]:.3f}, {msg.velocity[1]:.3f})")
    return cb

for t in candidatos:
    s = ChannelSubscriber(t, SportModeState_)
    s.Init(make_cb(t), 5)
    print(f"A escutar: {t}")

time.sleep(5)
print("Feito.")