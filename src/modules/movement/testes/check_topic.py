# ver_todos_topicos.py
from cyclonedds.domain import DomainParticipant
from cyclonedds.builtin import BuiltinDataReader, BuiltinTopicDcpsPublication
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
import time

ChannelFactoryInitialize(0, "enp117s0")
time.sleep(0.5)

dp = DomainParticipant(0)
rd = BuiltinDataReader(dp, BuiltinTopicDcpsPublication)
time.sleep(3)

topicos = rd.take()
print(f"\n{len(topicos)} tópico(s) encontrado(s):\n")
for t in sorted(topicos, key=lambda x: x.topic_name):
    print(f"  {t.topic_name:<45} | {t.type_name}")