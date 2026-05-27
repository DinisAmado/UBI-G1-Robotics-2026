# testar_lowstate.py
import os
os.environ["CYCLONEDDS_URI"] = """
<CycloneDDS><Domain><General>
  <NetworkInterfaceAddress>enp117s0</NetworkInterfaceAddress>
</General></Domain></CycloneDDS>
"""

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
import time

ChannelFactoryInitialize(0, "enp117s0")

dados = {"recebido": False}

def cb(msg):
    if not dados["recebido"]:
        dados["recebido"] = True
        imu = msg.imu_state
        print(f"✅ LowState_ recebido!")
        print(f"   quaternion : {list(imu.quaternion)}")
        print(f"   gyroscope  : {list(imu.gyroscope)}")
        print(f"   rpy        : {list(imu.rpy)}")

subs = []
for topico in ["rt/lf/lowstate", "rt/lowstate"]:
    s = ChannelSubscriber(topico, LowState_)
    s.Init(cb, 10)
    subs.append(s)
    print(f"A escutar: {topico}")

time.sleep(5)
print("Feito." if not dados["recebido"] else "")