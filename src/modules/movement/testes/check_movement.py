import time
from cyclonedds.domain import DomainParticipant
from cyclonedds.topic import Topic
from cyclonedds.pub import Publisher
from idl_ri import CmdVel, Header
from qos_profiles import QOS_MOTION

def simple_test():
    dp = DomainParticipant(0)
    topic = Topic(dp, "rt/motion/cmd_vel", CmdVel, qos=QOS_MOTION)
    writer = Publisher(dp).create_writer(topic)

    print("A enviar comando: 0.1 m/s para a frente...")
    for _ in range(150): # 3 segundos a 50Hz
        msg = CmdVel(header=Header(timestamp_ns=time.time_ns()), vx=0.1, vy=0.0, wz=0.0)
        writer.write(msg)
        time.sleep(0.02)
    
    print("A parar robô...")
    writer.write(CmdVel(vx=0.0, vy=0.0, wz=0.0))

if __name__ == "__main__":
    simple_test()