import time
import logging
from cyclonedds.domain import DomainParticipant
from cyclonedds.topic import Topic
from cyclonedds.pub import Publisher, DataWriter
from idl_ri import Header, CmdVel
from qos_profiles import QOS_MOTION

logging.basicConfig(level=logging.INFO, format="%(asctime)s [TEST] %(message)s")

def run_test():
    dp = DomainParticipant(0)
    pub = Publisher(dp)
    writer = DataWriter(pub, Topic(dp, "rt/motion/cmd_vel", CmdVel, qos=QOS_MOTION))
    
    seq = 0
    log = logging.getLogger("test")
    
    print("Iniciando teste de movimento: 0.2 m/s durante 10 segundos.")
    start_time = time.time()
    
    try:
        while time.time() - start_time < 10:
            msg = CmdVel(
                header=Header(timestamp_ns=time.time_ns(), frame_id="test", seq=seq),
                vx=0.2, vy=0.0, wz=0.0
            )
            writer.write(msg)
            seq += 1
            time.sleep(0.05) # 20Hz é suficiente para este teste
            
        # Parar no fim
        writer.write(CmdVel(header=Header(timestamp_ns=time.time_ns(), seq=seq+1), vx=0.0, vy=0.0, wz=0.0))
        print("Teste concluído. Comando de paragem enviado.")
        
    except KeyboardInterrupt:
        writer.write(CmdVel(header=Header(timestamp_ns=time.time_ns(), seq=seq+1), vx=0.0, vy=0.0, wz=0.0))

if __name__ == "__main__":
    run_test()