import zmq  
import time  
  
ctx = zmq.Context()  
sock = ctx.socket(zmq.PUB)  
sock.bind("tcp://*:5556")  
  
while True:  
    sock.send_multipart([b"g1_audio", b"12345"])  
    print("a enviar...")  
    time.sleep(1)
