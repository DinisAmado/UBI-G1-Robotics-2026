import zmq

AUDIO_TOPIC = b"g1_audio"
G1_IP = "192.168.123.164"
PORT = 5556

ctx = zmq.Context()
sock = ctx.socket(zmq.SUB)
sock.connect(f"tcp://{G1_IP}:{PORT}")
sock.setsockopt(zmq.SUBSCRIBE, AUDIO_TOPIC)

print("À espera de áudio...")

while True:
    parts = sock.recv_multipart()
    print("Recebi mensagem!", len(parts))
