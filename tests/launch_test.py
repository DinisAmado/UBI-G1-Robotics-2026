#!/usr/bin/env python3
"""
launch_test.py — Lança o teste isolado: Orquestrador + Movimentação + TesteMotion.

Uso:
  python tests/launch_test.py

Ctrl+C para terminar tudo.
"""

import sys
import os
import subprocess
import threading
import time
import signal
from datetime import datetime

RESET = "\033[0m"
BOLD  = "\033[1m"

SCRIPTS = [
    {"nome": "Orquestrador", "path": "tests/main_test.py",                  "cor": "\033[96m"},
    {"nome": "Movimentação", "path": "tests/main_motion.py","cor": "\033[91m"},
    {"nome": "TesteMotion",       "path": "tests/teste_motion.py",                    "cor": "\033[93m"},
]

processos = []
a_correr  = True


def prefixo(nome: str, cor: str) -> str:
    ts = datetime.now().strftime("%H:%M:%S")
    return f"{cor}[{nome:<13} {ts}]{RESET} "


def stream(proc, nome, cor):
    for linha in proc.stdout:
        if not a_correr:
            break
        print(prefixo(nome, cor) + linha.rstrip(), flush=True)


def shutdown(sig=None, frame=None):
    global a_correr
    a_correr = False
    print(f"\n{BOLD}A terminar...{RESET}")
    for proc in processos:
        if proc.poll() is None:
            proc.terminate()
    deadline = time.time() + 3.0
    for proc in processos:
        try:
            proc.wait(timeout=max(0.0, deadline - time.time()))
        except subprocess.TimeoutExpired:
            proc.kill()
    print(f"{BOLD}Terminado.{RESET}")
    sys.exit(0)


signal.signal(signal.SIGINT,  shutdown)
signal.signal(signal.SIGTERM, shutdown)

print(f"\n{BOLD}=== Teste Orquestrador + Movimentação + TesteMotion ==={RESET}\n")

for s in SCRIPTS:
    if not os.path.exists(s["path"]):
        print(f"{BOLD}[AVISO]{RESET} {s['path']} não encontrado — a saltar.")
        continue

    proc = subprocess.Popen(
        [sys.executable, s["path"]],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    processos.append(proc)
    threading.Thread(target=stream, args=(proc, s["nome"], s["cor"]), daemon=True).start()
    print(f"{s['cor']}[{s['nome']}]{RESET} a correr (PID {proc.pid})")
    time.sleep(0.5)   # pequeno delay para o orquestrador arrancar antes do testemotion

print(f"\n{BOLD}Ctrl+C para terminar tudo.{RESET}\n")

while a_correr:
    time.sleep(1.0)