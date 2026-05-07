import sys
import subprocess
import threading
import time
import os
import signal
from datetime import datetime

# ─── Configuração dos módulos ─────────────────────────────────────────────────

MODULES = {
    1: {"nome": "Orquestrador",   "path": "src/main.py", "cor": "\033[96m"},   # ciano
    2: {"nome": "SLAM+Navegação", "path": "src/navigation/main.py", "cor": "\033[92m"},   # verde
    3: {"nome": "Visão",          "path": "src/vision/main.py", "cor": "\033[93m"},   # amarelo
    4: {"nome": "Grasping",       "path": "src/grasping/main.py", "cor": "\033[95m"},   # magenta
    5: {"nome": "HMI",            "path": "src/hmi/main.py", "cor": "\033[94m"},   # azul
    6: {"nome": "Movimentação",   "path": "src/movement/main.py", "cor": "\033[91m"},   # vermelho
}

RESET = "\033[0m"
BOLD  = "\033[1m"

# ─── Estado global ────────────────────────────────────────────────────────────

processos: dict[int, subprocess.Popen] = {}
a_correr   = True


# ─── Funções de output ────────────────────────────────────────────────────────

def prefixo(grupo: int) -> str:
    m  = MODULES[grupo]
    ts = datetime.now().strftime("%H:%M:%S")
    return f"{m['cor']}[G{grupo} {m['nome']:<14} {ts}]{RESET} "


def stream_output(proc: subprocess.Popen, grupo: int) -> None:
    """Lê stdout do processo linha a linha e imprime com prefixo colorido."""
    for linha in proc.stdout:
        if not a_correr:
            break
        print(prefixo(grupo) + linha.rstrip())


# ─── Lançamento ───────────────────────────────────────────────────────────────

def arrancar_modulo(grupo: int) -> subprocess.Popen | None:
    m    = MODULES[grupo]
    path = m["path"]

    if not os.path.exists(path):
        print(f"{BOLD}[AVISO]{RESET} {path} não encontrado — grupo {grupo} ignorado.")
        return None

    print(f"{m['cor']}{BOLD}[G{grupo}] A arrancar {m['nome']}...{RESET}")

    proc = subprocess.Popen(
        [sys.executable, path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,   # mistura stderr com stdout para simplificar
        text=True,
        bufsize=1,                  # line-buffered
    )

    # Thread dedicada para fazer stream do output sem bloquear
    t = threading.Thread(target=stream_output, args=(proc, grupo), daemon=True)
    t.start()

    return proc


# ─── Shutdown limpo ───────────────────────────────────────────────────────────

def shutdown(sig=None, frame=None) -> None:
    global a_correr
    a_correr = False

    print(f"\n{BOLD}[LAUNCH] A terminar todos os módulos...{RESET}")

    for grupo, proc in processos.items():
        if proc.poll() is None:
            nome = MODULES[grupo]["nome"]
            print(f"  → A terminar G{grupo} {nome} (PID {proc.pid})...")
            proc.terminate()

    # Dar 3 segundos para terminarem graciosamente
    deadline = time.time() + 3.0
    for proc in processos.values():
        restante = max(0.0, deadline - time.time())
        try:
            proc.wait(timeout=restante)
        except subprocess.TimeoutExpired:
            proc.kill()

    print(f"{BOLD}[LAUNCH] Sistema encerrado.{RESET}")
    sys.exit(0)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    signal.signal(signal.SIGINT,  shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Determinar quais grupos arrancar a partir dos argumentos
    args    = sys.argv[1:]
    excluir = {int(a[2:]) for a in args if a.startswith("--no-")}
    incluir = {int(a) for a in args if a.isdigit()}

    if incluir:
        grupos = sorted(incluir)
    else:
        grupos = sorted(set(MODULES.keys()) - excluir)

    print(f"\n{BOLD}{'=' * 55}")
    print(f"  Sistema RI — Unitree G1  |  {len(grupos)} módulo(s)")
    print(f"{'=' * 55}{RESET}\n")

    # Arrancar cada módulo com pequeno delay para não inundar os logs
    for g in grupos:
        proc = arrancar_modulo(g)
        if proc:
            processos[g] = proc
        time.sleep(0.3)

    if not processos:
        print("[LAUNCH] Nenhum módulo encontrado. Verifica a estrutura de pastas.")
        sys.exit(1)

    print(f"\n{BOLD}[LAUNCH] {len(processos)} módulo(s) a correr. Ctrl+C para terminar tudo.{RESET}\n")

    # Monitorizar processos — reiniciar se crasharem inesperadamente
    while a_correr:
        for grupo, proc in list(processos.items()):
            ret = proc.poll()
            if ret is not None and a_correr:
                nome = MODULES[grupo]["nome"]
                print(f"\n{BOLD}[LAUNCH] G{grupo} {nome} terminou com código {ret}.{RESET}")

                # Orquestrador não reinicia sozinho — é o núcleo do sistema
                if grupo != 1 and ret != 0:
                    print(f"[LAUNCH] A reiniciar G{grupo} {nome} em 2 s...")
                    time.sleep(2.0)
                    novo = arrancar_modulo(grupo)
                    if novo:
                        processos[grupo] = novo
                else:
                    del processos[grupo]

        if not processos:
            print("[LAUNCH] Todos os módulos terminaram.")
            break

        time.sleep(1.0)


if __name__ == "__main__":
    main()


"""

Uso:
  python launch.py            # arranca todos
  python launch.py 1 2 3      # arranca só os grupos indicados
  python launch.py --no-3     # arranca todos menos a Visão (para debug)
  
  """