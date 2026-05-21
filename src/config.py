"""
config.py — Paths partilhados entre todos os módulos

Uso em qualquer main_*.py:
    import sys, os
    from config import SRC_DIR
    sys.path.insert(0, SRC_DIR)
"""

import os

# Diretório src/ — onde estão idl_ri.py, qos_profiles.py e este ficheiro
SRC_DIR = os.path.dirname(os.path.abspath(__file__))