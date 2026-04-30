# livox_receiver.py

from __future__ import annotations

import os
import threading
import numpy as np


MOUNT = os.environ.get("LIVOX_MOUNT", "upside_down").lower()

if MOUNT not in {"normal", "upside_down"}:
    raise SystemExit("LIVOX_MOUNT must be 'normal' or 'upside_down'")


try:
    from livox2_python import Livox2 as _Livox
    _SDK2 = True
except Exception as e:
    print("[INFO] livox2_python indisponível:", e)
    print("[INFO] A tentar livox_python...")
    from livox_python import Livox as _Livox
    _SDK2 = False


class LivoxReceiver(_Livox):
    """
    Recebe point clouds do LiDAR Livox MID-360.

    Guarda sempre a frame mais recente em self.latest_xyz.
    O formato é uma matriz NumPy Nx3:
        xyz[:, 0] = x
        xyz[:, 1] = y
        xyz[:, 2] = z
    """

    def __init__(self, config_path="mid360_config.json", host_ip="192.168.123.165"):
        if _SDK2:
            super().__init__(config_path, host_ip=host_ip)
        else:
            super().__init__()

        self._lock = threading.Lock()
        self.latest_xyz = None
        self.frame_count = 0

    def handle_points(self, xyz: np.ndarray):
        """
        Callback chamado automaticamente pelo SDK do Livox.
        """

        if xyz is None or xyz.shape[0] == 0:
            return

        # Corrige montagem invertida do MID-360 no G1
        if MOUNT == "upside_down":
            xyz = xyz * np.array([1.0, -1.0, -1.0], dtype=xyz.dtype)

        # Reduz pontos se vierem demasiados
        if xyz.shape[0] > 100_000:
            step = max(1, xyz.shape[0] // 100_000)
            xyz = xyz[::step]

        with self._lock:
            self.latest_xyz = xyz.copy()
            self.frame_count += 1

    def get_latest_points(self):
        with self._lock:
            if self.latest_xyz is None:
                return None
            return self.latest_xyz.copy()

    def shutdown(self):
        try:
            super().shutdown()
        except Exception:
            pass
