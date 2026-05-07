#unitree_mujoco.py

import time
import math
import json
import numpy as np
import mujoco
import mujoco.viewer
from threading import Thread
import threading
import os
import tempfile

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py_bridge import UnitreeSdk2Bridge, ElasticBand
import config

# Ficheiro partilhado entre processos
RAY_DATA_FILE = "/tmp/mujoco_rays.json"

MAP_ORIGIN_X = -40.0
MAP_ORIGIN_Y = -40.0
MAP_RESOLUTION = 0.1

locker = threading.Lock()

mj_model = mujoco.MjModel.from_xml_path(config.ROBOT_SCENE)
mj_data  = mujoco.MjData(mj_model)

# Identificar o ID do tronco para o sensor o ignorar
try:
    TORSO_ID = mj_model.body("torso_link").id
except:
    print("Aviso: 'torso_link' não encontrado no modelo XML.")
    TORSO_ID = -1

if config.ENABLE_ELASTIC_BAND:
    elastic_band = ElasticBand()
    if config.ROBOT in ("h1", "g1"):
        band_attached_link = mj_model.body("torso_link").id
    else:
        band_attached_link = mj_model.body("base_link").id
    viewer = mujoco.viewer.launch_passive(
        mj_model, mj_data, key_callback=elastic_band.MujuocoKeyCallback
    )
else:
    viewer = mujoco.viewer.launch_passive(mj_model, mj_data)

mj_model.opt.timestep = config.SIMULATE_DT
time.sleep(0.2)

yaw_angle = 0.0
YAW_SPEED = 0.01

def SimulationThread():
    global mj_data, mj_model, yaw_angle, TORSO_ID

    ChannelFactoryInitialize(config.DOMAIN_ID, config.INTERFACE)
    unitree = UnitreeSdk2Bridge(mj_model, mj_data)

    if config.USE_JOYSTICK:
        unitree.SetupJoystick(device_id=0, js_type=config.JOYSTICK_TYPE)
    if config.PRINT_SCENE_INFORMATION:
        unitree.PrintSceneInformation()

    while viewer.is_running():
        step_start = time.perf_counter()
        locker.acquire()

        if config.ENABLE_ELASTIC_BAND and elastic_band.enable:
            mj_data.xfrc_applied[band_attached_link, :3] = elastic_band.Advance(
                mj_data.qpos[:3], mj_data.qvel[:3]
            )

        # Congelar base do robô em [0, 0, 0.8]
        mj_data.qpos[0] = 0.0
        mj_data.qpos[1] = 0.0
        mj_data.qpos[2] = 0.8
        mj_data.qpos[7:] = 0.0
        mj_data.qvel[:]  = 0.0

        # Rotação contínua para explorar o ambiente
        yaw_angle += YAW_SPEED
        qw = math.cos(yaw_angle / 2)
        qz = math.sin(yaw_angle / 2)
        mj_data.qpos[3] = qw
        mj_data.qpos[4] = 0.0
        mj_data.qpos[5] = 0.0
        mj_data.qpos[6] = qz

        # ---- Raycasting real no MuJoCo (Multi-camada) ----
        robot_pos = mj_data.qpos[:3].copy()
        robot_pos[2] = 1.2  # altura do sensor
        num_yaw_steps = 144  # CORRIGIDO: nome da variável sincronizado
        max_range_m = 40.0
        vertical_pitches = [0.25, 0.0, -0.20, -0.45]

        ray_hits = []

        for i in range(num_yaw_steps):
            h_angle = yaw_angle + i * (2 * math.pi / num_yaw_steps)

            for v_pitch in vertical_pitches:
                dir_x = math.cos(h_angle) * math.cos(v_pitch)
                dir_y = math.sin(h_angle) * math.cos(v_pitch)
                dir_z = math.sin(v_pitch)
                direction = np.array([dir_x, dir_y, dir_z])

                geom_id = np.array([-1], dtype=np.int32)
                dist = mujoco.mj_ray(mj_model, mj_data, robot_pos, direction, None, 1, TORSO_ID, geom_id)

                # Filtro para ignorar o chão distante
                dist_teorica_chao = abs(robot_pos[2] / math.sin(v_pitch)) if v_pitch < 0 else 1000.0
                if v_pitch < 0 and dist > (dist_teorica_chao - 0.1):
                    continue

                if 0 <= dist <= max_range_m:
                    ray_hits.append([h_angle, dist / MAP_RESOLUTION])

        robot_cell_x = int((robot_pos[0] - MAP_ORIGIN_X) / MAP_RESOLUTION)
        robot_cell_y = int((robot_pos[1] - MAP_ORIGIN_Y) / MAP_RESOLUTION)

        try:
            with tempfile.NamedTemporaryFile('w', dir='/tmp', delete=False, suffix='.json') as tf:
                json.dump({
                    "yaw":           yaw_angle,
                    "robot_world_x": float(robot_pos[0]),
                    "robot_world_y": float(robot_pos[1]),
                    "robot_cell_x":  robot_cell_x,
                    "robot_cell_y":  robot_cell_y,
                    "rays":          ray_hits
                }, tf)
                tmp_name = tf.name
            os.replace(tmp_name, RAY_DATA_FILE)
        except Exception:
            pass

        mujoco.mj_step(mj_model, mj_data)
        locker.release()
        time.sleep(max(0, mj_model.opt.timestep - (time.perf_counter() - step_start)))

def PhysicsViewerThread():
    while viewer.is_running():
        locker.acquire()
        viewer.sync()
        locker.release()
        time.sleep(config.VIEWER_DT)

if __name__ == "__main__":
    Thread(target=PhysicsViewerThread).start()
    Thread(target=SimulationThread).start()
