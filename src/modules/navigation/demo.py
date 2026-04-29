"""
demo.py

1 - Em simulação:
    abrir terminal e correr:
        python3 demo.py
    noutro terminal:
        ./arranca.sh

2 - No robô real:
    correr com a interface correta:
        python3 demo.py eth0
    ou:
        python3 demo.py enp...
"""

import os
import json
import time
import sys
import math
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_

from slam_navigation import SLAMNavigation
from sensores import sensor_mujoco_json, sensor_real


# ---------------------------------------------------------------
# Configurações do mapa
# ---------------------------------------------------------------
MAP_ORIGIN_X = -5.0
MAP_ORIGIN_Y = -5.0
MAP_RESOLUTION = 0.05


def world_to_cell(world_x, world_y):
    cell_x = int((world_x - MAP_ORIGIN_X) / MAP_RESOLUTION)
    cell_y = int((world_y - MAP_ORIGIN_Y) / MAP_RESOLUTION)
    return cell_x, cell_y


def cell_to_world(cell_x, cell_y):
    world_x = MAP_ORIGIN_X + cell_x * MAP_RESOLUTION
    world_y = MAP_ORIGIN_Y + cell_y * MAP_RESOLUTION
    return world_x, world_y


def find_approach_goal(
    slam,
    table_cell,
    robot_cell,
    target_distance_m=0.30,
    tolerance_m=0.10
):
    tx, ty = table_cell
    rx, ry = robot_cell

    target_cells = int(target_distance_m / MAP_RESOLUTION)
    tolerance_cells = int(tolerance_m / MAP_RESOLUTION)

    min_dist = max(1, target_cells - tolerance_cells)
    max_dist = target_cells + tolerance_cells

    candidates = []

    for dx in range(-max_dist, max_dist + 1):
        for dy in range(-max_dist, max_dist + 1):
            gx = tx + dx
            gy = ty + dy

            if not (0 <= gx < slam.map_size and 0 <= gy < slam.map_size):
                continue

            dist_to_table = math.sqrt(dx * dx + dy * dy)

            if dist_to_table < min_dist or dist_to_table > max_dist:
                continue

            if slam.is_occupied(gx, gy):
                continue

            dist_to_robot = math.sqrt((gx - rx) ** 2 + (gy - ry) ** 2)
            score = abs(dist_to_table - target_cells) + 0.05 * dist_to_robot

            candidates.append((score, gx, gy))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])
    return candidates[0][1], candidates[0][2]


class RobotController:
    def __init__(self):
        self.current_state = None
        self.crc = CRC()

        self.pos_x = 0.0
        self.pos_y = 0.0

    def LowStateHandler(self, msg: LowState_):
        self.current_state = msg

    def run(self):
        plt.close('all')

        if len(sys.argv) < 2:
            ChannelFactoryInitialize(1, "lo")
        else:
            ChannelFactoryInitialize(0, sys.argv[1])

        low_state_sub = ChannelSubscriber("rt/lowstate", LowState_)
        low_state_sub.Init(self.LowStateHandler, 10)

        low_cmd_pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        low_cmd_pub.Init()

        print("Waiting for lowstate...")

        timeout = 5
        start_time = time.time()

        while self.current_state is None:
            if time.time() - start_time > timeout:
                print("Aviso: não recebi lowstate. Vou continuar sem bloquear.")
                break
            time.sleep(0.01)

        print("Continuing...")

        # ---------------- SLAM e Navegação ----------------
        slam = SLAMNavigation(
            map_size=200,
            resolution=MAP_RESOLUTION,
            num_rays=144,
            max_range=100
        )

        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)

        # ---------------------------------------------------
        # Objetivo inicial
        # No robô real começa sem objetivo conhecido.
        # A perceção/interação deverá preencher isto depois.
        # ---------------------------------------------------
        table_world = None
        table_cell = None
        current_goal_cell = None
        current_path = []

        # Posição inicial
        init_cell_x, init_cell_y = world_to_cell(self.pos_x, self.pos_y)
        slam.update_pose(init_cell_x, init_cell_y, 0.0)

        # ---------------- Visualização ----------------
        custom_cmap = ListedColormap(['white', 'lightgrey', 'black'])

        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 8))

        img = ax.imshow(
            slam.get_visualization_grid(),
            cmap=custom_cmap,
            vmin=0,
            vmax=2,
            origin="lower"
        )

        leg_free = mpatches.Patch(color='white', label='Livre', ec='black')
        leg_unk = mpatches.Patch(color='lightgrey', label='Desconhecido')
        leg_obs = mpatches.Patch(color='black', label='Obstáculo')

        robot_dot, = ax.plot([], [], "ro", markersize=8, label="Robô G1")
        path_line, = ax.plot([], [], "g-", linewidth=2, label="Caminho A*")
        goal_dot, = ax.plot([], [], "bo", markersize=6, label="Objetivo")

        ax.set_title("SLAM e Navegação")
        ax.legend(
            handles=[leg_free, leg_unk, leg_obs, path_line, robot_dot, goal_dot],
            loc='upper right',
            fontsize='small'
        )

        plt.show(block=False)
        plt.pause(0.1)

        viz_counter = 0
        last_map_save_time = 0.0
        last_saved_path = None
        MAP_SAVE_INTERVAL = 2.0

        while True:
            step_start = time.perf_counter()

            yaw = 0.0

            if self.current_state and hasattr(self.current_state, 'imu_state'):
                yaw = self.current_state.imu_state.rpy[2]

            # ---------------------------------------------------
            # 1. Tentar ler dados do MuJoCo
            # ---------------------------------------------------
            mujoco_ok, mx, my, myaw, rays = sensor_mujoco_json()

            if mujoco_ok:
                self.pos_x = mx
                self.pos_y = my
                yaw = myaw
                slam.update_from_mujoco_rays(rays)

            else:
                # ---------------------------------------------------
                # 2. Robô real: tentar usar LiDAR real
                # ---------------------------------------------------
                curr_cell_x, curr_cell_y = world_to_cell(self.pos_x, self.pos_y)

                obs, free = sensor_real(
                    self.current_state,
                    curr_cell_x,
                    curr_cell_y,
                    yaw,
                    map_size=slam.map_size,
                    resolution=MAP_RESOLUTION,
                    max_range_meters=5.0
                )

                for pt in obs:
                    slam._update_cell(pt[0], pt[1], True)

                for pt in free:
                    slam._update_cell(pt[0], pt[1], False)

            curr_cell_x, curr_cell_y = world_to_cell(self.pos_x, self.pos_y)
            slam.update_pose(curr_cell_x, curr_cell_y, yaw)

            # ---------------------------------------------------
            # Visualização e planeamento
            # ---------------------------------------------------
            viz_counter += 1

            if viz_counter >= 20:
                viz_counter = 0

                path_needs_replan = not current_path or not slam.is_path_valid(current_path)

                if path_needs_replan:

                    if table_cell is not None:
                        current_goal_cell = find_approach_goal(
                            slam=slam,
                            table_cell=table_cell,
                            robot_cell=(curr_cell_x, curr_cell_y),
                            target_distance_m=0.30,
                            tolerance_m=0.10
                        )

                        if current_goal_cell is not None:
                            current_path = slam.plan_path(current_goal_cell)
                        else:
                            current_path = []

                    else:
                        current_goal_cell = None
                        current_path = []

                img.set_data(slam.get_visualization_grid())

                if current_path:
                    path_y = [p[1] for p in current_path]
                    path_x = [p[0] for p in current_path]
                    path_line.set_data(path_y, path_x)
                else:
                    path_line.set_data([], [])

                robot_dot.set_data([curr_cell_y], [curr_cell_x])

                if current_goal_cell is not None:
                    goal_dot.set_data([current_goal_cell[1]], [current_goal_cell[0]])
                else:
                    goal_dot.set_data([], [])

                ax.set_xlim(curr_cell_y - 100, curr_cell_y + 100)
                ax.set_ylim(curr_cell_x - 100, curr_cell_x + 100)

                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.001)

                now = time.time()

                if now - last_map_save_time >= MAP_SAVE_INTERVAL:
                    np.save(
                        os.path.join(output_dir, "occupancy_grid.npy"),
                        slam.get_probability_grid()
                    )

                    plt.imsave(
                        os.path.join(output_dir, "map_preview.png"),
                        slam.get_visualization_grid(),
                        cmap=custom_cmap,
                        vmin=0,
                        vmax=2
                    )

                    last_map_save_time = now

                if current_path and current_path != last_saved_path:
                    path_data = {
                        "goal_cell": {
                            "x": int(current_goal_cell[0]),
                            "y": int(current_goal_cell[1])
                        },
                        "goal_world": {
                            "x": float(cell_to_world(current_goal_cell[0], current_goal_cell[1])[0]),
                            "y": float(cell_to_world(current_goal_cell[0], current_goal_cell[1])[1])
                        },
                        "path": [
                            {
                                "cell_x": int(p[0]),
                                "cell_y": int(p[1]),
                                "world_x": float(cell_to_world(p[0], p[1])[0]),
                                "world_y": float(cell_to_world(p[0], p[1])[1])
                            }
                            for p in current_path
                        ]
                    }

                    with open(os.path.join(output_dir, "latest_path.json"), "w") as f:
                        json.dump(path_data, f, indent=2)

                    last_saved_path = list(current_path)

            # ---------------------------------------------------
            # Segurança: por agora não enviar movimento real.
            # Mantemos LowCmd vazio como heartbeat/comunicação.
            # Para teste totalmente passivo, comenta estas 3 linhas.
            # ---------------------------------------------------
            cmd = unitree_hg_msg_dds__LowCmd_()
            cmd.crc = self.crc.Crc(cmd)
            low_cmd_pub.Write(cmd)

            elapsed = time.perf_counter() - step_start

            if 0.002 - elapsed > 0:
                time.sleep(0.002 - elapsed)


if __name__ == '__main__':
    RobotController().run()
```
