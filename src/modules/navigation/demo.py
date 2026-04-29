#demo.py

"""
exemplo básico de como controlar o robô:
1-abrir um terminal e correr este script com
python3 demo.py
2-noutro terminal correr ./arranca.sh
"""
import os
import json
import time
import sys
import math
import json
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_

from slam_navigation import SLAMNavigation
from sensores import sensor_sim, sensor_real

# ---------------------------------------------------------------
# Configurações do Mapa
# ---------------------------------------------------------------
MAP_ORIGIN_X = -40.0  # Centrado para permitir 40m em todas as direções
MAP_ORIGIN_Y = -40.0
MAP_RESOLUTION = 0.1

def world_to_cell(world_x, world_y):
    """Converte coordenadas do mundo (metros) para células do grid."""
    cell_x = int((world_x - MAP_ORIGIN_X) / MAP_RESOLUTION)
    cell_y = int((world_y - MAP_ORIGIN_Y) / MAP_RESOLUTION)
    return cell_x, cell_y


def cell_to_world(cell_x, cell_y):
    """Converte células do grid para coordenadas do mundo em metros."""
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
    """
    Escolhe automaticamente uma célula livre perto da mesa.

    A ideia é:
    - a mesa está no centro table_cell;
    - o robô deve parar a aproximadamente target_distance_m da mesa;
    - escolhemos uma célula livre nesse anel;
    - preferimos células próximas da distância desejada e próximas do robô.
    """
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


def save_outputs(output_dir, slam, path, goal_cell, custom_cmap):
    os.makedirs(output_dir, exist_ok=True)

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

    path_data = {
        "goal_cell": None if goal_cell is None else {
            "x": int(goal_cell[0]),
            "y": int(goal_cell[1])
        },
        "path": [
            {"x": int(p[0]), "y": int(p[1])}
            for p in path
        ]
    }

    with open(os.path.join(output_dir, "latest_path.json"), "w") as f:
        json.dump(path_data, f, indent=2)


class RobotController:
    def __init__(self):
        self.current_state = None
        self.crc = CRC()
        # Posição real do robô em metros (lida do MuJoCo via JSON)
        self.pos_x = 0.0
        self.pos_y = 0.0

    def LowStateHandler(self, msg: LowState_):
        self.current_state = msg

    def run(self):
        # Fecha janelas de visualização anteriores
        plt.close('all')

        # Inicializar comunicação
        if len(sys.argv) < 2:
            ChannelFactoryInitialize(1, "lo")  # Simulação
        else:
            ChannelFactoryInitialize(0, sys.argv[1])  # Robô real

        low_state_sub = ChannelSubscriber("rt/lowstate", LowState_)
        low_state_sub.Init(self.LowStateHandler, 10)

        low_cmd_pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        low_cmd_pub.Init()

        print("Waiting for simulator...")
        while self.current_state is None:
            time.sleep(0.01)
        print("Connected")

        # ---------------- SLAM e Navegação ----------------
        slam = SLAMNavigation(map_size=800, resolution=0.1, num_rays=144, max_range=400)

        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)

        # Configurar Objetivo: Mesa em [2.0, 0.0] no mundo MuJoCo
        table_world = [2.0, 0.0]  # Centro da mesa no MuJoCo
        table_cell = world_to_cell(table_world[0], table_world[1])
        goal_cell = None

        # Posição inicial
        self.pos_x = 0.0
        self.pos_y = 0.0
        init_cell_x, init_cell_y = world_to_cell(self.pos_x, self.pos_y)
        slam.update_pose(init_cell_x, init_cell_y, 0.0)

        # ---------------- Visualização Única ----------------
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


        # Criar elementos da legenda
        leg_free = mpatches.Patch(color='white', label='Livre', ec='black')
        leg_unk  = mpatches.Patch(color='lightgrey', label='Desconhecido')
        leg_obs  = mpatches.Patch(color='black', label='Obstáculo')


        robot_dot, = ax.plot([], [], "ro", markersize=8, label="Robô G1")
        path_line, = ax.plot([], [], "g-", linewidth=2, label="Caminho A*")
        goal_dot, = ax.plot([], [], "bo", markersize=6, label="Objetivo automático")

        ax.set_title("SLAM e Navegação")
        ax.legend(
            handles=[leg_free, leg_unk, leg_obs, path_line, robot_dot, goal_dot],
            loc='upper right',
            fontsize='small'
        )
        plt.show(block=False)

        # FORÇAR ABERTURA DA JANELA
        plt.pause(0.1)

        viz_counter = 0
        last_map_save_time = 0.0
        last_path_save_time = 0.0
        last_saved_path = None

        MAP_SAVE_INTERVAL = 2.0

        current_path = []
        current_goal_cell = None
        while True:
            step_start = time.perf_counter()
            # Obter orientação (Yaw)
            if self.current_state and hasattr(self.current_state, 'imu_state'):
                yaw = self.current_state.imu_state.rpy[2]
            else:
                yaw = 0.0

            try:
                with open("/tmp/mujoco_rays.json", "r") as f:
                    ray_data = json.load(f)
                self.pos_x = ray_data["robot_world_x"]
                self.pos_y = ray_data["robot_world_y"]
                yaw = ray_data["yaw"]
                slam.update_from_mujoco_rays(ray_data["rays"])
            except Exception:
                cx, cy = world_to_cell(self.pos_x, self.pos_y)
                obs, free = sensor_sim(cx, cy, yaw, max_range=400, map_size=800)
                for pt in obs: slam._update_cell(pt[0], pt[1], True)
                for pt in free: slam._update_cell(pt[0], pt[1], False)

            curr_cell_x, curr_cell_y = world_to_cell(self.pos_x, self.pos_y)
            slam.update_pose(curr_cell_x, curr_cell_y, yaw)


            # ---- Atualizar Visualização (a cada 50 iterações) ----
            viz_counter += 1
            if viz_counter >= 20:
                viz_counter = 0

                # Planeamento de Caminho A*
                # Recalcular caminho apenas se:
                # - ainda não existe caminho
                # - o caminho atual ficou inválido por novo obstáculo
                path_needs_replan = not current_path or not slam.is_path_valid(current_path)

                if path_needs_replan:
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

                # Atualizar Grid
                img.set_data(slam.get_visualization_grid())

                if current_path:
                    path_y = [p[1] for p in current_path]
                    path_x = [p[0] for p in current_path]
                    path_line.set_data(path_y, path_x)
                else:
                    path_line.set_data([], [])

                # Atualizar a posição do robô (ponto vermelho)
                robot_dot.set_data([curr_cell_y], [curr_cell_x])

                if current_goal_cell is not None:
                    goal_dot.set_data([current_goal_cell[1]], [current_goal_cell[0]])
                else:
                    goal_dot.set_data([], [])

                # Zoom e Desenho
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

                if current_path:
                    path_changed = current_path != last_saved_path
                    if path_changed:
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
                        last_path_save_time = now

            cmd = unitree_hg_msg_dds__LowCmd_()
            cmd.crc = self.crc.Crc(cmd)
            low_cmd_pub.Write(cmd)

            elapsed = time.perf_counter() - step_start

            if 0.002 - elapsed > 0:
                time.sleep(0.002 - elapsed)


if __name__ == '__main__':
    RobotController().run()
