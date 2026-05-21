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
from unitree_sdk2py.idl.nav_msgs.msg.dds_ import Odometry_

from slam_navigation import SLAMNavigation
from sensores import sensor_mujoco_json, pointcloud_to_occupancy_points
from livox_receiver import LivoxReceiver

from navigation import (
    PERSON_CENTER_TOLERANCE,
    PERSON_STOP_DISTANCE_M,
    PERSON_FRONT_ANGLE_DEG,
    PERSON_MIN_DISTANCE_M,
    PERSON_MAX_DISTANCE_M,
    PERSON_SEARCH_RADIUS_M,
    person_is_centered,
    reached_goal,
)

# ---------------------------------------------------------------
# Configurações do mapa
# ---------------------------------------------------------------
MAP_ORIGIN_X = -5.0
MAP_ORIGIN_Y = -5.0
MAP_RESOLUTION = 0.05


# ---------------------------------------------------------------
# Configurações da navegação local
# ---------------------------------------------------------------
LOCAL_GOAL_STOP_DISTANCE_M = 0.30       # parar a 30 cm do obstáculo
LOCAL_GOAL_FRONT_ANGLE_DEG = 70         # cone frontal de procura
LOCAL_GOAL_MIN_DISTANCE_M = 0.45        # ignora obstáculos demasiado perto
LOCAL_GOAL_MAX_DISTANCE_M = 4.00        # procura até 4 m
LOCAL_GOAL_SEARCH_RADIUS_M = 0.35       # se o goal cair mal, procura livre à volta
REPLAN_INTERVAL_S = 1.0                 # em modo local, recalcula no máximo a cada 1 s


def world_to_cell(world_x, world_y):
    cell_x = int((world_x - MAP_ORIGIN_X) / MAP_RESOLUTION)
    cell_y = int((world_y - MAP_ORIGIN_Y) / MAP_RESOLUTION)
    return cell_x, cell_y


def cell_to_world(cell_x, cell_y):
    world_x = MAP_ORIGIN_X + cell_x * MAP_RESOLUTION
    world_y = MAP_ORIGIN_Y + cell_y * MAP_RESOLUTION
    return world_x, world_y


def yaw_from_quaternion(qx, qy, qz, qw):
    """
    Converte quaternion em yaw.
    """
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def normalize_angle(angle):
    """
    Normaliza ângulo para [-pi, pi].
    """
    return math.atan2(math.sin(angle), math.cos(angle))


def is_goal_cell_usable(slam, x, y):
    """
    Verifica se uma célula pode ser usada como objetivo.

    Para já aceitamos células livres ou desconhecidas, desde que não estejam ocupadas.
    Isto é útil porque o mapa local pode ter zonas ainda não totalmente observadas.
    """
    if not (0 <= x < slam.map_size and 0 <= y < slam.map_size):
        return False

    if slam.is_occupied(x, y):
        return False

    return True


def find_nearby_free_goal(slam, desired_goal, robot_cell, search_radius_m=0.35):
    """
    Se o ponto de paragem calculado não for válido, procura uma célula livre/desocupada
    próxima desse ponto.

    Escolhe a célula mais próxima do ponto desejado e, em empate, a mais próxima do robô.
    """
    gx, gy = desired_goal
    rx, ry = robot_cell

    search_radius_cells = max(1, int(search_radius_m / MAP_RESOLUTION))

    candidates = []

    for dx in range(-search_radius_cells, search_radius_cells + 1):
        for dy in range(-search_radius_cells, search_radius_cells + 1):
            cx = gx + dx
            cy = gy + dy

            if not (0 <= cx < slam.map_size and 0 <= cy < slam.map_size):
                continue

            if not is_goal_cell_usable(slam, cx, cy):
                continue

            dist_to_desired = math.sqrt(dx * dx + dy * dy)
            dist_to_robot = math.sqrt((cx - rx) ** 2 + (cy - ry) ** 2)

            score = dist_to_desired + 0.02 * dist_to_robot
            candidates.append((score, cx, cy))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])
    return candidates[0][1], candidates[0][2]


def find_nearest_front_obstacle_goal(
    slam,
    robot_cell,
    robot_yaw,
    stop_distance_m=0.30,
    front_angle_deg=70,
    min_distance_m=0.45,
    max_distance_m=4.00,
    search_radius_m=0.35
):
    """
    Encontra automaticamente um objetivo local perto do obstáculo frontal mais próximo.

    Lógica:
    1. procura células ocupadas na occupancy grid;
    2. filtra apenas obstáculos à frente do robô, dentro de um cone frontal;
    3. escolhe o obstáculo mais próximo;
    4. cria um objetivo 30 cm antes desse obstáculo;
    5. se o objetivo não for utilizável, procura uma célula livre próxima.

    Isto serve para o cenário:
        robô -> mesa -> pessoas

    Como a mesa é o obstáculo frontal mais próximo, o robô escolhe parar antes dela.
    """
    rx, ry = robot_cell

    min_cells = int(min_distance_m / MAP_RESOLUTION)
    max_cells = int(max_distance_m / MAP_RESOLUTION)
    stop_cells = max(1, int(stop_distance_m / MAP_RESOLUTION))

    half_angle_rad = math.radians(front_angle_deg / 2.0)

    # Direção "frente" no referencial da grelha.
    # No teu código, x da grelha corresponde ao eixo vertical e y ao horizontal.
    forward_x = math.cos(robot_yaw)
    forward_y = math.sin(robot_yaw)

    best_obstacle = None
    best_dist = float("inf")

    x_min = max(0, rx - max_cells)
    x_max = min(slam.map_size - 1, rx + max_cells)
    y_min = max(0, ry - max_cells)
    y_max = min(slam.map_size - 1, ry + max_cells)

    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):

            if not slam.is_occupied(x, y):
                continue

            vx = x - rx
            vy = y - ry

            dist = math.sqrt(vx * vx + vy * vy)

            if dist < min_cells or dist > max_cells:
                continue

            # Ângulo entre a direção do robô e a direção para o obstáculo
            dot = (vx * forward_x + vy * forward_y) / dist
            dot = max(-1.0, min(1.0, dot))
            angle_to_forward = math.acos(dot)

            if angle_to_forward > half_angle_rad:
                continue

            if dist < best_dist:
                best_dist = dist
                best_obstacle = (x, y)

    if best_obstacle is None:
        return None, None

    ox, oy = best_obstacle

    # Vetor unitário do robô para o obstáculo
    vx = ox - rx
    vy = oy - ry
    dist = math.sqrt(vx * vx + vy * vy)

    if dist <= stop_cells:
        return None, best_obstacle

    ux = vx / dist
    uy = vy / dist

    # Ponto de paragem: 30 cm antes do obstáculo
    desired_gx = int(round(ox - ux * stop_cells))
    desired_gy = int(round(oy - uy * stop_cells))

    desired_goal = (desired_gx, desired_gy)

    if is_goal_cell_usable(slam, desired_gx, desired_gy):
        return desired_goal, best_obstacle

    nearby_goal = find_nearby_free_goal(
        slam=slam,
        desired_goal=desired_goal,
        robot_cell=robot_cell,
        search_radius_m=search_radius_m
    )

    return nearby_goal, best_obstacle


class RobotController:
    def __init__(self):
        self.current_state = None
        self.current_odom = None
        self.crc = CRC()

        self.pos_x = 0.0
        self.pos_y = 0.0

    def LowStateHandler(self, msg: LowState_):
        self.current_state = msg

    def OdomHandler(self, msg: Odometry_):
        self.current_odom = msg

    def run(self):
        plt.close('all')

        real_robot = len(sys.argv) >= 2

        if real_robot:
            interface = sys.argv[1]
            print(f"Modo robô real na interface: {interface}")
            ChannelFactoryInitialize(0, interface)
        else:
            print("Modo simulação")
            ChannelFactoryInitialize(1, "lo")

        low_state_sub = ChannelSubscriber("rt/lowstate", LowState_)
        low_state_sub.Init(self.LowStateHandler, 10)

        odom_sub = ChannelSubscriber("rt/unitree/slam_mapping/odom", Odometry_)
        odom_sub.Init(self.OdomHandler, 10)

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

        lidar = None

        if real_robot:
            print("A iniciar Livox MID-360...")
            lidar = LivoxReceiver(
                config_path="mid360_config.json",
                host_ip="192.168.123.165"
            )
            print("Livox iniciado.")

        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)

        # ---------------------------------------------------
        # Estado da missão
        # ---------------------------------------------------
        # Para já, é manual.
        # Quando a orquestração estiver pronta, isto passa a vir do tópico DDS.
        #
        # Opções:
        #   "GO_TO_TABLE"   -> ir até à mesa
        #   "GO_TO_PERSON"  -> ir até à pessoa
        #   "DONE"          -> terminado
        # ---------------------------------------------------
        
        mission_phase = "GO_TO_TABLE"
        
        current_goal_cell = None
        current_obstacle_cell = None
        current_path = []
        last_replan_time = 0.0
        
        # Valores temporários da visão.
        # Mais tarde estes valores vêm do módulo da visão.
        person_visible = False
        person_offset_x = None

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

        robot_arrow = ax.quiver(
            [0], [0], [0], [0],
            angles='xy',
            scale_units='xy',
            scale=1,
            color='red',
            width=0.004
        )

        path_line, = ax.plot([], [], "g-", linewidth=2, label="Caminho A*")
        goal_dot, = ax.plot([], [], "bo", markersize=7, label="Objetivo")
        obstacle_dot, = ax.plot([], [], "mo", markersize=5, label="Obstáculo alvo")

        ax.set_title("SLAM e Navegação — objetivo automático no obstáculo frontal")
        ax.legend(
            handles=[leg_free, leg_unk, leg_obs, path_line, robot_dot, goal_dot, obstacle_dot],
            loc='upper right',
            fontsize='small'
        )

        plt.show(block=False)
        plt.pause(0.1)

        viz_counter = 0
        last_map_save_time = 0.0
        last_saved_path = None
        MAP_SAVE_INTERVAL = 2.0

        last_odom_debug_time = 0.0
        printed_odom_structure = False
        last_goal_debug_time = 0.0

        while True:
            step_start = time.perf_counter()

            yaw = 0.0

            # ---------------------------------------------------
            # VISÃO — temporário para testes
            # ---------------------------------------------------
            # Mais tarde, estes valores vêm do tópico da visão.
            # Por agora:
            #   person_visible = True significa que a pessoa foi detetada
            #   person_offset_x = 0.0 significa que está centrada
            # ---------------------------------------------------
            
            if mission_phase == "GO_TO_PERSON":
                person_visible = True
                person_offset_x = 0.0
            else:
                person_visible = False
                person_offset_x = None

            # Preferir odometria real do módulo SLAM da Unitree
            if self.current_odom is not None:
                try:
                    if not printed_odom_structure:
                        printed_odom_structure = True
                        print("\n===== DEBUG ESTRUTURA ODOM =====")
                        print("Tipo:", type(self.current_odom))
                        print("Campos:", dir(self.current_odom))
                        print("Mensagem completa:")
                        print(self.current_odom)
                        print("================================\n")

                    pose = self.current_odom.pose.pose

                    self.pos_x = float(pose.position.x)
                    self.pos_y = float(pose.position.y)

                    qx = float(pose.orientation.x)
                    qy = float(pose.orientation.y)
                    qz = float(pose.orientation.z)
                    qw = float(pose.orientation.w)

                    yaw = yaw_from_quaternion(qx, qy, qz, qw)

                except Exception as e:
                    print("Erro ao ler odometria:", e)

                    if self.current_state and hasattr(self.current_state, 'imu_state'):
                        yaw = self.current_state.imu_state.rpy[2]

            else:
                # Fallback: usar IMU caso ainda não haja odometria
                if self.current_state and hasattr(self.current_state, 'imu_state'):
                    yaw = self.current_state.imu_state.rpy[2]

            # ---------------------------------------------------
            # Sensores
            # ---------------------------------------------------

            if not real_robot:
                # ---------------------------------------------------
                # Simulação: ler dados do MuJoCo
                # ---------------------------------------------------
                mujoco_ok, mx, my, myaw, rays = sensor_mujoco_json()

                if mujoco_ok:
                    self.pos_x = mx
                    self.pos_y = my
                    yaw = myaw
                    slam.update_from_mujoco_rays(rays)

            else:
                # ---------------------------------------------------
                # Robô real: ler point cloud do Livox
                # ---------------------------------------------------
                curr_cell_x, curr_cell_y = world_to_cell(self.pos_x, self.pos_y)

                now_debug = time.time()

                if now_debug - last_odom_debug_time >= 1.0:
                    last_odom_debug_time = now_debug

                    if self.current_odom is not None:
                        print(f"ODOM RECEBIDA | pos=({self.pos_x:.3f}, {self.pos_y:.3f}) yaw={yaw:.3f}")
                    else:
                        print("ODOM NÃO RECEBIDA | a usar fallback da IMU/pose inicial")

                xyz = None

                if lidar is not None:
                    xyz = lidar.get_latest_points()

                if xyz is not None:
                    obs, free = pointcloud_to_occupancy_points(
                        xyz,
                        curr_cell_x,
                        curr_cell_y,
                        yaw,
                        map_size=slam.map_size,
                        resolution=MAP_RESOLUTION,
                        max_range_meters=4.0,
                        min_z=-0.30,
                        max_z=1.50,
                        min_dist_m=0.20,
                        point_step=5,
                        apply_yaw=True
                    )

                    # Esquece gradualmente leituras antigas para evitar rastos
                    slam.decay_map(decay_factor=0.97)

                    for pt in free:
                        slam._update_cell(pt[0], pt[1], False)

                    for pt in obs:
                        slam._update_cell(pt[0], pt[1], True)

            curr_cell_x, curr_cell_y = world_to_cell(self.pos_x, self.pos_y)
            slam.update_pose(curr_cell_x, curr_cell_y, yaw)

            # ---------------------------------------------------
            # Visualização e planeamento
            # ---------------------------------------------------
            viz_counter += 1

            if viz_counter >= 20:
                viz_counter = 0

                now = time.time()

                # Recalcular objetivo/caminho se:
                # - ainda não há caminho;
                # - o caminho ficou inválido;
                # - passou algum tempo desde o último replaneamento.
                #
                # Em modo local, isto ajuda porque o mapa muda muito com o LiDAR.
                path_needs_replan = (
                    not current_path
                    or not slam.is_path_valid(current_path)
                    or (now - last_replan_time >= REPLAN_INTERVAL_S)
                )

                if path_needs_replan:
                    # ---------------------------------------------------
                    # FASE 1 — ir até à mesa
                    # ---------------------------------------------------
                    if mission_phase == "GO_TO_TABLE":
                
                        current_goal_cell, current_obstacle_cell = find_nearest_front_obstacle_goal(
                            slam=slam,
                            robot_cell=(curr_cell_x, curr_cell_y),
                            robot_yaw=yaw,
                            stop_distance_m=LOCAL_GOAL_STOP_DISTANCE_M,
                            front_angle_deg=LOCAL_GOAL_FRONT_ANGLE_DEG,
                            min_distance_m=LOCAL_GOAL_MIN_DISTANCE_M,
                            max_distance_m=LOCAL_GOAL_MAX_DISTANCE_M,
                            search_radius_m=LOCAL_GOAL_SEARCH_RADIUS_M
                        )
                
                        if current_goal_cell is not None:
                            current_path = slam.plan_path(current_goal_cell, allow_unknown=True)
                
                            if not current_path:
                                current_goal_cell = None
                        else:
                            current_path = []
                
                    # ---------------------------------------------------
                    # FASE 2 — ir até à pessoa
                    # ---------------------------------------------------
                    elif mission_phase == "GO_TO_PERSON":
                
                        # A pessoa ainda não foi detetada pela visão
                        if not person_visible:
                            current_goal_cell = None
                            current_obstacle_cell = None
                            current_path = []
                            print("PESSOA | ainda não visível pela visão")
                
                        # A pessoa foi detetada, mas ainda não está centrada
                        elif not person_is_centered(person_offset_x):
                            current_goal_cell = None
                            current_obstacle_cell = None
                            current_path = []

            print(
                f"PESSOA | pessoa visível mas não centrada "
                f"(offset_x={person_offset_x:.2f})"
            )

            # Mais tarde, aqui vamos enviar rotação:
            #   offset_x < 0 -> rodar para um lado
            #   offset_x > 0 -> rodar para o outro

        # Pessoa centrada: agora o obstáculo frontal é assumido como a pessoa
        else:
            current_goal_cell, current_obstacle_cell = find_nearest_front_obstacle_goal(
                slam=slam,
                robot_cell=(curr_cell_x, curr_cell_y),
                robot_yaw=yaw,
                stop_distance_m=PERSON_STOP_DISTANCE_M,
                front_angle_deg=PERSON_FRONT_ANGLE_DEG,
                min_distance_m=PERSON_MIN_DISTANCE_M,
                max_distance_m=PERSON_MAX_DISTANCE_M,
                search_radius_m=PERSON_SEARCH_RADIUS_M
            )

            if current_goal_cell is not None:
                current_path = slam.plan_path(current_goal_cell, allow_unknown=True)

                if not current_path:
                    current_goal_cell = None
            else:
                current_path = []

    # ---------------------------------------------------
    # FASE FINAL
    # ---------------------------------------------------
    else:
        current_goal_cell = None
        current_obstacle_cell = None
        current_path = []

    last_replan_time = now

                # Debug do objetivo automático
                if now - last_goal_debug_time >= 1.5:
                    last_goal_debug_time = now

                    if current_goal_cell is not None and current_obstacle_cell is not None:
                        gx, gy = current_goal_cell
                        ox, oy = current_obstacle_cell
                        dist_obs_m = math.sqrt((ox - curr_cell_x) ** 2 + (oy - curr_cell_y) ** 2) * MAP_RESOLUTION
                        dist_goal_m = math.sqrt((gx - curr_cell_x) ** 2 + (gy - curr_cell_y) ** 2) * MAP_RESOLUTION

                        print(
                            f"FASE={mission_phase} | obstáculo=({ox},{oy}) dist={dist_obs_m:.2f} m | "
                            f"goal=({gx},{gy}) dist={dist_goal_m:.2f} m | path={len(current_path)}"
                        )
                    else:
                        print(f"FASE={mission_phase} | nenhum obstáculo frontal válido encontrado")

                img.set_data(slam.get_visualization_grid())

                if current_path:
                    path_y = [p[1] for p in current_path]
                    path_x = [p[0] for p in current_path]
                    path_line.set_data(path_y, path_x)
                else:
                    path_line.set_data([], [])

                robot_dot.set_data([curr_cell_y], [curr_cell_x])

                arrow_len = 10  # comprimento da seta em células
                # eixo horizontal = cell_y
                # eixo vertical   = cell_x
                arrow_dx = arrow_len * math.sin(yaw)
                arrow_dy = arrow_len * math.cos(yaw)

                robot_arrow.set_offsets(np.array([[curr_cell_y, curr_cell_x]]))
                robot_arrow.set_UVC(np.array([arrow_dx]), np.array([arrow_dy]))

                if current_goal_cell is not None:
                    goal_dot.set_data([current_goal_cell[1]], [current_goal_cell[0]])
                else:
                    goal_dot.set_data([], [])

                if current_obstacle_cell is not None:
                    obstacle_dot.set_data([current_obstacle_cell[1]], [current_obstacle_cell[0]])
                else:
                    obstacle_dot.set_data([], [])

                ax.set_xlim(curr_cell_y - 100, curr_cell_y + 100)
                ax.set_ylim(curr_cell_x - 100, curr_cell_x + 100)

                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.001)

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
                        "obstacle_cell": None if current_obstacle_cell is None else {
                            "x": int(current_obstacle_cell[0]),
                            "y": int(current_obstacle_cell[1])
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
