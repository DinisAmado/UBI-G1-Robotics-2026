# sensores.py

import json
import math
import os
import numpy as np


# ---------------------------------------------------------------
# SENSOR MUJOCO
# ---------------------------------------------------------------
def sensor_mujoco_json(json_path="/tmp/mujoco_rays.json"):
    """
    Lê os raios simulados gerados pelo MuJoCo.

    Devolve:
        success, robot_world_x, robot_world_y, yaw, rays
    """

    if not os.path.exists(json_path):
        return False, None, None, None, []

    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        return (
            True,
            data["robot_world_x"],
            data["robot_world_y"],
            data["yaw"],
            data["rays"]
        )

    except Exception:
        return False, None, None, None, []


# ---------------------------------------------------------------
# SENSOR REAL / LIDAR
# ---------------------------------------------------------------
def sensor_real(
    state,
    robot_x,
    robot_y,
    robot_yaw,
    map_size=800,
    resolution=0.1,
    max_range_meters=10.0
):
    """
    Converte dados reais de LiDAR em pontos livres e obstáculos.

    Funciona apenas se o estado tiver algo equivalente a:
        state.lidar.ranges
        state.lidar.angle_min
        state.lidar.angle_increment

    Devolve:
        obstacle_points, free_points
    """

    obstacle_points = []
    free_points = []

    if state is None:
        return obstacle_points, free_points

    if not hasattr(state, "lidar"):
        return obstacle_points, free_points

    lidar = state.lidar

    if not hasattr(lidar, "ranges"):
        return obstacle_points, free_points

    ranges = lidar.ranges

    if len(ranges) == 0:
        return obstacle_points, free_points

    angle_min = getattr(lidar, "angle_min", -math.pi)
    angle_increment = getattr(
        lidar,
        "angle_increment",
        (2 * math.pi) / len(ranges)
    )

    for i, r in enumerate(ranges):
        if r is None:
            continue

        if math.isnan(r) or r <= 0 or r > max_range_meters:
            continue

        angle = robot_yaw + angle_min + i * angle_increment
        dist_cells = int(r / resolution)

        for d in range(1, dist_cells):
            cx = int(robot_x + d * math.cos(angle))
            cy = int(robot_y + d * math.sin(angle))

            if 0 <= cx < map_size and 0 <= cy < map_size:
                free_points.append((cx, cy))

        ox = int(robot_x + dist_cells * math.cos(angle))
        oy = int(robot_y + dist_cells * math.sin(angle))

        if 0 <= ox < map_size and 0 <= oy < map_size:
            obstacle_points.append((ox, oy))

    return obstacle_points, free_points


# ---------------------------------------------------------------
# SENSOR SIMULADO ALEATÓRIO
# ---------------------------------------------------------------
def sensor_sim(
    robot_x,
    robot_y,
    robot_yaw,
    num_rays=144,
    max_range=400,
    map_size=800,
    min_obstacle_dist=15
):
    """
    Sensor simulado aleatório.

    Deve ser usado apenas para testes, não para demonstração final.
    """

    obstacle_points = []
    free_points = []

    angle_step = (2 * math.pi) / num_rays

    for i in range(num_rays):
        angle = robot_yaw + i * angle_step

        obstacle_dist = np.random.randint(
            min_obstacle_dist,
            max_range + 1
        )

        for d in range(1, obstacle_dist):
            cx = int(robot_x + d * math.cos(angle))
            cy = int(robot_y + d * math.sin(angle))

            if 0 <= cx < map_size and 0 <= cy < map_size:
                free_points.append((cx, cy))

        ox = int(robot_x + obstacle_dist * math.cos(angle))
        oy = int(robot_y + obstacle_dist * math.sin(angle))

        if 0 <= ox < map_size and 0 <= oy < map_size:
            obstacle_points.append((ox, oy))

    return obstacle_points, free_points


# ---------------------------------------------------------------
# POINT CLOUD LIVOX -> OCCUPANCY GRID
# ---------------------------------------------------------------

def bresenham_cells(x0, y0, x1, y1):
    """
    Devolve células entre dois pontos usando Bresenham.
    Usado para marcar espaço livre entre o robô e o obstáculo.
    """

    cells = []

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)

    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1

    err = dx - dy

    while True:
        cells.append((x0, y0))

        if x0 == x1 and y0 == y1:
            break

        e2 = 2 * err

        if e2 > -dy:
            err -= dy
            x0 += sx

        if e2 < dx:
            err += dx
            y0 += sy

    return cells


def pointcloud_to_occupancy_points(
    xyz,
    robot_x,
    robot_y,
    robot_yaw,
    map_size=200,
    resolution=0.05,
    max_range_meters=5.0,
    min_z=-0.30,
    max_z=1.50,
    min_dist_m=0.20,
    point_step=5
):
    """
    Converte point cloud Livox Nx3 em pontos ocupados e livres.

    xyz:
        matriz Nx3 com pontos no referencial do LiDAR/robô

    robot_x, robot_y:
        posição do robô em células

    robot_yaw:
        orientação do robô em radianos

    Devolve:
        obstacle_points, free_points
    """

    obstacle_points = set()
    free_points = set()

    if xyz is None or len(xyz) == 0:
        return [], []

    cos_yaw = math.cos(robot_yaw)
    sin_yaw = math.sin(robot_yaw)

    # Processa só alguns pontos para não ficar demasiado pesado
    for p in xyz[::point_step]:
        x_lidar = float(p[0])
        y_lidar = float(p[1])
        z_lidar = float(p[2])

        # Filtrar chão/teto/pontos demasiado altos ou baixos
        if z_lidar < min_z or z_lidar > max_z:
            continue

        dist = math.sqrt(x_lidar * x_lidar + y_lidar * y_lidar)

        if dist < min_dist_m or dist > max_range_meters:
            continue

        # Transformar do referencial do LiDAR/robô para o referencial do mapa
        x_map_rel = cos_yaw * x_lidar - sin_yaw * y_lidar
        y_map_rel = sin_yaw * x_lidar + cos_yaw * y_lidar

        ox = int(robot_x + x_map_rel / resolution)
        oy = int(robot_y + y_map_rel / resolution)

        if not (0 <= ox < map_size and 0 <= oy < map_size):
            continue

        obstacle_points.add((ox, oy))

        # Marcar espaço livre desde o robô até ao obstáculo
        ray_cells = bresenham_cells(robot_x, robot_y, ox, oy)

        for cx, cy in ray_cells[:-1]:
            if 0 <= cx < map_size and 0 <= cy < map_size:
                free_points.add((cx, cy))

    return list(obstacle_points), list(free_points)
