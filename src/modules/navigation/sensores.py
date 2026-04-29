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