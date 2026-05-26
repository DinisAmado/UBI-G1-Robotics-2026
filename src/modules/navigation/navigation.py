# navigation.py
"""
Funções auxiliares de navegação usadas pelo módulo SLAM/Navegação.

Este ficheiro contém apenas lógica reutilizável:
- conversão mundo <-> célula;
- conversão quaternion <-> yaw;
- escolha automática de objetivo antes de obstáculo frontal;
- procura de célula livre próxima;
- validações simples usadas na mesa e na pessoa.

Não deve conter:
- Matplotlib;
- Livox;
- Unitree SDK;
- loop principal;
- código DDS.
"""

import math


# ---------------------------------------------------------------
# Configurações do mapa
# ---------------------------------------------------------------

MAP_ORIGIN_X = -5.0
MAP_ORIGIN_Y = -5.0
MAP_RESOLUTION = 0.05


# ---------------------------------------------------------------
# Configurações da navegação local até à mesa
# ---------------------------------------------------------------

LOCAL_GOAL_STOP_DISTANCE_M = 0.30       # parar a 30 cm da mesa/obstáculo
LOCAL_GOAL_FRONT_ANGLE_DEG = 70         # cone frontal de procura
LOCAL_GOAL_MIN_DISTANCE_M = 0.45        # ignora obstáculos demasiado perto
LOCAL_GOAL_MAX_DISTANCE_M = 4.00        # procura até 4 m
LOCAL_GOAL_SEARCH_RADIUS_M = 0.35       # se o goal cair mal, procura livre à volta
REPLAN_INTERVAL_S = 1.0                 # recalcula no máximo a cada 1 s


# ---------------------------------------------------------------
# Configurações para navegação até à pessoa
# ---------------------------------------------------------------

PERSON_CENTER_TOLERANCE = 0.15          # yaw/offset visual considerado centrado
PERSON_STOP_DISTANCE_M = 0.70           # parar a 70 cm da pessoa
PERSON_FRONT_ANGLE_DEG = 40             # cone mais apertado porque a pessoa deve estar centrada
PERSON_MIN_DISTANCE_M = 0.70            # ignora obstáculos demasiado perto
PERSON_MAX_DISTANCE_M = 4.00
PERSON_SEARCH_RADIUS_M = 0.35


# ---------------------------------------------------------------
# Conversões de coordenadas
# ---------------------------------------------------------------

def world_to_cell(world_x, world_y):
    """
    Converte coordenadas do mundo, em metros, para células da grelha.
    """
    cell_x = int((world_x - MAP_ORIGIN_X) / MAP_RESOLUTION)
    cell_y = int((world_y - MAP_ORIGIN_Y) / MAP_RESOLUTION)
    return cell_x, cell_y


def cell_to_world(cell_x, cell_y):
    """
    Converte células da grelha para coordenadas do mundo, em metros.
    """
    world_x = MAP_ORIGIN_X + cell_x * MAP_RESOLUTION
    world_y = MAP_ORIGIN_Y + cell_y * MAP_RESOLUTION
    return world_x, world_y


# ---------------------------------------------------------------
# Orientação
# ---------------------------------------------------------------

def yaw_from_quaternion(qx, qy, qz, qw):
    """
    Converte quaternion em yaw.
    """
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def quaternion_from_yaw(yaw):
    """
    Converte yaw para quaternion simples no eixo Z.

    Devolve:
        qx, qy, qz, qw
    """
    qx = 0.0
    qy = 0.0
    qz = math.sin(yaw / 2.0)
    qw = math.cos(yaw / 2.0)
    return qx, qy, qz, qw


def normalize_angle(angle):
    """
    Normaliza um ângulo para o intervalo [-pi, pi].
    """
    return math.atan2(math.sin(angle), math.cos(angle))


# ---------------------------------------------------------------
# Validação de objetivos
# ---------------------------------------------------------------

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

    Escolhe a célula mais próxima do ponto desejado e, em empate,
    a mais próxima do robô.
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


# ---------------------------------------------------------------
# Objetivo automático antes do obstáculo frontal
# ---------------------------------------------------------------

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
    4. cria um objetivo stop_distance_m antes desse obstáculo;
    5. se o objetivo não for utilizável, procura uma célula livre próxima.
    """

    rx, ry = robot_cell

    min_cells = int(min_distance_m / MAP_RESOLUTION)
    max_cells = int(max_distance_m / MAP_RESOLUTION)
    stop_cells = max(1, int(stop_distance_m / MAP_RESOLUTION))

    half_angle_rad = math.radians(front_angle_deg / 2.0)

    # Direção "frente" no referencial da grelha.
    # No código atual, x da grelha corresponde ao eixo vertical e y ao horizontal.
    forward_x = math.sin(robot_yaw)
    forward_y = math.cos(robot_yaw)

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

    vx = ox - rx
    vy = oy - ry
    dist = math.sqrt(vx * vx + vy * vy)

    if dist <= stop_cells:
        return None, best_obstacle

    ux = vx / dist
    uy = vy / dist

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


def person_is_centered(offset_x, tolerance=PERSON_CENTER_TOLERANCE):
    """
    Verifica se a pessoa está suficientemente centrada.

    offset_x/yaw:
        valor vindo da visão.
        0 significa centrado.
    """
    if offset_x is None:
        return False

    return abs(offset_x) <= tolerance


def reached_goal(robot_cell, goal_cell, tolerance_m=0.25):
    """
    Verifica se o robô chegou perto do objetivo.
    """
    if goal_cell is None:
        return False

    rx, ry = robot_cell
    gx, gy = goal_cell

    tolerance_cells = max(1, int(tolerance_m / MAP_RESOLUTION))
    dist_cells = math.sqrt((gx - rx) ** 2 + (gy - ry) ** 2)

    return dist_cells <= tolerance_cells
