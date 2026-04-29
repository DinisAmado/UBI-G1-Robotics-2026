# slam_navigation.py

import numpy as np
import math
import heapq


# Valores em log-odds para atualização Bayesiana
LOG_ODD_OCCUPIED = 0.85
LOG_ODD_FREE     = -0.4
LOG_ODD_MIN      = -5.0
LOG_ODD_MAX      = 5.0


def prob_from_log_odds(l):
    """Converte log-odds em probabilidade."""
    return 1.0 - 1.0 / (1.0 + math.exp(l))


class SLAMNavigation:

    def __init__(self, map_size=800, resolution=0.1, num_rays=144, max_range=400):
        self.map_size   = map_size
        self.resolution = resolution
        self.num_rays   = num_rays
        self.max_range  = max_range

        # Occupancy grid em log-odds
        # 0.0 = desconhecido, valores negativos = livre, valores positivos = ocupado
        self.log_odds_grid = np.zeros((map_size, map_size), dtype=float)

        # Pose do robô em células
        self.robot_x   = map_size // 2
        self.robot_y   = map_size // 2
        self.robot_yaw = 0.0

    # ------------------------------------------------------------------
    # Pose
    # ------------------------------------------------------------------

    def update_pose(self, x, y, yaw):
        self.robot_x   = int(np.clip(x, 0, self.map_size - 1))
        self.robot_y   = int(np.clip(y, 0, self.map_size - 1))
        self.robot_yaw = yaw

    # ------------------------------------------------------------------
    # Atualização com raios do MuJoCo
    # ------------------------------------------------------------------

    def update_from_mujoco_rays(self, ray_hits):
        for angle, dist_cells in ray_hits:
            max_d = int(dist_cells) if dist_cells > 0 else self.max_range
            max_d = min(max_d, self.max_range)

            # Células livres ao longo do raio
            for d in range(1, max_d):
                cx = int(self.robot_x + d * math.cos(angle))
                cy = int(self.robot_y + d * math.sin(angle))

                if self._in_bounds(cx, cy):
                    self._update_cell(cx, cy, occupied=False)

            # Célula final como obstáculo
            if 0 < dist_cells <= self.max_range:
                ox = int(self.robot_x + dist_cells * math.cos(angle))
                oy = int(self.robot_y + dist_cells * math.sin(angle))

                if self._in_bounds(ox, oy):
                    self._update_cell(ox, oy, occupied=True)

    # ------------------------------------------------------------------
    # Occupancy grid
    # ------------------------------------------------------------------

    def get_probability_grid(self):
        """Converte toda a grelha de log-odds para probabilidades."""
        return 1.0 - 1.0 / (1.0 + np.exp(self.log_odds_grid))

    def get_visualization_grid(self):
        """
        0 = livre
        1 = desconhecido
        2 = ocupado
        """

        viz = np.full((self.map_size, self.map_size), 1)
        prob = self.get_probability_grid()

        viz[prob < 0.4] = 0
        viz[prob > 0.6] = 2

        return viz

    def is_occupied(self, x, y):
        """
        True se a célula estiver ocupada ou fora do mapa.
        Usa apenas occupancy grid, sem costmap nem inflação.
        """

        if not self._in_bounds(x, y):
            return True

        prob = prob_from_log_odds(self.log_odds_grid[x, y])
        return prob > 0.6

    def is_unknown(self, x, y):
        if not self._in_bounds(x, y):
            return True

        prob = prob_from_log_odds(self.log_odds_grid[x, y])
        return 0.4 <= prob <= 0.6

    # ------------------------------------------------------------------
    # Planeamento A* apenas sobre occupancy grid
    # ------------------------------------------------------------------

    def plan_path(self, goal, allow_unknown=True):
        """
        A* simples sobre occupancy grid.

        goal:
            (gx, gy) em células

        allow_unknown:
            True  -> permite planear por zonas desconhecidas
            False -> bloqueia zonas desconhecidas

        Devolve:
            lista de (x, y) com o caminho
            lista vazia se não houver caminho
        """

        start = (self.robot_x, self.robot_y)
        end   = (int(goal[0]), int(goal[1]))

        if not self._in_bounds(end[0], end[1]):
            return []

        if self.is_occupied(end[0], end[1]):
            return []

        if not allow_unknown and self.is_unknown(end[0], end[1]):
            return []

        def heuristic(a, b):
            dx = abs(a[0] - b[0])
            dy = abs(a[1] - b[1])
            return (dx + dy) + (1.414 - 2.0) * min(dx, dy)

        neighbors = [
            (-1,  0), (1,  0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)
        ]

        open_set = []
        heapq.heappush(open_set, (0.0, start))

        came_from = {}
        g_score = {start: 0.0}
        closed_set = set()

        while open_set:
            _, curr = heapq.heappop(open_set)

            if curr in closed_set:
                continue

            if curr == end:
                return self._reconstruct_path(came_from, curr)

            closed_set.add(curr)

            for dx, dy in neighbors:
                nx = curr[0] + dx
                ny = curr[1] + dy
                neighbor = (nx, ny)

                if not self._in_bounds(nx, ny):
                    continue

                if self.is_occupied(nx, ny):
                    continue

                if not allow_unknown and self.is_unknown(nx, ny):
                    continue

                # Evita cortar cantos na diagonal
                if dx != 0 and dy != 0:
                    if self.is_occupied(curr[0] + dx, curr[1]):
                        continue
                    if self.is_occupied(curr[0], curr[1] + dy):
                        continue

                step_cost = 1.414 if dx != 0 and dy != 0 else 1.0
                tentative_g = g_score[curr] + step_cost

                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = curr
                    g_score[neighbor] = tentative_g

                    f_score = tentative_g + heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score, neighbor))

        return []


    def is_path_valid(self, path):
        """
        Verifica se o caminho atual continua válido.

        Devolve False se:
        - o caminho estiver vazio
        - alguma célula do caminho ficou ocupada
        - alguma célula está fora do mapa
        """

        if not path:
            return False

        for x, y in path:
            if self.is_occupied(x, y):
                return False

        return True


    # ------------------------------------------------------------------
    # Utilitários
    # ------------------------------------------------------------------

    def _in_bounds(self, x, y):
        return 0 <= x < self.map_size and 0 <= y < self.map_size

    def _update_cell(self, x, y, occupied: bool):
        if not self._in_bounds(x, y):
            return

        delta = LOG_ODD_OCCUPIED if occupied else LOG_ODD_FREE

        self.log_odds_grid[x, y] = np.clip(
            self.log_odds_grid[x, y] + delta,
            LOG_ODD_MIN,
            LOG_ODD_MAX
        )

    def _reconstruct_path(self, came_from, curr):
        path = []

        while curr in came_from:
            path.append(curr)
            curr = came_from[curr]

        path.reverse()
        return path

    def _bresenham(self, x0, y0, x1, y1):
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
