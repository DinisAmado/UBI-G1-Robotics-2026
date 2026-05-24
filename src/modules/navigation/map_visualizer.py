# map_visualizer.py
"""
Visualização do occupancy grid para o módulo SLAM/Navegação.

Este ficheiro contém apenas Matplotlib.
Não contém DDS, Livox, Unitree SDK nem lógica de planeamento.
"""

import math
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap


class MapVisualizer:
    def __init__(self, title="SLAM e Navegação — Occupancy Grid", zoom_cells=100):
        self.zoom_cells = zoom_cells

        self.custom_cmap = ListedColormap(["white", "lightgrey", "black"])

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 8))

        self.img = None

        leg_free = mpatches.Patch(color="white", label="Livre", ec="black")
        leg_unk = mpatches.Patch(color="lightgrey", label="Desconhecido")
        leg_obs = mpatches.Patch(color="black", label="Obstáculo")

        self.robot_dot, = self.ax.plot([], [], "ro", markersize=8, label="Robô G1")

        self.robot_arrow = self.ax.quiver(
            [0], [0], [0], [0],
            angles="xy",
            scale_units="xy",
            scale=1,
            color="red",
            width=0.004,
        )

        self.path_line, = self.ax.plot([], [], "g-", linewidth=2, label="Caminho A*")
        self.goal_dot, = self.ax.plot([], [], "bo", markersize=7, label="Objetivo")
        self.obstacle_dot, = self.ax.plot([], [], "mo", markersize=5, label="Obstáculo alvo")

        self.ax.set_title(title)
        self.ax.legend(
            handles=[
                leg_free,
                leg_unk,
                leg_obs,
                self.path_line,
                self.robot_dot,
                self.goal_dot,
                self.obstacle_dot,
            ],
            loc="upper right",
            fontsize="small",
        )

        plt.show(block=False)
        plt.pause(0.1)

    def setup_image(self, slam):
        if self.img is None:
            self.img = self.ax.imshow(
                slam.get_visualization_grid(),
                cmap=self.custom_cmap,
                vmin=0,
                vmax=2,
                origin="lower",
            )

    def update(self, slam, robot_cell, yaw, path=None, goal_cell=None, obstacle_cell=None):
        """
        Atualiza a janela do mapa.

        robot_cell:
            (cell_x, cell_y)

        path:
            lista de (cell_x, cell_y)
        """
        self.setup_image(slam)

        curr_cell_x, curr_cell_y = robot_cell

        self.img.set_data(slam.get_visualization_grid())

        if path:
            path_y = [p[1] for p in path]
            path_x = [p[0] for p in path]
            self.path_line.set_data(path_y, path_x)
        else:
            self.path_line.set_data([], [])

        self.robot_dot.set_data([curr_cell_y], [curr_cell_x])

        arrow_len = 10
        arrow_dx = arrow_len * math.sin(yaw)
        arrow_dy = arrow_len * math.cos(yaw)

        self.robot_arrow.set_offsets(np.array([[curr_cell_y, curr_cell_x]]))
        self.robot_arrow.set_UVC(np.array([arrow_dx]), np.array([arrow_dy]))

        if goal_cell is not None:
            self.goal_dot.set_data([goal_cell[1]], [goal_cell[0]])
        else:
            self.goal_dot.set_data([], [])

        if obstacle_cell is not None:
            self.obstacle_dot.set_data([obstacle_cell[1]], [obstacle_cell[0]])
        else:
            self.obstacle_dot.set_data([], [])

        self.ax.set_xlim(curr_cell_y - self.zoom_cells, curr_cell_y + self.zoom_cells)
        self.ax.set_ylim(curr_cell_x - self.zoom_cells, curr_cell_x + self.zoom_cells)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)
