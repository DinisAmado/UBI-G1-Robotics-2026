#!/usr/bin/env python3
"""
main.py — Módulo SLAM + Navegação para integração com a Orquestração
Grupo 2 — Robótica Inteligente 2025/2026

Publica:
    rt/slam/pose        (SlamPoseMsg)
    rt/slam/locations   (Locations)
    rt/nav/status       (NavStatusMsg)
    rt/nav/path         (NavPath)
    rt/motion/cmd_vel   (CmdVel)

Subscreve:
    rt/nav/goal              (NavGoal)
    rt/motion/odometry       (OdometryMsg)
    rt/vision/persons        (Persons)
    rt/orchestration/state   (OrchestratorState)

Modos úteis:
    python3 main.py
    python3 main.py --viz
    python3 main.py --lidar --viz
    python3 main.py --lidar --viz --host-ip 192.168.123.165
"""

import argparse
import os
import sys
import time
import math
import logging

# ==============================================================================
# CAMINHO PARA IMPORTAR idl_ri.py E qos_profiles.py
# ==============================================================================

pasta_atual = os.path.dirname(os.path.abspath(__file__))
pasta_src = os.path.abspath(os.path.join(pasta_atual, "../.."))
sys.path.append(pasta_src)

# ==============================================================================
# IMPORTS DDS
# ==============================================================================

from cyclonedds.domain import DomainParticipant
from cyclonedds.topic import Topic
from cyclonedds.pub import Publisher, DataWriter
from cyclonedds.sub import Subscriber, DataReader

from idl_ri import (
    Header,
    Status,
    Vector3,
    Quaternion,
    Pose,
    Location,
    Locations,
    SlamPoseMsg,
    NavGoal,
    NavStatusMsg,
    NavPath,
    CmdVel,
    OdometryMsg,
    Persons,
    OrchestratorState,
    Phase,
)

from qos_profiles import (
    QOS_SLAM_POSE,
    QOS_SLAM_MAP,
    QOS_NAV,
    QOS_MOTION,
    QOS_ODOMETRY,
    QOS_VISION,
    QOS_ORCHESTRATION,
)

# ==============================================================================
# IMPORTS DO NOSSO MÓDULO
# ==============================================================================

from slam_navigation import SLAMNavigation
from sensores import pointcloud_to_occupancy_points
from livox_receiver import LivoxReceiver

from navigation import (
    MAP_RESOLUTION,

    LOCAL_GOAL_STOP_DISTANCE_M,
    LOCAL_GOAL_FRONT_ANGLE_DEG,
    LOCAL_GOAL_MIN_DISTANCE_M,
    LOCAL_GOAL_MAX_DISTANCE_M,
    LOCAL_GOAL_SEARCH_RADIUS_M,
    REPLAN_INTERVAL_S,

    PERSON_CENTER_TOLERANCE,
    PERSON_STOP_DISTANCE_M,
    PERSON_FRONT_ANGLE_DEG,
    PERSON_MIN_DISTANCE_M,
    PERSON_MAX_DISTANCE_M,
    PERSON_SEARCH_RADIUS_M,

    world_to_cell,
    cell_to_world,
    yaw_from_quaternion,
    quaternion_from_yaw,
    normalize_angle,
    find_nearest_front_obstacle_goal,
)

try:
    from map_visualizer import MapVisualizer
except Exception:
    MapVisualizer = None


# ==============================================================================
# CONFIGURAÇÃO
# ==============================================================================

DOMAIN_ID = 0
log = logging.getLogger("navigation")

MAP_SIZE = 200
LOOP_DT = 0.02  # 50 Hz

ENABLE_MOTION = False

GOAL_TOLERANCE_M = 0.25
WAYPOINT_TOLERANCE_M = 0.20

MAX_LINEAR_SPEED = 0.15
MAX_ANGULAR_SPEED = 0.30
YAW_ALIGN_THRESHOLD = 0.35

ODOM_TIMEOUT_S = 1.0
CMD_ZERO_BURST_COUNT = 3
SAFE_STATUS_INTERVAL_S = 0.5

LIDAR_MAX_RANGE_M = 4.0
LIDAR_MIN_Z = -0.30
LIDAR_MAX_Z = 1.50
LIDAR_MIN_DIST_M = 0.20
LIDAR_POINT_STEP = 5
MAP_DECAY_FACTOR = 0.97

KNOWN_LOCATIONS = {
    "inicio": (0.0, 0.0),
    "mesa": (2.0, 0.0),
    "table": (2.0, 0.0),
    "pessoa_1": (3.0, -1.0),
    "pessoa_2": (3.0, 0.0),
    "pessoa_3": (3.0, 1.0),
}


# ==============================================================================
# CLASSE PRINCIPAL
# ==============================================================================

class NavigationModule:
    def __init__(
        self,
        enable_lidar=False,
        enable_viz=False,
        host_ip="192.168.123.165",
        config_path="mid360_config.json",
    ):
        self.seq = 0

        self.enable_lidar = enable_lidar
        self.enable_viz = enable_viz
        self.host_ip = host_ip
        self.config_path = config_path

        self.current_pose = Pose(
            position=Vector3(x=0.0, y=0.0, z=0.0),
            orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
        )

        self.current_yaw = 0.0
        self.current_goal = None
        self.current_goal_pose = None
        self.current_goal_cell = None
        self.current_obstacle_cell = None
        self.current_path = []
        self.current_waypoint_index = 0
        self.navigation_active = False

        self.current_phase = Phase.IDLE
        self.target_person_id = ""
        self.latest_persons = None
        self.person_visible = False
        self.person_yaw_error = None

        self.last_locations_time = 0.0
        self.last_status_time = 0.0
        self.last_orch_log_time = 0.0
        self.last_vision_log_time = 0.0
        self.last_replan_time = 0.0

        self.last_odom_time = 0.0
        self.last_safe_stop_reason = ""
        self.last_safe_stop_log_time = 0.0
        self.orchestration_allows_navigation = True

        self.slam = SLAMNavigation(
            map_size=MAP_SIZE,
            resolution=MAP_RESOLUTION,
            num_rays=144,
            max_range=int(LIDAR_MAX_RANGE_M / MAP_RESOLUTION),
        )

        self.lidar = None
        if self.enable_lidar:
            log.info("A iniciar Livox MID-360 | config=%s | host_ip=%s", self.config_path, self.host_ip)
            self.lidar = LivoxReceiver(
                config_path=self.config_path,
                host_ip=self.host_ip,
            )
            log.info("Livox iniciado.")

        self.visualizer = None
        if self.enable_viz:
            if MapVisualizer is None:
                raise RuntimeError("Não foi possível importar MapVisualizer. Verifica o ficheiro map_visualizer.py.")
            self.visualizer = MapVisualizer()

        # ----------------------------------------------------------------------
        # DDS SETUP
        # ----------------------------------------------------------------------

        self.dp = DomainParticipant(DOMAIN_ID)
        pub = Publisher(self.dp)
        sub = Subscriber(self.dp)

        # SLAM
        t_pose = Topic(self.dp, "rt/slam/pose", SlamPoseMsg, qos=QOS_SLAM_POSE)
        t_locations = Topic(self.dp, "rt/slam/locations", Locations, qos=QOS_SLAM_MAP)

        self.w_pose = DataWriter(pub, t_pose)
        self.w_locations = DataWriter(pub, t_locations)

        # Navegação
        t_goal = Topic(self.dp, "rt/nav/goal", NavGoal, qos=QOS_NAV)
        t_status = Topic(self.dp, "rt/nav/status", NavStatusMsg, qos=QOS_NAV)
        t_path = Topic(self.dp, "rt/nav/path", NavPath, qos=QOS_NAV)

        self.r_goal = DataReader(sub, t_goal)
        self.w_status = DataWriter(pub, t_status)
        self.w_path = DataWriter(pub, t_path)

        # Movimento
        t_cmd_vel = Topic(self.dp, "rt/motion/cmd_vel", CmdVel, qos=QOS_MOTION)
        t_odom = Topic(self.dp, "rt/motion/odometry", OdometryMsg, qos=QOS_ODOMETRY)

        self.w_cmd_vel = DataWriter(pub, t_cmd_vel)
        self.r_odom = DataReader(sub, t_odom)

        # Visão — pessoas
        t_persons = Topic(self.dp, "rt/vision/persons", Persons, qos=QOS_VISION)
        self.r_persons = DataReader(sub, t_persons)

        # Orquestração — estado global
        t_orch_state = Topic(self.dp, "rt/orchestration/state", OrchestratorState, qos=QOS_ORCHESTRATION)
        self.r_orch_state = DataReader(sub, t_orch_state)

    # ==========================================================================
    # HELPERS
    # ==========================================================================

    def header(self, frame_id="nav") -> Header:
        self.seq += 1
        return Header(timestamp_ns=time.time_ns(), frame_id=frame_id, seq=self.seq)

    def current_robot_cell(self):
        return world_to_cell(
            self.current_pose.position.x,
            self.current_pose.position.y,
        )

    def pose_from_xy(self, x, y, yaw=0.0) -> Pose:
        qx, qy, qz, qw = quaternion_from_yaw(yaw)
        return Pose(
            position=Vector3(x=x, y=y, z=0.0),
            orientation=Quaternion(x=qx, y=qy, z=qz, w=qw),
        )

    def distance_to_goal(self, goal_pose: Pose) -> float:
        dx = goal_pose.position.x - self.current_pose.position.x
        dy = goal_pose.position.y - self.current_pose.position.y
        return math.sqrt(dx * dx + dy * dy)

    # ==========================================================================
    # PUBLICAÇÕES
    # ==========================================================================

    def publish_pose(self):
        self.w_pose.write(SlamPoseMsg(header=self.header("slam"), pose=self.current_pose))

    def publish_locations(self):
        locs = [
            Location(name=name, pose=self.pose_from_xy(x, y))
            for name, (x, y) in KNOWN_LOCATIONS.items()
        ]

        self.w_locations.write(Locations(header=self.header("slam"), locations=locs))

    def publish_status(self, status: Status, reason="", progress=0.0):
        self.w_status.write(
            NavStatusMsg(
                header=self.header("nav"),
                status=status,
                reason=reason,
                progress=progress,
            )
        )
        log.info("[STATUS] %s | %s | %.2f", status.name, reason, progress)

    def publish_path(self, path_cells):
        waypoints = []

        for cell_x, cell_y in path_cells:
            x, y = cell_to_world(cell_x, cell_y)
            waypoints.append(self.pose_from_xy(x, y))

        self.w_path.write(NavPath(header=self.header("nav"), waypoints=waypoints))

    def send_cmd_vel(self, vx=0.0, vy=0.0, wz=0.0):
        self.w_cmd_vel.write(CmdVel(header=self.header("nav"), vx=vx, vy=vy, wz=wz))

    def stop_robot(self):
        self.send_cmd_vel(0.0, 0.0, 0.0)

    def safe_stop(self, reason="Paragem de segurança", publish_status=False):
        for _ in range(CMD_ZERO_BURST_COUNT):
            self.stop_robot()

        now = time.time()

        if reason != self.last_safe_stop_reason or now - self.last_safe_stop_log_time >= SAFE_STATUS_INTERVAL_S:
            log.warning("[SAFE STOP] %s", reason)
            self.last_safe_stop_reason = reason
            self.last_safe_stop_log_time = now

            if publish_status:
                self.publish_status(Status.FAILED, reason, 0.0)

    # ==========================================================================
    # LEITURAS DDS
    # ==========================================================================

    def poll_goal(self):
        samples = self.r_goal.take(1)
        return samples[0] if samples else None

    def poll_odometry(self):
        samples = self.r_odom.take(1)
        return samples[0] if samples else None

    def poll_persons(self):
        samples = self.r_persons.take(1)
        return samples[0] if samples else None

    def poll_orchestration_state(self):
        samples = self.r_orch_state.take(1)
        return samples[0] if samples else None

    # ==========================================================================
    # ORQUESTRAÇÃO E VISÃO
    # ==========================================================================

    def update_orchestration_state(self, state: OrchestratorState):
        previous_phase = self.current_phase
        previous_target = self.target_person_id

        self.current_phase = state.phase
        self.target_person_id = state.current_target_person
        self.orchestration_allows_navigation = bool(state.active_modules.navigation)

        if not self.orchestration_allows_navigation:
            self.navigation_active = False
            self.current_path = []
            self.current_waypoint_index = 0
            self.safe_stop("Orquestração desativou navigation")

        now = time.time()
        changed = previous_phase != self.current_phase or previous_target != self.target_person_id

        if changed or now - self.last_orch_log_time >= 2.0:
            log.info(
                "[ORCH] fase=%s | target_person=%s | navigation_active=%s",
                self.current_phase.name,
                self.target_person_id,
                state.active_modules.navigation,
            )
            self.last_orch_log_time = now

    def update_persons(self, persons: Persons):
        self.latest_persons = persons
        target = self.get_target_person_detection()

        if target is None:
            self.person_visible = False
            self.person_yaw_error = None
            return

        self.person_visible = True
        self.person_yaw_error = float(target.yaw)

        now = time.time()
        if now - self.last_vision_log_time >= 1.0:
            log.info(
                "[VISION] pessoa=%s | yaw_error=%.3f | confidence=%.2f",
                target.id,
                self.person_yaw_error,
                target.lip_movement_confidence,
            )
            self.last_vision_log_time = now

    def get_target_person_detection(self):
        if self.latest_persons is None:
            return None

        detections = list(self.latest_persons.detections)

        if not detections:
            return None

        if self.target_person_id:
            for person in detections:
                if person.id == self.target_person_id:
                    return person

        detections.sort(key=lambda p: p.lip_movement_confidence, reverse=True)
        return detections[0]

    def person_is_centered(self):
        if not self.person_visible:
            return False

        if self.person_yaw_error is None:
            return False

        return abs(self.person_yaw_error) <= PERSON_CENTER_TOLERANCE

    # ==========================================================================
    # ODOMETRIA
    # ==========================================================================

    def update_odometry(self, odom: OdometryMsg):
        self.current_pose = odom.pose
        self.last_odom_time = time.time()

        q = self.current_pose.orientation
        self.current_yaw = yaw_from_quaternion(q.x, q.y, q.z, q.w)

        cell_x, cell_y = self.current_robot_cell()
        self.slam.update_pose(cell_x, cell_y, self.current_yaw)

    def odometry_is_recent(self):
        if not ENABLE_MOTION:
            return True

        if self.last_odom_time <= 0.0:
            return False

        return (time.time() - self.last_odom_time) <= ODOM_TIMEOUT_S

    # ==========================================================================
    # LIDAR E MAPA
    # ==========================================================================

    def update_lidar_map(self):
        if not self.enable_lidar or self.lidar is None:
            return

        xyz = self.lidar.get_latest_points()

        if xyz is None:
            return

        curr_cell_x, curr_cell_y = self.current_robot_cell()

        try:
            obs, free = pointcloud_to_occupancy_points(
                xyz,
                curr_cell_x,
                curr_cell_y,
                self.current_yaw,
                map_size=self.slam.map_size,
                resolution=MAP_RESOLUTION,
                max_range_meters=LIDAR_MAX_RANGE_M,
                min_z=LIDAR_MIN_Z,
                max_z=LIDAR_MAX_Z,
                min_dist_m=LIDAR_MIN_DIST_M,
                point_step=LIDAR_POINT_STEP,
                apply_yaw=True,
            )
        except TypeError:
            # Compatibilidade com versões antigas do sensores.py sem apply_yaw.
            obs, free = pointcloud_to_occupancy_points(
                xyz,
                curr_cell_x,
                curr_cell_y,
                self.current_yaw,
                map_size=self.slam.map_size,
                resolution=MAP_RESOLUTION,
                max_range_meters=LIDAR_MAX_RANGE_M,
                min_z=LIDAR_MIN_Z,
                max_z=LIDAR_MAX_Z,
                min_dist_m=LIDAR_MIN_DIST_M,
                point_step=LIDAR_POINT_STEP,
            )

        self.slam.decay_map(decay_factor=MAP_DECAY_FACTOR)

        for pt in free:
            self.slam._update_cell(pt[0], pt[1], False)

        for pt in obs:
            self.slam._update_cell(pt[0], pt[1], True)

    # ==========================================================================
    # OBJETIVOS E PLANEAMENTO
    # ==========================================================================

    def get_goal_pose(self, goal: NavGoal):
        try:
            name = goal.data.name

            if name:
                name = name.strip().lower()

                if name in KNOWN_LOCATIONS:
                    x, y = KNOWN_LOCATIONS[name]
                    return self.pose_from_xy(x, y)

        except Exception:
            pass

        try:
            return goal.data.pose
        except Exception:
            pass

        return None

    def plan_path_to_cell(self, goal_cell):
        if goal_cell is None:
            return []
        return self.slam.plan_path(goal_cell, allow_unknown=True)

    def plan_path_to_pose(self, goal_pose: Pose):
        goal_cell = world_to_cell(goal_pose.position.x, goal_pose.position.y)
        return self.plan_path_to_cell(goal_cell)

    def set_goal_from_cell(self, goal_cell, obstacle_cell=None, reason="Objetivo local calculado"):
        self.current_goal_cell = goal_cell
        self.current_obstacle_cell = obstacle_cell

        if goal_cell is None:
            self.current_path = []
            self.current_waypoint_index = 0
            self.navigation_active = False
            self.safe_stop("Não foi possível calcular objetivo local", publish_status=True)
            return

        goal_x, goal_y = cell_to_world(goal_cell[0], goal_cell[1])
        self.current_goal_pose = self.pose_from_xy(goal_x, goal_y)

        self.current_path = self.plan_path_to_cell(goal_cell)
        self.current_waypoint_index = 0

        if not self.current_path:
            self.navigation_active = False
            self.safe_stop("Não foi possível planear caminho para objetivo local", publish_status=True)
            return

        self.navigation_active = True
        self.publish_path(self.current_path)
        self.publish_status(Status.RUNNING, reason, 0.0)

    def replan_from_orchestration_phase(self):
        if not self.enable_lidar:
            return

        now = time.time()

        if now - self.last_replan_time < REPLAN_INTERVAL_S and self.current_path:
            return

        robot_cell = self.current_robot_cell()

        if self.current_phase == Phase.NAVIGATING_TO_TABLE:
            goal_cell, obstacle_cell = find_nearest_front_obstacle_goal(
                slam=self.slam,
                robot_cell=robot_cell,
                robot_yaw=self.current_yaw,
                stop_distance_m=LOCAL_GOAL_STOP_DISTANCE_M,
                front_angle_deg=LOCAL_GOAL_FRONT_ANGLE_DEG,
                min_distance_m=LOCAL_GOAL_MIN_DISTANCE_M,
                max_distance_m=LOCAL_GOAL_MAX_DISTANCE_M,
                search_radius_m=LOCAL_GOAL_SEARCH_RADIUS_M,
            )

            self.set_goal_from_cell(
                goal_cell,
                obstacle_cell,
                reason="Objetivo local calculado para a mesa",
            )
            self.last_replan_time = now

        elif self.current_phase == Phase.NAVIGATING_TO_PERSON:
            if not self.person_visible:
                self.safe_stop("Pessoa alvo ainda não visível")
                return

            if not self.person_is_centered():
                wz = max(
                    -PERSON_MAX_ROT_SPEED,
                    min(PERSON_MAX_ROT_SPEED, PERSON_YAW_GAIN * self.person_yaw_error),
                )

                if ENABLE_MOTION:
                    self.send_cmd_vel(0.0, 0.0, wz)
                else:
                    self.safe_stop("Pessoa alvo visível, mas movimento desativado")

                self.publish_status(Status.RUNNING, "A centrar pessoa alvo", 0.3)
                return

            goal_cell, obstacle_cell = find_nearest_front_obstacle_goal(
                slam=self.slam,
                robot_cell=robot_cell,
                robot_yaw=self.current_yaw,
                stop_distance_m=PERSON_STOP_DISTANCE_M,
                front_angle_deg=PERSON_FRONT_ANGLE_DEG,
                min_distance_m=PERSON_MIN_DISTANCE_M,
                max_distance_m=PERSON_MAX_DISTANCE_M,
                search_radius_m=PERSON_SEARCH_RADIUS_M,
            )

            self.set_goal_from_cell(
                goal_cell,
                obstacle_cell,
                reason="Objetivo local calculado para a pessoa",
            )
            self.last_replan_time = now

    def handle_new_goal(self, goal: NavGoal):
        if not self.orchestration_allows_navigation:
            self.navigation_active = False
            self.current_path = []
            self.current_waypoint_index = 0
            self.safe_stop("Goal recebido, mas navigation está desativado pela orquestração")
            return

        self.current_goal = goal
        self.current_goal_pose = self.get_goal_pose(goal)
        self.current_goal_cell = None
        self.current_obstacle_cell = None

        log.info("[GOAL] recebido: %s", goal)

        if self.current_goal_pose is None:
            self.current_path = []
            self.current_waypoint_index = 0
            self.navigation_active = False
            self.safe_stop("Objetivo inválido", publish_status=True)
            return

        self.current_path = self.plan_path_to_pose(self.current_goal_pose)
        self.current_waypoint_index = 0

        if not self.current_path:
            self.navigation_active = False
            self.safe_stop("Não foi possível planear caminho", publish_status=True)
            return

        self.navigation_active = True
        self.publish_path(self.current_path)
        self.publish_status(Status.RUNNING, "Caminho planeado", 0.0)

    # ==========================================================================
    # EXECUÇÃO DA NAVEGAÇÃO
    # ==========================================================================

    def follow_path_step(self):
        if not self.current_path:
            self.safe_stop("Sem caminho para seguir")
            return 0.0

        if self.current_waypoint_index >= len(self.current_path):
            self.safe_stop("Índice do waypoint fora do caminho")
            return 1.0

        target_cell = self.current_path[self.current_waypoint_index]
        target_x, target_y = cell_to_world(target_cell[0], target_cell[1])

        robot_x = self.current_pose.position.x
        robot_y = self.current_pose.position.y

        dx = target_x - robot_x
        dy = target_y - robot_y

        dist = math.sqrt(dx * dx + dy * dy)

        if dist <= WAYPOINT_TOLERANCE_M:
            self.current_waypoint_index += 1

            if self.current_waypoint_index >= len(self.current_path):
                self.safe_stop("Último waypoint atingido")
                return 1.0

            target_cell = self.current_path[self.current_waypoint_index]
            target_x, target_y = cell_to_world(target_cell[0], target_cell[1])

            dx = target_x - robot_x
            dy = target_y - robot_y
            dist = math.sqrt(dx * dx + dy * dy)

        desired_yaw = math.atan2(dy, dx)
        yaw_error = normalize_angle(desired_yaw - self.current_yaw)

        if abs(yaw_error) > YAW_ALIGN_THRESHOLD:
            vx = 0.0
            vy = 0.0
            wz = max(-MAX_ANGULAR_SPEED, min(MAX_ANGULAR_SPEED, yaw_error))
        else:
            vx = min(MAX_LINEAR_SPEED, 0.5 * dist)
            vy = 0.0
            wz = max(-MAX_ANGULAR_SPEED, min(MAX_ANGULAR_SPEED, yaw_error))

        self.send_cmd_vel(vx, vy, wz)

        progress = self.current_waypoint_index / max(len(self.current_path), 1)
        return max(0.0, min(1.0, progress))

    def update_navigation(self):
        if not self.orchestration_allows_navigation:
            self.navigation_active = False
            self.safe_stop("Navigation desativado pela orquestração")
            return

        if not self.odometry_is_recent():
            self.navigation_active = False
            self.safe_stop("Odometria ausente ou desatualizada", publish_status=True)
            return

        if not self.navigation_active or self.current_goal_pose is None:
            self.safe_stop("Sem navegação ativa")
            return

        if not self.current_path:
            self.navigation_active = False
            self.safe_stop("Sem caminho ativo", publish_status=True)
            return

        dist = self.distance_to_goal(self.current_goal_pose)

        if dist <= GOAL_TOLERANCE_M:
            self.navigation_active = False
            self.current_path = []
            self.current_waypoint_index = 0
            self.safe_stop("Objetivo alcançado")
            self.publish_status(Status.DONE, "Objetivo alcançado", 1.0)
            return

        if not self.slam.is_path_valid(self.current_path):
            self.navigation_active = False
            self.current_path = []
            self.current_waypoint_index = 0
            self.safe_stop("Caminho ficou inválido", publish_status=True)
            return

        if not ENABLE_MOTION:
            self.safe_stop("Movimento desativado por segurança")
            progress = max(0.0, min(1.0, 1.0 - dist / 4.0))

            now = time.time()
            if now - self.last_status_time >= 0.5:
                self.publish_status(
                    Status.RUNNING,
                    "A navegar - movimento ainda desativado por segurança",
                    progress,
                )
                self.last_status_time = now

            return

        progress = self.follow_path_step()

        now = time.time()
        if now - self.last_status_time >= 0.5:
            self.publish_status(Status.RUNNING, "A navegar", progress)
            self.last_status_time = now

    # ==========================================================================
    # VISUALIZAÇÃO
    # ==========================================================================

    def update_visualization(self):
        if not self.enable_viz or self.visualizer is None:
            return

        self.visualizer.update(
            slam=self.slam,
            robot_cell=self.current_robot_cell(),
            yaw=self.current_yaw,
            path=self.current_path,
            goal_cell=self.current_goal_cell,
            obstacle_cell=self.current_obstacle_cell,
        )

    # ==========================================================================
    # LOOP PRINCIPAL
    # ==========================================================================

    def run(self):
        log.info("SLAM + Navegação a iniciar no domínio %d", DOMAIN_ID)
        log.info("Modo LiDAR=%s | Visualização=%s | ENABLE_MOTION=%s", self.enable_lidar, self.enable_viz, ENABLE_MOTION)

        self.publish_locations()
        self.publish_status(Status.DONE, "Módulo de navegação iniciado", 0.0)
        self.safe_stop("Arranque do módulo")

        try:
            while True:
                now = time.time()

                odom = self.poll_odometry()
                if odom:
                    self.update_odometry(odom)

                orch_state = self.poll_orchestration_state()
                if orch_state:
                    self.update_orchestration_state(orch_state)

                persons = self.poll_persons()
                if persons:
                    self.update_persons(persons)

                self.update_lidar_map()
                self.replan_from_orchestration_phase()

                self.publish_pose()

                if now - self.last_locations_time >= 5.0:
                    self.publish_locations()
                    self.last_locations_time = now

                goal = self.poll_goal()
                if goal:
                    self.handle_new_goal(goal)

                self.update_navigation()
                self.update_visualization()

                time.sleep(LOOP_DT)

        except KeyboardInterrupt:
            log.info("Encerramento manual. A parar robô.")
            self.safe_stop("Encerramento manual")

        except Exception as e:
            log.exception("Erro inesperado no módulo de navegação: %s", e)
            self.safe_stop(f"Erro inesperado: {e}", publish_status=True)
            raise

        finally:
            self.safe_stop("Saída do programa")


def parse_args():
    parser = argparse.ArgumentParser(description="Módulo SLAM/Navegação com DDS, LiDAR e visualização opcional.")
    parser.add_argument("--lidar", action="store_true", help="Ativa leitura do Livox MID-360.")
    parser.add_argument("--viz", action="store_true", help="Mostra janela Matplotlib com occupancy grid.")
    parser.add_argument("--host-ip", default="192.168.123.165", help="IP usado pelo LivoxReceiver.")
    parser.add_argument("--config-path", default="mid360_config.json", help="Caminho para o ficheiro de configuração do Livox.")
    return parser.parse_args()


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] NAVIGATION: %(message)s",
    )

    args = parse_args()

    NavigationModule(
        enable_lidar=args.lidar,
        enable_viz=args.viz,
        host_ip=args.host_ip,
        config_path=args.config_path,
    ).run()
