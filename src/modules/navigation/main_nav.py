import argparse
import os
import sys
import time
import math
import logging

import numpy as np
from sklearn.cluster import DBSCAN

pasta_atual = os.path.dirname(os.path.abspath(__file__))
pasta_src = os.path.abspath(os.path.join(pasta_atual, "../.."))
sys.path.append(pasta_src)

from cyclonedds.domain import DomainParticipant
from cyclonedds.topic import Topic
from cyclonedds.pub import Publisher, DataWriter
from cyclonedds.sub import Subscriber, DataReader

from idl_ri import (
    Header, Status, Vector3, Quaternion, Pose,
    SlamPoseMsg, NavGoal, NavStatusMsg, NavPath,
    CmdVel, OdometryMsg, Locations, Location, GoalType
)
from qos_profiles import (
    QOS_SLAM_POSE, QOS_NAV, QOS_MOTION, QOS_ODOMETRY, QOS_SLAM_MAP
)

from slam_navigation import SLAMNavigation
from sensores import pointcloud_to_occupancy_points
from navigation import (
    MAP_RESOLUTION,
    world_to_cell, cell_to_world,
    yaw_from_quaternion, quaternion_from_yaw,
    normalize_angle,
)

try:
    from map_visualizer import MapVisualizer
except Exception as e:
    print(f"[AVISO] MapVisualizer não disponível: {e}")
    MapVisualizer = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] NAV: %(message)s")
log = logging.getLogger("navigation")

DOMAIN_ID        = 0
MAP_SIZE         = 200
LOOP_DT          = 0.02

MAX_LINEAR_SPEED     = 0.20
MAX_ANGULAR_SPEED    = 0.30
YAW_ALIGN_THRESHOLD  = 0.35
ODOM_TIMEOUT_S       = 1.0

LIDAR_MAX_RANGE_M = 4.0
LIDAR_MIN_Z       = -0.80
LIDAR_MAX_Z       =  2.0
LIDAR_MIN_DIST_M  =  0.35
LIDAR_POINT_STEP  = 5
MAP_DECAY_FACTOR  = 0.97
FRONTAL_CONE_ANGLE_DEG = 25.0

DBSCAN_EPS         = 0.25
DBSCAN_MIN_SAMPLES = 5

PERSON_CLUSTER_MIN_POINTS = 25
PERSON_MAX_WIDTH  = 0.8
PERSON_MIN_HEIGHT = 1.0

TABLE_CLUSTER_MIN_POINTS = 30
TABLE_MIN_WIDTH  = 0.15
TABLE_MAX_WIDTH  = 1.5
TABLE_MIN_DEPTH  = 0.00
TABLE_MIN_HEIGHT = 0.10
NOISE_MAX_POINTS = 10


class NavigationModule:

    def __init__(self, enable_lidar=False, enable_viz=False,
                 host_ip="192.168.123.165", config_path="mid360_config.json"):

        self.seq = 0
        self.enable_lidar = enable_lidar
        self.enable_viz   = enable_viz

        self.current_pose = Pose()
        self.current_yaw          = 0.0
        self.current_goal_cell    = None
        self.current_obstacle_cell = None
        self.current_path         = []
        
        # Gestão de Objetivos
        self.navigation_active    = False
        self.active_goal_name     = None
        
        self.last_odom_time       = 0.0
        self.detected_objects     = []
        self.tracked_objects      = []
        self.last_safe_stop_reason    = ""
        self.last_safe_stop_log_time  = 0.0
        self.available_colors = [
            "cyan", "yellow", "orange", "purple",
            "lime", "deepskyblue", "magenta", "gold"
        ]

        self.slam = SLAMNavigation(
            map_size=MAP_SIZE,
            resolution=MAP_RESOLUTION,
            num_rays=144,
            max_range=int(LIDAR_MAX_RANGE_M / MAP_RESOLUTION),
        )

        self.lidar = None
        if enable_lidar:
            try:
                from livox_receiver import LivoxReceiver
                self.lidar = LivoxReceiver(config_path=config_path, host_ip=host_ip)
                log.info("LiDAR iniciado.")
            except Exception as e:
                log.error("LiDAR falhou: %s", e)
                self.enable_lidar = False

        self.visualizer = None
        if enable_viz:
            if MapVisualizer is None:
                raise RuntimeError("MapVisualizer não encontrado.")
            self.visualizer = MapVisualizer()

        # ─── DDS Setup ────────────────────────────────────────────────────────
        self.dp  = DomainParticipant(DOMAIN_ID)
        pub      = Publisher(self.dp)
        sub      = Subscriber(self.dp)

        self.w_pose    = DataWriter(pub, Topic(self.dp, "rt/slam/pose",       SlamPoseMsg,  qos=QOS_SLAM_POSE))
        self.w_cmd_vel = DataWriter(pub, Topic(self.dp, "rt/motion/cmd_vel",  CmdVel,       qos=QOS_MOTION))
        self.w_status  = DataWriter(pub, Topic(self.dp, "rt/nav/status",      NavStatusMsg, qos=QOS_NAV))
        self.w_path    = DataWriter(pub, Topic(self.dp, "rt/nav/path",        NavPath,      qos=QOS_NAV))
        self.w_locs    = DataWriter(pub, Topic(self.dp, "rt/slam/locations",  Locations,    qos=QOS_SLAM_MAP))

        self.r_odom    = DataReader(sub, Topic(self.dp, "rt/motion/odometry", OdometryMsg,  qos=QOS_ODOMETRY))
        self.r_goal    = DataReader(sub, Topic(self.dp, "rt/nav/goal",        NavGoal,      qos=QOS_NAV))

        log.info("DDS configurado.")

    def header(self, frame_id="nav"):
        self.seq += 1
        return Header(timestamp_ns=time.time_ns(), frame_id=frame_id, seq=self.seq)

    def current_robot_cell(self):
        return world_to_cell(self.current_pose.position.x, self.current_pose.position.y)

    def pose_from_xy(self, x, y, yaw=0.0):
        qx, qy, qz, qw = quaternion_from_yaw(yaw)
        return Pose(
            position=Vector3(x=x, y=y, z=0.0),
            orientation=Quaternion(x=qx, y=qy, z=qz, w=qw),
        )

    def odometry_is_recent(self):
        if self.last_odom_time <= 0.0:
            return False
        return (time.time() - self.last_odom_time) <= ODOM_TIMEOUT_S

    def poll_odometry(self):
        samples = self.r_odom.take(1)
        return samples[0] if samples else None

    def poll_goal(self):
        """Lê os objetivos enviados pelo Orquestrador."""
        samples = self.r_goal.take()
        for sample in samples:
            if sample and sample.data._d == GoalType.NAMED:
                self.active_goal_name = sample.data.name
                self.navigation_active = True
                self.publish_status(Status.RUNNING, f"A navegar para {self.active_goal_name}")
                log.info("Novo objetivo recebido: '%s'", self.active_goal_name)

    def update_odometry(self, odom):
        self.current_pose   = odom.pose
        self.last_odom_time = time.time()
        q = self.current_pose.orientation
        self.current_yaw = yaw_from_quaternion(q.x, q.y, q.z, q.w)
        cell_x, cell_y = self.current_robot_cell()
        self.slam.update_pose(cell_x, cell_y, self.current_yaw)

    def publish_pose(self):
        self.w_pose.write(SlamPoseMsg(header=self.header("slam"), pose=self.current_pose))

    def publish_status(self, status, reason="", progress=0.0):
        self.w_status.write(NavStatusMsg(
            header=self.header("nav"), status=status,
            reason=reason, progress=progress,
        ))

    def publish_path(self, path_cells):
        waypoints = [self.pose_from_xy(*cell_to_world(cx, cy)) for cx, cy in path_cells]
        self.w_path.write(NavPath(header=self.header("nav"), waypoints=waypoints))

    def publish_locations(self):
        """Partilha as localizações encontradas com o Orquestrador."""
        locs = []
        for obj in self.tracked_objects:
            # Para a mesa, passamos sempre 'table'. Para pessoas, passamos o ID.
            name = "table" if obj["class"] == "table" else obj["label"]
            px, py = obj["centroid"][0], obj["centroid"][1]
            locs.append(Location(name=name, pose=self.pose_from_xy(px, py)))
            
        self.w_locs.write(Locations(header=self.header("slam"), locations=locs))

    def send_cmd_vel(self, vx=0.0, vy=0.0, wz=0.0):
        # Apenas publica velocidades SE a navegação estiver ativamente a tentar chegar a um alvo
        if self.navigation_active:
            self.w_cmd_vel.write(CmdVel(header=self.header("nav"), vx=vx, vy=vy, wz=wz))

    def stop_robot(self):
        self.send_cmd_vel(0.0, 0.0, 0.0)

    def safe_stop(self, reason="safe stop"):
        self.stop_robot()
        now = time.time()
        if reason != self.last_safe_stop_reason or now - self.last_safe_stop_log_time >= 0.5:
            log.warning("[SAFE STOP] %s", reason)
            self.last_safe_stop_reason    = reason
            self.last_safe_stop_log_time  = now

    def update_lidar_map(self):
        if not self.lidar:
            return
        xyz = self.lidar.get_latest_points()
        if xyz is None:
            return
        curr_cell_x, curr_cell_y = self.current_robot_cell()
        try:
            obs, free = pointcloud_to_occupancy_points(
                xyz, curr_cell_x, curr_cell_y, self.current_yaw,
                map_size=self.slam.map_size, resolution=MAP_RESOLUTION,
                max_range_meters=LIDAR_MAX_RANGE_M,
                min_z=LIDAR_MIN_Z, max_z=LIDAR_MAX_Z,
                min_dist_m=LIDAR_MIN_DIST_M, point_step=LIDAR_POINT_STEP,
                apply_yaw=True,
            )
        except TypeError:
            obs, free = pointcloud_to_occupancy_points(
                xyz, curr_cell_x, curr_cell_y, self.current_yaw,
                map_size=self.slam.map_size, resolution=MAP_RESOLUTION,
                max_range_meters=LIDAR_MAX_RANGE_M,
                min_z=LIDAR_MIN_Z, max_z=LIDAR_MAX_Z,
                min_dist_m=LIDAR_MIN_DIST_M, point_step=LIDAR_POINT_STEP,
            )
        self.slam.decay_map(decay_factor=MAP_DECAY_FACTOR)
        for pt in free:
            self.slam._update_cell(pt[0], pt[1], False)
        for pt in obs:
            self.slam._update_cell(pt[0], pt[1], True)
        self.detect_objects(xyz)

    def cluster_lidar_points(self, xyz):
        if xyz is None:
            return []
        points = np.array(xyz)
        if len(points) == 0:
            return []

        rx = self.current_pose.position.x
        ry = self.current_pose.position.y
        yaw = self.current_yaw

        dx = points[:, 0] - rx
        dy = points[:, 1] - ry
        
        dist_xy = np.sqrt(dx**2 + dy**2)
        angles = np.arctan2(dy, dx)
        angle_diff = np.mod(angles - yaw + np.pi, 2 * np.pi) - np.pi

        half_cone = math.radians(FRONTAL_CONE_ANGLE_DEG / 2.0)
        
        mask = (
            (points[:, 2] > LIDAR_MIN_Z) &
            (points[:, 2] < LIDAR_MAX_Z) &
            (dist_xy >= LIDAR_MIN_DIST_M) & 
            (np.abs(angle_diff) <= half_cone) 
        )

        points = points[mask]
        if len(points) == 0:
            return []

        db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(points[:, :2])
        labels = db.labels_
        return [points[labels == label] for label in set(labels) if label != -1]

    def extract_cluster_features(self, cluster):
        min_xyz  = np.min(cluster, axis=0)
        max_xyz  = np.max(cluster, axis=0)
        centroid = np.mean(cluster, axis=0)
        return {
            "points":   len(cluster),
            "width":    max_xyz[0] - min_xyz[0],
            "depth":    max_xyz[1] - min_xyz[1],
            "height":   max_xyz[2] - min_xyz[2],
            "centroid": centroid,
        }

    def classify_cluster(self, f):
        pts    = f["points"]
        width  = f["width"]
        depth  = f["depth"]
        height = f["height"]
        if pts < NOISE_MAX_POINTS or width > 1.2 or depth > 1.2:
            return "noise"
        if (pts >= PERSON_CLUSTER_MIN_POINTS
                and width <= PERSON_MAX_WIDTH
                and depth <= PERSON_MAX_WIDTH
                and height >= PERSON_MIN_HEIGHT):
            return "person"
        if (pts >= TABLE_CLUSTER_MIN_POINTS
                and TABLE_MIN_WIDTH <= width <= TABLE_MAX_WIDTH
                and depth >= TABLE_MIN_DEPTH
                and height >= TABLE_MIN_HEIGHT):
            return "table"
        return "unknown"

    def detect_objects(self, xyz):
        if xyz is None or len(xyz) == 0:
            self.detected_objects = []
            self.tracked_objects  = []
            return

        points   = np.array(xyz)
        yaw      = self.current_yaw
        rx       = self.current_pose.position.x
        ry       = self.current_pose.position.y
        cos_y    = math.cos(yaw)
        sin_y    = math.sin(yaw)

        global_xyz          = np.zeros_like(points)
        global_xyz[:, 0]    = rx + (points[:, 0] * cos_y - points[:, 1] * sin_y)
        global_xyz[:, 1]    = ry + (points[:, 0] * sin_y + points[:, 1] * cos_y)
        global_xyz[:, 2]    = points[:, 2]

        clusters        = self.cluster_lidar_points(global_xyz)
        detected        = []
        current_tracked = []

        for cluster in clusters:
            f   = self.extract_cluster_features(cluster)
            cls = self.classify_cluster(f)
            if cls == "noise":
                continue

            centroid   = f["centroid"][:2]
            best_match = None
            best_dist  = 0.55

            for old in self.tracked_objects:
                dist = math.sqrt((centroid[0] - old["centroid"][0]) ** 2 + (centroid[1] - old["centroid"][1]) ** 2)
                if dist < best_dist:
                    best_dist  = dist
                    best_match = old

            if best_match is not None:
                color      = best_match["color"]
                label_name = best_match["label"]
                outro_id   = best_match.get("outro_id")
            else:
                if cls == "table":
                    color      = "green"
                    label_name = "Mesa"
                    outro_id   = None
                elif cls == "person":
                    color      = "red"
                    label_name = "Pessoa"
                    outro_id   = None
                else:
                    used_ids     = {o.get("outro_id") for o in current_tracked + self.tracked_objects if o.get("outro_id") is not None}
                    candidate_id = 1
                    while candidate_id in used_ids:
                        candidate_id += 1
                    outro_id   = candidate_id
                    label_name = f"Outro {outro_id}"
                    color      = self.available_colors[(outro_id - 1) % len(self.available_colors)]

            obj_data = {
                "class":    cls,
                "features": f,
                "cluster":  cluster,
                "centroid": centroid,
                "color":    color,
                "label":    label_name,
                "outro_id": outro_id,
            }
            detected.append(obj_data)
            current_tracked.append(obj_data)

        self.detected_objects = detected
        self.tracked_objects  = current_tracked

    def replan(self):
        # Se a navegação não estiver ativa ou não houver objetivo, ignoramos os cálculos de motor.
        # Isto previne que o Nav lute contra a Orquestração na fase de recuo!
        if not self.navigation_active or not self.active_goal_name:
            return 

        # 1. Procurar o alvo pedido pelo Orquestrador na nossa lista de objetos rastreados
        target = None
        for obj in self.tracked_objects:
            if self.active_goal_name == "table" and obj["class"] == "table":
                target = obj
                break
            elif obj["label"] == self.active_goal_name:
                target = obj
                break

        if target is None:
            self.safe_stop(f"À procura do alvo '{self.active_goal_name}'...")
            return

        cx = target["centroid"][0]
        cy = target["centroid"][1]
        
        rx = self.current_pose.position.x
        ry = self.current_pose.position.y
        
        dx = cx - rx
        dy = cy - ry
        dist_to_target = math.sqrt(dx ** 2 + dy ** 2)

        # 2. Offset dinâmico baseado no tipo de alvo!
        if target["class"] == "table":
            distancia_paragem = 0.70  # Robô para a 70cm da mesa
        else:
            distancia_paragem = 0.85  # Robô para a 85cm da pessoa para não a atropelar

        # 3. Lógica Direta de Movimento
        if dist_to_target <= distancia_paragem:
            # Chegou ao objetivo
            self.stop_robot()
            self.navigation_active = False # Desliga-se a si próprio
            
            # Avisa o Orquestrador que chegou
            self.publish_status(Status.DONE, f"{self.active_goal_name} alcançado!", 1.0)
            log.info("[SUCESSO] Distância atingida! Motores a 0. (%s)", self.active_goal_name)
            
            self.active_goal_name = None 
        else:
            # Ainda está longe, avança
            desired_yaw = math.atan2(dy, dx)
            yaw_error   = normalize_angle(desired_yaw - self.current_yaw)

            if abs(yaw_error) > YAW_ALIGN_THRESHOLD:
                vx = 0.0
            else:
                vx = MAX_LINEAR_SPEED
                
            wz = max(-MAX_ANGULAR_SPEED, min(MAX_ANGULAR_SPEED, yaw_error))
            
            self.send_cmd_vel(vx, 0.0, wz)

        # Atualizar Visualização
        goal_cell = world_to_cell(cx, cy)
        self.current_goal_cell = goal_cell
        self.current_path = [self.current_robot_cell(), goal_cell]
        self.publish_path(self.current_path)

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
            detected_objects=self.detected_objects,
        )

    def run(self):
        log.info("Navegação iniciada. LiDAR=%s | Viz=%s", self.enable_lidar, self.enable_viz)

        try:
            while True:
                # 1. Lê ordens do Orquestrador
                self.poll_goal()
                
                # 2. Lê a posição do robô
                odom = self.poll_odometry()
                if odom:
                    self.update_odometry(odom)
                else:
                    if not self.odometry_is_recent():
                        log.warning("A aguardar odometria do motion...")

                # 3. Processa Visão e partilha dados com o Orquestrador
                self.update_lidar_map()
                self.publish_locations()
                
                # 4. Planeia o caminho SE houver um objetivo ativo
                self.replan()
                
                self.publish_pose()
                self.update_visualization()

                time.sleep(LOOP_DT)

        except KeyboardInterrupt:
            log.info("Encerramento manual.")
            self.safe_stop("Encerramento manual")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lidar",       action="store_true")
    parser.add_argument("--viz",         action="store_true")
    parser.add_argument("--host-ip",     default="192.168.123.51")
    parser.add_argument("--config-path", default="mid360_config.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    NavigationModule(
        enable_lidar=args.lidar,
        enable_viz=args.viz,
        host_ip=args.host_ip,
        config_path=args.config_path,
    ).run()