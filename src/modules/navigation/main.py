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
    rt/nav/goal         (NavGoal)
    rt/motion/odometry  (OdometryMsg)

Nota:
    Este ficheiro é apenas para integração DDS.
    O demo.py continua a ser usado para testes com LiDAR, mapa e visualização.
"""

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

from navigation import (
    MAP_RESOLUTION,
    MAP_ORIGIN_X,
    MAP_ORIGIN_Y,
    world_to_cell,
    cell_to_world,
    yaw_from_quaternion,
    quaternion_from_yaw,
)


# ==============================================================================
# CONFIGURAÇÃO
# ==============================================================================

DOMAIN_ID = 0
log = logging.getLogger("navigation")

MAP_SIZE = 200
LOOP_DT = 0.02  # 50 Hz

# Segurança:
# False -> publica cmd_vel sempre a zero.
# True  -> permite ativar movimento simples no futuro.
ENABLE_MOTION = False

GOAL_TOLERANCE_M = 0.25

KNOWN_LOCATIONS = {
    "inicio": (0.0, 0.0),
    "mesa": (2.0, 0.0),
    "table": (2.0, 0.0),
    "pessoa_1": (3.0, -1.0),
    "pessoa_2": (3.0, 0.0),
    "pessoa_3": (3.0, 1.0),
}

PERSON_CENTER_TOLERANCE = 0.15
PERSON_STOP_DISTANCE_M = 0.45
PERSON_YAW_GAIN = 0.4
PERSON_MAX_ROT_SPEED = 0.30

# ==============================================================================
# CLASSE PRINCIPAL
# ==============================================================================

class NavigationModule:
    def __init__(self):
        self.seq = 0

        self.current_pose = Pose(
            position=Vector3(x=0.0, y=0.0, z=0.0),
            orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
        )

        self.current_yaw = 0.0
        self.current_goal = None
        self.current_goal_pose = None
        self.current_path = []
        self.navigation_active = False

        self.current_phase = Phase.IDLE
        self.target_person_id = ""
        self.latest_persons = None
        self.person_visible = False
        self.person_yaw_error = None

        self.last_locations_time = 0.0
        self.last_status_time = 0.0

        self.slam = SLAMNavigation(
            map_size=MAP_SIZE,
            resolution=MAP_RESOLUTION,
            num_rays=144,
            max_range=int(4.0 / MAP_RESOLUTION),
        )

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
        t_persons = Topic(
            self.dp,
            "rt/vision/persons",
            Persons,
            qos=QOS_VISION,
        )
        
        self.r_persons = DataReader(sub, t_persons)
        
        # Orquestração — estado global
        t_orch_state = Topic(
            self.dp,
            "rt/orchestration/state",
            OrchestratorState,
            qos=QOS_ORCHESTRATION,
        )
        
        self.r_orch_state = DataReader(sub, t_orch_state)
    # ==========================================================================
    # HELPERS
    # ==========================================================================

    def header(self, frame_id="nav") -> Header:
        self.seq += 1
        return Header(
            timestamp_ns=time.time_ns(),
            frame_id=frame_id,
            seq=self.seq,
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
        msg = SlamPoseMsg(
            header=self.header("slam"),
            pose=self.current_pose,
        )
        self.w_pose.write(msg)

    def publish_locations(self):
        locs = []

        for name, (x, y) in KNOWN_LOCATIONS.items():
            locs.append(
                Location(
                    name=name,
                    pose=self.pose_from_xy(x, y),
                )
            )

        msg = Locations(
            header=self.header("slam"),
            locations=locs,
        )

        self.w_locations.write(msg)

    def publish_status(self, status: Status, reason="", progress=0.0):
        msg = NavStatusMsg(
            header=self.header("nav"),
            status=status,
            reason=reason,
            progress=progress,
        )

        self.w_status.write(msg)
        log.info("[STATUS] %s | %s | %.2f", status.name, reason, progress)

    def publish_path(self, path_cells):
        waypoints = []

        for cell_x, cell_y in path_cells:
            x, y = cell_to_world(cell_x, cell_y)
            waypoints.append(self.pose_from_xy(x, y))

        msg = NavPath(
            header=self.header("nav"),
            waypoints=waypoints,
        )

        self.w_path.write(msg)

    def send_cmd_vel(self, vx=0.0, vy=0.0, wz=0.0):
        msg = CmdVel(
            header=self.header("nav"),
            vx=vx,
            vy=vy,
            wz=wz,
        )

        self.w_cmd_vel.write(msg)

    def stop_robot(self):
        self.send_cmd_vel(0.0, 0.0, 0.0)

    # ==========================================================================
    # LEITURAS
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

    def update_orchestration_state(self, state: OrchestratorState):
    """
    Atualiza a fase atual e a pessoa alvo vinda da orquestração.
    """

    self.current_phase = state.phase
    self.target_person_id = state.current_target_person

    log.info(
        "[ORCH] fase=%s | target_person=%s",
        self.current_phase.name,
        self.target_person_id,
    )


    def update_persons(self, persons: Persons):
        """
        Guarda as deteções de pessoas vindas da visão.
        """
    
        self.latest_persons = persons
    
        target = self.get_target_person_detection()
    
        if target is None:
            self.person_visible = False
            self.person_yaw_error = None
            return
    
        self.person_visible = True
        self.person_yaw_error = float(target.yaw)
    
        log.info(
            "[VISION] pessoa=%s | yaw_error=%.3f | confidence=%.2f",
            target.id,
            self.person_yaw_error,
            target.lip_movement_confidence,
        )
    
    
    def get_target_person_detection(self):
        """
        Devolve a deteção da pessoa alvo.
    
        Se a orquestração indicar current_target_person, procura esse id.
        Se não houver id definido, escolhe a pessoa com maior lip_movement_confidence.
        """
    
        if self.latest_persons is None:
            return None
    
        detections = list(self.latest_persons.detections)
    
        if not detections:
            return None
    
        if self.target_person_id:
            for person in detections:
                if person.id == self.target_person_id:
                    return person
    
        # fallback: escolher a pessoa com maior confiança
        detections.sort(key=lambda p: p.lip_movement_confidence, reverse=True)
        return detections[0]
    
    
    def person_is_centered(self):
        """
        Verifica se a pessoa está centrada usando o yaw publicado pela visão.
        """
    
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

        q = self.current_pose.orientation
        self.current_yaw = yaw_from_quaternion(q.x, q.y, q.z, q.w)

        cell_x, cell_y = world_to_cell(
            self.current_pose.position.x,
            self.current_pose.position.y,
        )

        self.slam.update_pose(cell_x, cell_y, self.current_yaw)

    # ==========================================================================
    # OBJETIVOS E PLANEAMENTO
    # ==========================================================================

    def get_goal_pose(self, goal: NavGoal):
        """
        Converte NavGoal para Pose.

        Se o goal for NAMED, procura em KNOWN_LOCATIONS.
        Se for POSE, usa a pose enviada.
        """

        # Tentar objetivo nomeado
        try:
            name = goal.data.name

            if name:
                name = name.strip().lower()

                if name in KNOWN_LOCATIONS:
                    x, y = KNOWN_LOCATIONS[name]
                    return self.pose_from_xy(x, y)

        except Exception:
            pass

        # Tentar objetivo por pose
        try:
            return goal.data.pose
        except Exception:
            pass

        return None

    def plan_path(self, goal_pose: Pose):
        goal_cell = world_to_cell(
            goal_pose.position.x,
            goal_pose.position.y,
        )

        return self.slam.plan_path(goal_cell, allow_unknown=True)

    def handle_new_goal(self, goal: NavGoal):
        self.current_goal = goal
        self.current_goal_pose = self.get_goal_pose(goal)

        log.info("[GOAL] recebido: %s", goal)

        if self.current_goal_pose is None:
            self.current_path = []
            self.navigation_active = False
            self.stop_robot()
            self.publish_status(Status.FAILED, "Objetivo inválido", 0.0)
            return

        self.current_path = self.plan_path(self.current_goal_pose)

        if not self.current_path:
            self.navigation_active = False
            self.stop_robot()
            self.publish_status(Status.FAILED, "Não foi possível planear caminho", 0.0)
            return

        self.navigation_active = True
        self.publish_path(self.current_path)
        self.publish_status(Status.RUNNING, "Caminho planeado", 0.0)

    # ==========================================================================
    # EXECUÇÃO DA NAVEGAÇÃO
    # ==========================================================================

    def update_navigation(self):
        if not self.navigation_active or self.current_goal_pose is None:
            self.stop_robot()
            return

        dist = self.distance_to_goal(self.current_goal_pose)

        if dist <= GOAL_TOLERANCE_M:
            self.navigation_active = False
            self.current_path = []
            self.stop_robot()
            self.publish_status(Status.DONE, "Objetivo alcançado", 1.0)
            return

        if not self.slam.is_path_valid(self.current_path):
            self.navigation_active = False
            self.stop_robot()
            self.publish_status(Status.FAILED, "Caminho ficou inválido", 0.0)
            return

        # Por segurança, ainda não enviamos velocidades reais.
        # O motion só recebe zeros até ENABLE_MOTION=True.
        if not ENABLE_MOTION:
            self.stop_robot()
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

        # Futuramente: aqui entra o path following real.
        self.stop_robot()

    # ==========================================================================
    # LOOP PRINCIPAL
    # ==========================================================================

    def run(self):
        log.info("SLAM + Navegação a iniciar no domínio %d", DOMAIN_ID)

        self.publish_locations()
        self.publish_status(Status.DONE, "Módulo de navegação iniciado", 0.0)
        self.stop_robot()

        try:
            while True:
                now = time.time()

                # 1. Ler odometria do motion
                odom = self.poll_odometry()
                if odom:
                    self.update_odometry(odom)

                # Ler estado da orquestração
                orch_state = self.poll_orchestration_state()
                if orch_state:
                    self.update_orchestration_state(orch_state)
                
                # Ler deteções de pessoas da visão
                persons = self.poll_persons()
                if persons:
                    self.update_persons(persons)

                # 2. Publicar pose SLAM a 50 Hz
                self.publish_pose()

                # 3. Publicar localizações periodicamente
                if now - self.last_locations_time >= 5.0:
                    self.publish_locations()
                    self.last_locations_time = now

                # 4. Ler novo objetivo
                goal = self.poll_goal()
                if goal:
                    self.handle_new_goal(goal)

                # 5. Atualizar navegação
                self.update_navigation()

                time.sleep(LOOP_DT)

        except KeyboardInterrupt:
            log.info("Encerramento manual. A parar robô.")
            self.stop_robot()


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] NAVIGATION: %(message)s",
    )

    NavigationModule().run()
