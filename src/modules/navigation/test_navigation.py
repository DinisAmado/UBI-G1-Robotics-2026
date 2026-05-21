#!/usr/bin/env python3

import os
import sys
import time

pasta_atual = os.path.dirname(os.path.abspath(__file__))
pasta_src = os.path.abspath(os.path.join(pasta_atual, "../.."))
sys.path.append(pasta_src)

from cyclonedds.domain import DomainParticipant
from cyclonedds.topic import Topic
from cyclonedds.sub import Subscriber, DataReader

from idl_ri import SlamPoseMsg, Locations, NavStatusMsg, NavPath, CmdVel
from qos_profiles import QOS_SLAM_POSE, QOS_SLAM_MAP, QOS_NAV, QOS_MOTION


dp = DomainParticipant(0)
sub = Subscriber(dp)

t_pose = Topic(dp, "rt/slam/pose", SlamPoseMsg, qos=QOS_SLAM_POSE)
t_locations = Topic(dp, "rt/slam/locations", Locations, qos=QOS_SLAM_MAP)
t_status = Topic(dp, "rt/nav/status", NavStatusMsg, qos=QOS_NAV)
t_path = Topic(dp, "rt/nav/path", NavPath, qos=QOS_NAV)
t_cmd_vel = Topic(dp, "rt/motion/cmd_vel", CmdVel, qos=QOS_MOTION)

r_pose = DataReader(sub, t_pose)
r_locations = DataReader(sub, t_locations)
r_status = DataReader(sub, t_status)
r_path = DataReader(sub, t_path)
r_cmd_vel = DataReader(sub, t_cmd_vel)

print("A ouvir tópicos publicados pelo módulo navigation...")

while True:
    pose_samples = r_pose.take(10)
    for msg in pose_samples:
        if not hasattr(msg, "pose"):
            continue

        print(
            f"[POSE] x={msg.pose.position.x:.2f} "
            f"y={msg.pose.position.y:.2f} "
            f"seq={msg.header.seq}"
        )

    loc_samples = r_locations.take(10)
    for msg in loc_samples:
        if not hasattr(msg, "locations"):
            continue

        print(f"[LOCATIONS] n={len(msg.locations)} seq={msg.header.seq}")

        for loc in msg.locations:
            print(
                f"  - {loc.name}: "
                f"({loc.pose.position.x:.2f}, {loc.pose.position.y:.2f})"
            )

    status_samples = r_status.take(10)
    for msg in status_samples:
        if not hasattr(msg, "status"):
            continue

        print(
            f"[STATUS] {msg.status.name} | "
            f"{msg.reason} | progress={msg.progress:.2f}"
        )

    path_samples = r_path.take(10)
    for msg in path_samples:
        if not hasattr(msg, "waypoints"):
            continue

        print(f"[PATH] waypoints={len(msg.waypoints)} seq={msg.header.seq}")

    cmd_samples = r_cmd_vel.take(10)
    for msg in cmd_samples:
        if not hasattr(msg, "vx"):
            continue

        print(
            f"[CMD_VEL] vx={msg.vx:.2f} "
            f"vy={msg.vy:.2f} "
            f"wz={msg.wz:.2f}"
        )

    time.sleep(0.05)
