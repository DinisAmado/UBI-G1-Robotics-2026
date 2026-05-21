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
    pose_samples = r_pose.take(1)
    if pose_samples:
        msg = pose_samples[0]
        print(f"[POSE] x={msg.pose.position.x:.2f} y={msg.pose.position.y:.2f} seq={msg.header.seq}")

    loc_samples = r_locations.take(1)
    if loc_samples:
        msg = loc_samples[0]
        print(f"[LOCATIONS] n={len(msg.locations)} seq={msg.header.seq}")

    status_samples = r_status.take(1)
    if status_samples:
        msg = status_samples[0]
        print(f"[STATUS] {msg.status.name} | {msg.reason} | progress={msg.progress:.2f}")

    path_samples = r_path.take(1)
    if path_samples:
        msg = path_samples[0]
        print(f"[PATH] waypoints={len(msg.waypoints)} seq={msg.header.seq}")

    cmd_samples = r_cmd_vel.take(1)
    if cmd_samples:
        msg = cmd_samples[0]
        print(f"[CMD_VEL] vx={msg.vx:.2f} vy={msg.vy:.2f} wz={msg.wz:.2f}")

    time.sleep(0.05)
