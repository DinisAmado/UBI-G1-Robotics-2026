#!/usr/bin/env python3

import os
import sys
import time
import math

pasta_atual = os.path.dirname(os.path.abspath(__file__))
pasta_src = os.path.abspath(os.path.join(pasta_atual, "../.."))
sys.path.append(pasta_src)

from cyclonedds.domain import DomainParticipant
from cyclonedds.topic import Topic
from cyclonedds.pub import Publisher, DataWriter

from idl_ri import Header, OdometryMsg, Pose, Vector3, Quaternion
from qos_profiles import QOS_ODOMETRY


dp = DomainParticipant(0)
pub = Publisher(dp)

t_odom = Topic(dp, "rt/motion/odometry", OdometryMsg, qos=QOS_ODOMETRY)
w_odom = DataWriter(pub, t_odom)

seq = 0


def quat_from_yaw(yaw):
    return Quaternion(
        x=0.0,
        y=0.0,
        z=math.sin(yaw / 2.0),
        w=math.cos(yaw / 2.0),
    )


print("A publicar odometria falsa em rt/motion/odometry...")

x = 0.0
y = 0.0
yaw = 0.0

while True:
    seq += 1

    msg = OdometryMsg(
        header=Header(
            timestamp_ns=time.time_ns(),
            frame_id="base_link",
            seq=seq,
        ),
        pose=Pose(
            position=Vector3(x=x, y=y, z=0.0),
            orientation=quat_from_yaw(yaw),
        ),
        vx=0.0,
        vy=0.0,
        wz=0.0,
    )

    w_odom.write(msg)

    x += 0.01
    yaw += 0.01

    time.sleep(0.02)
