import math
import time

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import (
    unitree_hg_msg_dds__HandCmd_,
    unitree_hg_msg_dds__HandState_,
)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_


class HandControl:

    def __init__(self, hand_side: str):
        """
        hand_side: "L" or "R"
        """

        if hand_side not in ["L", "R"]:
            raise ValueError("hand_side must be 'L' or 'R'")

        self.hand_side = hand_side
        self.is_left = hand_side == "L"

        # =========================
        # LIMITS
        # =========================
        self.maxLimits_left  = [1.05, 1.05, 1.75, 0, 0, 0, 0]
        self.minLimits_left  = [-1.05, -0.724, 0, -1.57, -1.75, -1.57, -1.75]
        self.maxLimits_right = [1.05, 0.742, 0, 1.57, 1.75, 1.57, 1.75]
        self.minLimits_right = [-1.05, -1.05, -1.75, 0, 0, 0, 0]

        self.MOTOR_MAX = 7

        self.msg = unitree_hg_msg_dds__HandCmd_()
        self.state = unitree_hg_msg_dds__HandState_()

        self.gripValue = 0.5
        self._count = 1
        self._dir = 1

        # =========================
        # DDS SETUP
        # =========================
        if self.is_left:
            dds_ns = "rt/dex3/left"
            sub_ns = "rt/lf/dex3/left/state"
        else:
            dds_ns = "rt/dex3/right"
            sub_ns = "rt/lf/dex3/right/state"

        #ChannelFactoryInitialize(0, interface)

        self.pub = ChannelPublisher(dds_ns + "/cmd", HandCmd_)
        self.sub = ChannelSubscriber(sub_ns, HandState_)

        self.pub.Init()
        self.sub.Init(self.state_handler, 1)

        # Initialize default positions
        for i in range(self.MOTOR_MAX):
            self.msg.motor_cmd[i].q = 0.5

        self.neutral_hand()

    # =========================
    # CALLBACK
    # =========================
    def state_handler(self, msg_in):
        self.state = msg_in

    # =========================
    # HELPERS
    # =========================
    def build_mode(self, i, status=1, timeout=0):
        return (i & 0x0F) | ((status & 0x07) << 4) | ((timeout & 0x01) << 7)

    def _limits(self):
        if self.is_left:
            return self.maxLimits_left, self.minLimits_left
        return self.maxLimits_right, self.minLimits_right

    # =========================
    # CORE ACTIONS
    # =========================
    def rotate_step(self):
        maxL, minL = self._limits()

        for i in range(self.MOTOR_MAX):
            cmd = self.msg.motor_cmd[i]

            cmd.mode = self.build_mode(i)
            cmd.tau = 0
            cmd.kp = 0.5
            cmd.kd = 0.1

            mid = (maxL[i] + minL[i]) / 2
            amp = (maxL[i] - minL[i]) / 2

            cmd.q = mid + amp * math.sin(self._count / 20000.0 * math.pi)

        self.pub.Write(self.msg)

        self._count += self._dir
        if self._count >= 10000:
            self._dir = -1
        elif self._count <= -10000:
            self._dir = 1

    def grip(self, value=None):
        time.sleep(0.5)

        if value is not None:
            self.gripValue = max(0.0, min(1.0, value))

        maxL, minL = self._limits()

        gripValueLocal = 1 - self.gripValue if self.is_left else self.gripValue

        for i in range(self.MOTOR_MAX):
            cmd = self.msg.motor_cmd[i]

            cmd.mode = self.build_mode(i)
            cmd.tau = 0
            cmd.kp = 1.5
            cmd.kd = 0.1

            if 0 < i < 3:
                q = minL[i] + (1 - gripValueLocal) * (maxL[i] - minL[i])
            elif i == 0:
                q = (maxL[i] + minL[i]) / 2
            else:
                q = minL[i] + gripValueLocal * (maxL[i] - minL[i])

            cmd.q = q
            cmd.dq = 0

        self.pub.Write(self.msg)

    def stop(self):
        time.sleep(0.5)
        for i in range(self.MOTOR_MAX):
            cmd = self.msg.motor_cmd[i]

            cmd.mode = self.build_mode(i, timeout=1)
            cmd.tau = 0
            cmd.dq = 0
            cmd.kp = 0
            cmd.kd = 0
            cmd.q = 0

        self.pub.Write(self.msg)
       
    def neutral_hand(self):
        """
        Reset the hand motors to a neutral state to avoid stuck fingers.
        Sends STOP -> neutral grip -> small delay.
        """
        print("Stopping all hand motors...")
        self.stop()      # stop everything first
        time.sleep(0.5)  # short delay
        print("Setting neutral hand grip...")
        self.grip(0.5)
        


# ----------------------------
def demo():
    interface="enp116s0"
    ChannelFactoryInitialize(0, interface)
    
    left_hand = HandControl("L")
    right_hand = HandControl("R")

    time.sleep(0.5)

    # Grip control
    left_hand.grip(0.8)
    right_hand.grip(0.8)

    '''
    # Smooth rotation loop
    for _ in range(10000):
        #left_hand.rotate_step()
        right_hand.rotate_step()
    '''

    left_hand.neutral_hand()
    right_hand.neutral_hand()

    # Stop everything
    left_hand.stop()
    right_hand.stop()  


#demo()
