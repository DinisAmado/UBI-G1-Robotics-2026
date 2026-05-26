# ==============================================================================
# INSTALLATION REQUIREMENTS:
# 1. Install CycloneDDS: sudo apt install ros-foxy-rmw-cyclonedds-cpp
# 2. Install Unitree SDK2 Python:
#    git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
#    cd unitree_sdk2_python && pip install -e .
#
# PIP COMMAND TO INSTALL ALL:
# pip install numpy unitree_sdk2py
# ==============================================================================

import time
import sys
import numpy as np
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_ as HandCmd

# ==========================================
# GLOBAL PARAMETERS (TUNED FOR REALITY)
# ==========================================
FORCE_THRESHOLD = 140000   # Absolute limit (Free air hits ~106k)
DELTA_THRESHOLD = 60000    # Jump limit (Free air jumps ~40k-50k)
NETWORK_INTERFACE = "enp117s0"
REFRESH_RATE = 0.02
GRIP_STEP = 0.015
EMA_ALPHA = 0.6

JOINT_NAMES = {
    0: "Thumb Rot", 1: "Thumb Flx", 2: "Thumb Tip",
    3: "Index Flx", 4: "Index Tip",
    5: "Middle Flx",6: "Middle Tip"
}
# ==========================================

maxLimits_left  = [ 1.05 ,  1.05  , 1.75 ,   0   ,  0    , 0      , 0  ]
minLimits_left  = [-1.05 , -0.724 ,   0  , -1.57 , -1.75 , -1.57  ,-1.75]
maxLimits_right = [1.05, 0.742, 0, 1.57, 1.75, 1.57, 1.75]
minLimits_right = [-1.05, -1.05, -1.75, 0, 0, 0, 0]

class Dex3SmartController:
    def __init__(self, side="L"):
        self.side = side.upper()
        self.is_left = (self.side == "L")

        self.topic_cmd = f"rt/dex3/{'left' if self.is_left else 'right'}/cmd"
        self.topic_state = f"rt/lf/dex3/{'left' if self.is_left else 'right'}/state"

        self.max_l = maxLimits_left if self.is_left else maxLimits_right
        self.min_l = minLimits_left if self.is_left else minLimits_right

        self.grip_value = 0.5

        # EMA Filter Tracking
        self.ema_tau = [0.0] * 7
        self.last_ema_tau = [0.0] * 7
        self.first_reading = True

        self.sub = ChannelSubscriber(self.topic_state, HandState_)
        self.sub.Init(self.state_callback)
        self.pub = ChannelPublisher(self.topic_cmd, HandCmd_)
        self.pub.Init()

    def state_callback(self, msg: HandState_):
        for i in range(7):
            current_tau = abs(msg.motor_state[i].tau_est)

            if self.first_reading:
                self.ema_tau[i] = current_tau
            else:
                self.last_ema_tau[i] = self.ema_tau[i]
                self.ema_tau[i] = (current_tau * EMA_ALPHA) + (self.ema_tau[i] * (1 - EMA_ALPHA))

        self.first_reading = False

    def calculate_q(self, grip):
        grip_local = (1 - grip) if self.is_left else grip
        q_targets = np.zeros(7)
        for i in range(7):
            if i > 0 and i < 3:
                q_targets[i] = self.min_l[i] + (1 - grip_local) * (self.max_l[i] - self.min_l[i])
            elif i == 0:
                q_targets[i] = (self.max_l[i] + self.min_l[i]) / 2.0
            else:
                q_targets[i] = self.min_l[i] + grip_local * (self.max_l[i] - self.min_l[i])
        return q_targets

    def send_cmd(self, grip, kp=1.5, kd=0.1):
        q_vals = self.calculate_q(grip)
        cmd = HandCmd()
        for i in range(7):
            cmd.motor_cmd[i].q = q_vals[i]
            cmd.motor_cmd[i].kp = kp
            cmd.motor_cmd[i].kd = kd
        self.pub.Write(cmd)
        self.grip_value = grip

    def reset_motors(self):
        print(f"\n[RESET] Relaxing {self.side} hand motors...")
        cmd = HandCmd()
        for i in range(7):
            cmd.motor_cmd[i].q = self.calculate_q(self.grip_value)[i]
            cmd.motor_cmd[i].kp = 0.0
            cmd.motor_cmd[i].kd = 0.0
            cmd.motor_cmd[i].tau = 0.0
        self.pub.Write(cmd)
        time.sleep(0.5)

        self.grip_value = 0.5
        self.send_cmd(self.grip_value)
        time.sleep(0.2)
        print("[STATUS] Reset Complete.")

    def open_gradually(self):
        print(f"\n[ACTION] Opening {self.side} hand...")
        while self.grip_value > 0.1:
            self.grip_value = max(0.0, round(self.grip_value - GRIP_STEP, 3))
            self.send_cmd(self.grip_value)
            time.sleep(REFRESH_RATE)
        print("[STATUS] Hand Open.")

    def diagnostic_close(self):
        """Closes while printing math for ALL motors, AND stops if force is detected."""
        print("\n[DIAGNOSTIC] Closing hand. Watch the torque values... (Will stop on force)")
        
        grace_steps = 3
        steps_taken = 0
        
        while self.grip_value < 0.95:
            # 1. Force Detection Logic
            if steps_taken > grace_steps:
                triggered_motors = []
                for i in range(7):
                    delta_tau = abs(self.ema_tau[i] - self.last_ema_tau[i])

                    if self.ema_tau[i] > FORCE_THRESHOLD:
                        triggered_motors.append(f"M{i} [ABS: {self.ema_tau[i]:.0f}]")
                    elif delta_tau > DELTA_THRESHOLD:
                        triggered_motors.append(f"M{i} [JUMP: {delta_tau:.0f}]")

                if triggered_motors:
                    print(f"\n\n!!! STOP: Force detected on: {', '.join(triggered_motors)}")
                    # BACK OFF slightly to release pressure
                    self.grip_value = max(0.0, self.grip_value - 0.02)
                    self.send_cmd(self.grip_value, kp=3.0)
                    return

            # 2. Advance Grip
            self.grip_value = min(1.0, round(self.grip_value + GRIP_STEP, 3))
            self.send_cmd(self.grip_value)

            # 3. Print Diagnostic Stream
            log_str = f"Grip {self.grip_value:.2f} | "
            for i in range(7):
                delta = abs(self.ema_tau[i] - self.last_ema_tau[i])
                log_str += f"M{i}: {self.ema_tau[i]:.0f} (J:{delta:.0f}) | "
            
            # Use sys.stdout to overwrite the same line or print normally depending on preference
            # Printing normally here to see the history of the build-up
            print(log_str)
            
            time.sleep(REFRESH_RATE)
            steps_taken += 1
            
        print("\n[STATUS] Fully closed (no object detected).")

    def close_with_force_stop(self):
        print(f"\n[ACTION] Closing {self.side} hand... monitoring force limits.")

        grace_steps = 3
        steps_taken = 0

        while self.grip_value < 0.95:
            if steps_taken > grace_steps:
                triggered_motors = []
                for i in range(7):
                    delta_tau = abs(self.ema_tau[i] - self.last_ema_tau[i])

                    if self.ema_tau[i] > FORCE_THRESHOLD:
                        triggered_motors.append(f"M{i} [ABS: {self.ema_tau[i]:.0f}]")
                    elif delta_tau > DELTA_THRESHOLD:
                        triggered_motors.append(f"M{i} [JUMP: {delta_tau:.0f}]")

                if triggered_motors:
                    print(f"\n!!! STOP: Force detected on: {', '.join(triggered_motors)}")
                    # BACK OFF slightly to release pressure
                    self.grip_value = max(0.0, self.grip_value - 0.02)
                    self.send_cmd(self.grip_value, kp=3.0)
                    return

            self.grip_value = min(1.0, round(self.grip_value + GRIP_STEP, 3))
            sys.stdout.write(f"\rClosing... Grip: {self.grip_value:.2f}")
            sys.stdout.flush()

            self.send_cmd(self.grip_value)
            time.sleep(REFRESH_RATE)
            steps_taken += 1

        print("\n[STATUS] Fully closed (no object detected).")


def main():
    try:
        ChannelFactoryInitialize(0, NETWORK_INTERFACE)
    except Exception as e:
        print(f"DDS Error: {e}")
        sys.exit(1)

    side = input("Test LEFT or RIGHT hand? (L/R): ").strip().upper()
    if side not in ["L", "R"]: side = "L"

    hand = Dex3SmartController(side=side)
    time.sleep(1.0)

    hand.send_cmd(0.5)

    print(f"\n--- {side} HAND READY ---")
    print(f"Limits -> ABS: {FORCE_THRESHOLD} | JUMP: {DELTA_THRESHOLD}")
    print("Commands: [o] Open | [c] Close | [d] Diagnostic | [r] Reset | [q] Quit")

    while True:
        cmd = input(f"\n[{side}] >> ").lower().strip()
        if cmd == 'o': hand.open_gradually()
        elif cmd == 'c': hand.close_with_force_stop()
        elif cmd == 'd': hand.diagnostic_close()
        elif cmd == 'r': hand.reset_motors()
        elif cmd == 'q': break

if __name__ == "__main__":
    main()

