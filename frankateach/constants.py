import numpy as np

# Network constants
HOST = "localhost"
CAM_PORT = 10005
VR_CONTROLLER_STATE_PORT = 8889
STATE_PORT = 8900
CONTROL_PORT = 8901
COMMANDED_STATE_PORT = 8902
RESKIN_STREAM_PORT = 12005

# Bimanual port allocation. The right arm keeps the historical ports so the
# single-arm VR teleop and data collection paths are unaffected; the left arm
# is offset by 100 (state 9000, control 9001, commanded 9002).
ARM_PORT_OFFSET = {"right": 0, "left": 100}


def arm_ports(arm="right"):
    """Return (control, state, commanded_state) ports for the given arm."""
    if arm not in ARM_PORT_OFFSET:
        raise ValueError(f"Unknown arm {arm!r}, expected one of {list(ARM_PORT_OFFSET)}")
    offset = ARM_PORT_OFFSET[arm]
    return CONTROL_PORT + offset, STATE_PORT + offset, COMMANDED_STATE_PORT + offset


STATE_TOPIC = "state"
CONTROL_TOPIC = "control"

# VR constants
VR_TCP_HOST = "10.19.225.15"
VR_TCP_PORT = 5555
VR_CONTROLLER_TOPIC = b"oculus_controller"

# Robot constants
GRIPPER_OPEN = -1
GRIPPER_CLOSE = 1
H_R_V = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
H_R_V_star = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
x_min, x_max = 0.2, 0.75
y_min, y_max = -0.4, 0.4
z_min, z_max = 0.05, 0.7  # 232, 550
ROBOT_WORKSPACE_MIN = np.array([x_min, y_min, z_min])
ROBOT_WORKSPACE_MAX = np.array([x_max, y_max, z_max])

# Panda joint velocity limits (rad/s). Exceeding one raises the
# "joint_velocity_violation" reflex, which latches and needs a Desk
# acknowledge + franka-interface restart. Joint 1 is the one air hockey can
# actually reach: it rotates about the base z axis, so it alone carries motion
# tangential to the base radius, and its Cartesian ceiling is only w*r.
JOINT_VELOCITY_LIMITS = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])
JOINT1_VELOCITY_LIMIT = float(JOINT_VELOCITY_LIMITS[0])

TRANSLATIONAL_POSE_VELOCITY_SCALE = 5
ROTATIONAL_POSE_VELOCITY_SCALE = 0.75
ROTATION_VELOCITY_LIMIT = 0.5
TRANSLATION_VELOCITY_LIMIT = 1

# Frequencies
# TODO: Separate VR and deploy frequencies
VR_FREQ = 20
CONTROL_FREQ = 20
STATE_FREQ = 100
CAM_FPS = 30
DEPTH_PORT_OFFSET = 1000

# Air hockey runs the control loop much faster than VR teleop. Kept separate so
# raising it cannot regress the existing 20 Hz VR path.
AIRHOCKEY_CONTROL_FREQ = 50
