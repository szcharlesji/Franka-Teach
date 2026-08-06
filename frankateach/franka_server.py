import os
from pathlib import Path
import pickle
import time
from contextlib import contextmanager
import numpy as np

from deoxys.utils import YamlConfig
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import transform_utils
from deoxys.utils.config_utils import (
    get_default_controller_config,
    verify_controller_config,
)

from frankateach.utils import notify_component_start
from frankateach.network import create_response_socket
from frankateach.messages import FrankaAction, FrankaState
from frankateach.constants import (
    HOST,
    CONTROL_FREQ,
    arm_ports,
)

CONFIG_ROOT = Path(__file__).parent / "configs"

# Deoxys' default ready pose. Overridable per arm so each robot can be parked
# facing its own half of the air hockey table.
DEFAULT_START_JOINT_POS = [
    0.09162008114028396,
    -0.19826458111314524,
    -0.01990020486871322,
    -2.4732269941140346,
    -0.01307073642274261,
    2.30396583422025,
    0.8480939705504309,
]


class FrankaServer:
    def __init__(
        self,
        cfg,
        arm="right",
        control_freq=CONTROL_FREQ,
        num_steps=3,
        start_joint_pos=None,
        control_gripper=True,
    ):
        self._robot = Robot(cfg, control_freq, control_gripper=control_gripper)
        self.num_steps = num_steps
        self.start_joint_pos = (
            list(start_joint_pos)
            if start_joint_pos is not None
            else DEFAULT_START_JOINT_POS
        )
        control_port, _, _ = arm_ports(arm)
        print(f"Franka server for {arm!r} arm listening on {HOST}:{control_port}")
        # Action REQ/REP
        self.action_socket = create_response_socket(HOST, control_port)

    def init_server(self):
        # connect to robot
        print("Starting Franka server...")
        self._robot.reset_robot()
        self.control_daemon()

    def get_state(self):
        quat, pos = self._robot.last_eef_quat_and_pos
        gripper = self._robot.last_gripper_action
        if quat is not None and pos is not None and gripper is not None:
            state = FrankaState(
                pos=pos.flatten().astype(np.float32),
                quat=quat.flatten().astype(np.float32),
                gripper=gripper,
                timestamp=time.time(),
            )
            return bytes(pickle.dumps(state, protocol=-1))
        else:
            return b"state_error"

    def control_daemon(self):
        notify_component_start(component_name="Franka Control Subscriber")
        try:
            while True:
                command = self.action_socket.recv()
                if command == b"get_state":
                    self.action_socket.send(self.get_state())
                else:
                    franka_control: FrankaAction = pickle.loads(command)
                    if franka_control.reset:
                        self._robot.reset_joints(
                            gripper_open=franka_control.gripper,
                            start_joint_pos=self.start_joint_pos,
                        )
                        time.sleep(1)
                    else:
                        self._robot.osc_move(
                            franka_control.pos,
                            franka_control.quat,
                            franka_control.gripper,
                            num_steps=self.num_steps,
                        )
                    self.action_socket.send(self.get_state())
        except KeyboardInterrupt:
            pass
        finally:
            self._robot.close()
            self.action_socket.close()


class Robot(FrankaInterface):
    def __init__(self, cfg, control_freq, control_gripper=True):
        super(Robot, self).__init__(
            general_cfg_file=os.path.join(CONFIG_ROOT, cfg),
            use_visualizer=False,
            control_freq=control_freq,
            has_gripper=control_gripper,
            automatic_gripper_reset=control_gripper,
        )
        self.velocity_controller_cfg = verify_controller_config(
            YamlConfig(
                os.path.join(CONFIG_ROOT, "osc-pose-controller.yml")
            ).as_easydict()
        )

    @contextmanager
    def _gripper_mode(self, enabled):
        """Temporarily suppress every gripper side effect for an arm-only command."""
        old_has_gripper = self.has_gripper
        old_automatic_reset = self.automatic_gripper_reset
        if not enabled:
            # control() otherwise treats zero as CLOSE, while preprocess() opens
            # the hand whenever the arm switches controller types.
            self.has_gripper = False
            self.automatic_gripper_reset = False
        try:
            yield
        finally:
            self.has_gripper = old_has_gripper
            self.automatic_gripper_reset = old_automatic_reset

    def reset_robot(self):
        self.reset()

        print("Waiting for the robot to connect...")
        while len(self._state_buffer) == 0:
            time.sleep(0.01)

        print("Franka is connected")

    def osc_move(self, target_pos, target_quat, gripper_state, num_steps=3):
        with self._gripper_mode(gripper_state is not None):
            for _ in range(num_steps):
                target_mat = transform_utils.pose2mat(pose=(target_pos, target_quat))

                current_quat, current_pos = self.last_eef_quat_and_pos
                current_mat = transform_utils.pose2mat(
                    pose=(current_pos.flatten(), current_quat.flatten())
                )

                pose_error = transform_utils.get_pose_error(
                    target_pose=target_mat, current_pose=current_mat
                )

                if np.dot(target_quat, current_quat) < 0.0:
                    current_quat = -current_quat

                quat_diff = transform_utils.quat_distance(target_quat, current_quat)
                axis_angle_diff = transform_utils.quat2axisangle(quat_diff)

                action_pos = pose_error[:3]
                action_axis_angle = axis_angle_diff.flatten()

                # Deoxys still requires the eighth element when has_gripper=False;
                # it is ignored rather than sent to gripper-interface.
                gripper_action = 0 if gripper_state is None else gripper_state
                action = (
                    action_pos.tolist()
                    + action_axis_angle.tolist()
                    + [gripper_action]
                )

                self.control(
                    controller_type="OSC_POSE",
                    action=action,
                    controller_cfg=self.velocity_controller_cfg,
                )

    def reset_joints(
        self,
        timeout=7,
        gripper_open=False,
        start_joint_pos=None,
    ):
        if start_joint_pos is None:
            start_joint_pos = DEFAULT_START_JOINT_POS
        assert type(start_joint_pos) is list or type(start_joint_pos) is np.ndarray
        controller_cfg = get_default_controller_config(controller_type="JOINT_POSITION")

        if gripper_open is None:
            gripper_action = 0
        elif gripper_open:
            gripper_action = -1
        else:
            gripper_action = 1

        # This is for varying initialization of joints a little bit to
        # increase data variation.
        # start_joint_pos = [
        #     e + np.clip(np.random.randn() * 0.005, -0.005, 0.005)
        #     for e in start_joint_pos
        # ]
        if type(start_joint_pos) is list:
            action = start_joint_pos + [gripper_action]
        else:
            action = start_joint_pos.tolist() + [gripper_action]
        start_time = time.time()
        with self._gripper_mode(gripper_open is not None):
            while True:
                if self.received_states and self.check_nonzero_configuration():
                    if (
                        np.max(
                            np.abs(np.array(self.last_q) - np.array(start_joint_pos))
                        )
                        < 1e-3
                    ):
                        break
                self.control(
                    controller_type="JOINT_POSITION",
                    action=action,
                    controller_cfg=controller_cfg,
                )
                end_time = time.time()

                # Add timeout
                if end_time - start_time > timeout:
                    break
        return True
