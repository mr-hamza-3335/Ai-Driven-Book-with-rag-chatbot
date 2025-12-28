# Chapter 6: Capstone Project - Building an Autonomous Humanoid System

## Learning Objectives

By the end of this chapter, you will be able to:
- Integrate all concepts from previous chapters into a complete system
- Design and implement a full humanoid robot pipeline
- Deploy perception, planning, and control on real hardware
- Evaluate system performance and iterate on design
- Document and present your humanoid robotics project

---

## 6.1 Project Overview

### Capstone Challenge

Build an autonomous humanoid robot system that can:

1. **Navigate** through an indoor environment
2. **Recognize** objects and people
3. **Manipulate** objects based on voice commands
4. **Communicate** status and ask for clarification

```
┌─────────────────────────────────────────────────────────────────┐
│                    Autonomous Humanoid System                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│   ┌───────────────┐    ┌───────────────┐    ┌───────────────┐   │
│   │   Perception  │───▶│   Planning    │───▶│   Execution   │   │
│   │               │    │               │    │               │   │
│   │ • Vision      │    │ • Navigation  │    │ • Locomotion  │   │
│   │ • Speech      │    │ • Task        │    │ • Manipulation│   │
│   │ • SLAM        │    │ • Motion      │    │ • Balance     │   │
│   └───────────────┘    └───────────────┘    └───────────────┘   │
│           │                    │                    │             │
│           └────────────────────┼────────────────────┘             │
│                                ▼                                  │
│                    ┌───────────────────┐                         │
│                    │    Coordination   │                         │
│                    │   State Machine   │                         │
│                    └───────────────────┘                         │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### System Requirements

| Component | Requirement | Technology |
|-----------|-------------|------------|
| Perception | Object detection >90% accuracy | YOLOv8 + SAM |
| Navigation | Autonomous in 10x10m space | Nav2 + SLAM |
| Manipulation | Pick objects within reach | MoveIt2 + VLA |
| Speech | Understand 50+ commands | Whisper + LLM |
| Response time | <500ms for commands | ROS 2 real-time |

---

## 6.2 System Architecture

### ROS 2 Node Graph

```python
# launch/humanoid_system.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription

def generate_launch_description():
    return LaunchDescription([
        # Perception Stack
        Node(
            package='humanoid_perception',
            executable='camera_node',
            name='camera',
            parameters=[{'device': '/dev/video0'}]
        ),
        Node(
            package='humanoid_perception',
            executable='object_detector',
            name='detector',
        ),
        Node(
            package='humanoid_perception',
            executable='slam_node',
            name='slam',
        ),

        # Planning Stack
        Node(
            package='humanoid_planning',
            executable='task_planner',
            name='task_planner',
        ),
        Node(
            package='nav2_bringup',
            executable='navigation_launch.py',
            name='navigation',
        ),

        # Control Stack
        Node(
            package='humanoid_control',
            executable='locomotion_controller',
            name='locomotion',
        ),
        Node(
            package='humanoid_control',
            executable='manipulation_controller',
            name='manipulation',
        ),

        # Coordination
        Node(
            package='humanoid_coordination',
            executable='state_machine',
            name='coordinator',
        ),
    ])
```

### Message Types

```python
# msg/HumanoidState.msg
std_msgs/Header header
string current_task
string current_state
geometry_msgs/Pose robot_pose
float32[] joint_positions
bool is_balanced
bool gripper_closed

# msg/TaskCommand.msg
std_msgs/Header header
string command_type  # "navigate", "pick", "place", "speak"
string target_object
geometry_msgs/Pose target_pose
string speech_text

# msg/PerceptionResult.msg
std_msgs/Header header
vision_msgs/Detection3DArray detected_objects
sensor_msgs/PointCloud2 scene_pointcloud
geometry_msgs/PoseArray object_poses
```

---

## 6.3 Perception Module

### Multi-Modal Perception

```python
import rclpy
from rclpy.node import Node
import torch
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

class PerceptionNode(Node):
    """Multi-modal perception for humanoid robot."""

    def __init__(self):
        super().__init__('perception')

        # Load models
        self.detector = YOLO('yolov8x.pt')
        self.sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
        self.sam_predictor = SamPredictor(self.sam)

        # Subscribers
        self.rgb_sub = self.create_subscription(
            Image, '/camera/rgb', self.rgb_callback, 10
        )
        self.depth_sub = self.create_subscription(
            Image, '/camera/depth', self.depth_callback, 10
        )

        # Publishers
        self.detection_pub = self.create_publisher(
            Detection3DArray, '/detections', 10
        )
        self.segmentation_pub = self.create_publisher(
            Image, '/segmentation', 10
        )

        self.current_depth = None
        self.bridge = CvBridge()

    def rgb_callback(self, msg):
        """Process RGB image for object detection."""
        image = self.bridge.imgmsg_to_cv2(msg, "rgb8")

        # Detect objects
        results = self.detector(image)

        # Segment each detection
        detections = []
        for box in results[0].boxes:
            # Get bounding box
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            # SAM segmentation
            self.sam_predictor.set_image(image)
            masks, _, _ = self.sam_predictor.predict(
                box=np.array([x1, y1, x2, y2])
            )

            # Estimate 3D pose from depth
            if self.current_depth is not None:
                pose = self._estimate_pose(masks[0], self.current_depth)

                detection = Detection3D()
                detection.bbox.center.position.x = pose[0]
                detection.bbox.center.position.y = pose[1]
                detection.bbox.center.position.z = pose[2]
                detection.results[0].hypothesis.class_id = str(int(box.cls))
                detection.results[0].hypothesis.score = float(box.conf)

                detections.append(detection)

        # Publish
        msg = Detection3DArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.detections = detections
        self.detection_pub.publish(msg)

    def _estimate_pose(self, mask, depth):
        """Estimate 3D position from mask and depth."""
        # Get masked depth values
        masked_depth = depth * mask
        valid_depths = masked_depth[masked_depth > 0]

        if len(valid_depths) == 0:
            return [0, 0, 0]

        # Centroid in image space
        y_indices, x_indices = np.where(mask)
        cx = np.mean(x_indices)
        cy = np.mean(y_indices)
        z = np.median(valid_depths)

        # Project to 3D (assuming pinhole camera)
        fx, fy = 525.0, 525.0  # Focal lengths
        x = (cx - 320) * z / fx
        y = (cy - 240) * z / fy

        return [x, y, z]
```

### Speech Recognition

```python
import whisper
from transformers import pipeline

class SpeechNode(Node):
    """Speech recognition and understanding."""

    def __init__(self):
        super().__init__('speech')

        # Load Whisper
        self.whisper_model = whisper.load_model("base")

        # Load LLM for command parsing
        self.command_parser = pipeline(
            "text-generation",
            model="microsoft/phi-2"
        )

        # Audio subscriber
        self.audio_sub = self.create_subscription(
            AudioData, '/microphone/audio', self.audio_callback, 10
        )

        # Command publisher
        self.command_pub = self.create_publisher(
            TaskCommand, '/task_command', 10
        )

        self.audio_buffer = []

    def audio_callback(self, msg):
        """Process incoming audio."""
        self.audio_buffer.extend(msg.data)

        # Process every 3 seconds
        if len(self.audio_buffer) >= 48000 * 3:  # 16kHz * 3s
            audio = np.array(self.audio_buffer[:48000*3], dtype=np.float32) / 32768.0
            self.audio_buffer = self.audio_buffer[48000*3:]

            # Transcribe
            result = self.whisper_model.transcribe(audio)
            text = result["text"].strip()

            if text:
                self.get_logger().info(f"Heard: {text}")
                self._parse_command(text)

    def _parse_command(self, text: str):
        """Parse natural language to robot command."""
        prompt = f"""Parse this command into a robot action:
Command: "{text}"

Output JSON with fields:
- command_type: "navigate", "pick", "place", or "speak"
- target_object: object name or null
- target_location: location name or null

JSON:"""

        response = self.command_parser(prompt, max_length=100)[0]["generated_text"]

        try:
            # Extract JSON from response
            json_str = response.split("JSON:")[-1].strip()
            command_data = json.loads(json_str)

            # Publish command
            msg = TaskCommand()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.command_type = command_data.get("command_type", "")
            msg.target_object = command_data.get("target_object", "")
            self.command_pub.publish(msg)

        except json.JSONDecodeError:
            self.get_logger().warn("Could not parse command")
```

---

## 6.4 Planning Module

### Task Planner

```python
from enum import Enum
from dataclasses import dataclass

class TaskState(Enum):
    IDLE = "idle"
    NAVIGATING = "navigating"
    APPROACHING = "approaching"
    PICKING = "picking"
    PLACING = "placing"
    SPEAKING = "speaking"
    ERROR = "error"

@dataclass
class Task:
    task_type: str
    target: str
    parameters: dict
    priority: int = 0

class TaskPlanner(Node):
    """High-level task planning for humanoid."""

    def __init__(self):
        super().__init__('task_planner')

        self.current_state = TaskState.IDLE
        self.task_queue = []
        self.current_task = None

        # World model
        self.known_objects = {}
        self.known_locations = {
            "kitchen": [2.0, 3.0, 0.0],
            "living_room": [5.0, 1.0, 0.0],
            "table": [3.0, 2.0, 0.8]
        }

        # Subscribers
        self.command_sub = self.create_subscription(
            TaskCommand, '/task_command', self.command_callback, 10
        )
        self.detection_sub = self.create_subscription(
            Detection3DArray, '/detections', self.detection_callback, 10
        )

        # Publishers
        self.nav_goal_pub = self.create_publisher(
            PoseStamped, '/goal_pose', 10
        )
        self.manipulation_goal_pub = self.create_publisher(
            ManipulationGoal, '/manipulation_goal', 10
        )

        # Action clients
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # Planning loop
        self.timer = self.create_timer(0.1, self.planning_loop)

    def command_callback(self, msg):
        """Receive and queue new task."""
        task = Task(
            task_type=msg.command_type,
            target=msg.target_object,
            parameters={}
        )
        self.task_queue.append(task)
        self.get_logger().info(f"Queued task: {task.task_type} -> {task.target}")

    def planning_loop(self):
        """Main planning state machine."""
        if self.current_state == TaskState.IDLE:
            if self.task_queue:
                self.current_task = self.task_queue.pop(0)
                self._start_task(self.current_task)

        elif self.current_state == TaskState.NAVIGATING:
            self._check_navigation_status()

        elif self.current_state == TaskState.PICKING:
            self._check_manipulation_status()

    def _start_task(self, task: Task):
        """Start executing a task."""
        if task.task_type == "navigate":
            self._start_navigation(task.target)
        elif task.task_type == "pick":
            self._start_pick(task.target)
        elif task.task_type == "place":
            self._start_place(task.target)

    def _start_navigation(self, target: str):
        """Start navigation to target location."""
        if target in self.known_locations:
            pose = self.known_locations[target]

            goal = NavigateToPose.Goal()
            goal.pose.header.frame_id = "map"
            goal.pose.pose.position.x = pose[0]
            goal.pose.pose.position.y = pose[1]
            goal.pose.pose.orientation.w = 1.0

            self.nav_client.send_goal_async(goal)
            self.current_state = TaskState.NAVIGATING
            self.get_logger().info(f"Navigating to {target}")

    def _start_pick(self, target: str):
        """Start picking an object."""
        if target in self.known_objects:
            obj = self.known_objects[target]

            goal = ManipulationGoal()
            goal.action = "pick"
            goal.target_pose = obj.pose
            goal.object_id = target

            self.manipulation_goal_pub.publish(goal)
            self.current_state = TaskState.PICKING
            self.get_logger().info(f"Picking {target}")
```

### Motion Planning with MoveIt2

```python
from moveit_msgs.msg import MoveGroupAction, Constraints
from moveit_msgs.srv import GetPositionIK

class ManipulationController(Node):
    """Manipulation control using MoveIt2."""

    def __init__(self):
        super().__init__('manipulation')

        # MoveIt interface
        self.move_group = MoveGroupInterface(
            group_name="arm",
            ns="",
            robot_description="robot_description"
        )

        # IK service
        self.ik_client = self.create_client(
            GetPositionIK, '/compute_ik'
        )

        # VLA model for learned manipulation
        self.vla_model = OpenVLA.load_pretrained("openvla-humanoid")
        self.vla_model.eval()

        # Goal subscriber
        self.goal_sub = self.create_subscription(
            ManipulationGoal, '/manipulation_goal',
            self.goal_callback, 10
        )

        # Joint command publisher
        self.joint_pub = self.create_publisher(
            JointTrajectory, '/arm_controller/command', 10
        )

    def goal_callback(self, msg):
        """Handle manipulation goal."""
        if msg.action == "pick":
            self._execute_pick(msg.target_pose, msg.object_id)
        elif msg.action == "place":
            self._execute_place(msg.target_pose)

    def _execute_pick(self, target_pose, object_id):
        """Execute pick action using VLA."""
        # Get current images
        images = self._get_camera_images()

        # Generate instruction
        instruction = f"Pick up the {object_id}"

        # Get VLA actions
        actions = []
        for step in range(20):  # Max 20 steps
            action = self.vla_model.predict(images, instruction)
            actions.append(action)

            # Execute action
            self._send_joint_command(action[:6])

            # Check grasp
            if self._check_grasp():
                break

            # Update images
            time.sleep(0.1)
            images = self._get_camera_images()

        self.get_logger().info(f"Pick completed in {len(actions)} steps")

    def _send_joint_command(self, positions):
        """Send joint position command."""
        msg = JointTrajectory()
        msg.joint_names = [
            "shoulder_pan", "shoulder_lift", "elbow",
            "wrist_1", "wrist_2", "wrist_3"
        ]

        point = JointTrajectoryPoint()
        point.positions = positions.tolist()
        point.time_from_start.sec = 0
        point.time_from_start.nanosec = 100000000  # 100ms

        msg.points = [point]
        self.joint_pub.publish(msg)
```

---

## 6.5 Control Module

### Whole-Body Control

```python
import numpy as np
from scipy.linalg import solve

class WholeBodyController:
    """Whole-body control for humanoid locomotion and manipulation."""

    def __init__(self, robot_model):
        self.model = robot_model
        self.num_joints = robot_model.num_joints

        # Control gains
        self.kp_balance = 100.0
        self.kd_balance = 20.0
        self.kp_task = 50.0
        self.kd_task = 10.0

    def compute_control(
        self,
        q: np.ndarray,           # Current joint positions
        dq: np.ndarray,          # Current joint velocities
        task_target: np.ndarray, # Desired end-effector pose
        com_target: np.ndarray,  # Desired center of mass
    ) -> np.ndarray:
        """Compute joint torques for whole-body control."""

        # Forward kinematics
        ee_pose = self.model.forward_kinematics(q)
        com = self.model.center_of_mass(q)

        # Jacobians
        J_ee = self.model.end_effector_jacobian(q)
        J_com = self.model.com_jacobian(q)

        # Task-space errors
        ee_error = task_target - ee_pose
        com_error = com_target - com

        # Task-space velocities
        ee_vel = J_ee @ dq
        com_vel = J_com @ dq

        # Desired task accelerations (PD control)
        ddx_ee = self.kp_task * ee_error - self.kd_task * ee_vel
        ddx_com = self.kp_balance * com_error - self.kd_balance * com_vel

        # Stack tasks (COM has higher priority)
        J_stack = np.vstack([J_com, J_ee])
        ddx_stack = np.hstack([ddx_com, ddx_ee])

        # Solve for joint accelerations (least squares)
        ddq = np.linalg.lstsq(J_stack, ddx_stack, rcond=None)[0]

        # Inverse dynamics to get torques
        tau = self.model.inverse_dynamics(q, dq, ddq)

        return tau

    def balance_control(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        imu_orientation: np.ndarray,
        foot_contacts: list[bool]
    ) -> np.ndarray:
        """Compute torques for maintaining balance."""

        # Desired upright orientation
        target_orientation = np.array([1, 0, 0, 0])  # Identity quaternion

        # Orientation error (quaternion difference)
        orientation_error = self._quaternion_error(
            target_orientation, imu_orientation
        )

        # Angular velocity (from IMU)
        angular_vel = self.model.angular_velocity(q, dq)

        # Compute corrective torques
        tau_balance = (
            self.kp_balance * orientation_error
            - self.kd_balance * angular_vel
        )

        # Map to joint torques based on contact state
        if all(foot_contacts):  # Double support
            tau = self._double_support_mapping(tau_balance, q)
        elif foot_contacts[0]:  # Left foot only
            tau = self._single_support_mapping(tau_balance, q, "left")
        elif foot_contacts[1]:  # Right foot only
            tau = self._single_support_mapping(tau_balance, q, "right")
        else:  # Flight phase
            tau = np.zeros(self.num_joints)

        return tau
```

### Locomotion Controller

```python
class LocomotionController(Node):
    """Bipedal locomotion controller."""

    def __init__(self):
        super().__init__('locomotion')

        # Load trained policy
        self.policy = torch.jit.load("locomotion_policy.pt")
        self.policy.eval()

        # State estimation
        self.state_estimator = StateEstimator()

        # Subscribers
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10
        )
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )
        self.cmd_vel_sub = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_vel_callback, 10
        )

        # Publisher
        self.joint_cmd_pub = self.create_publisher(
            JointTrajectory, '/joint_trajectory_controller/command', 10
        )

        # Control loop at 100 Hz
        self.timer = self.create_timer(0.01, self.control_loop)

        self.target_velocity = np.zeros(3)
        self.current_state = None

    def control_loop(self):
        """Main control loop."""
        if self.current_state is None:
            return

        # Build observation
        obs = self._build_observation()

        # Get action from policy
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
            action = self.policy(obs_tensor).numpy().squeeze()

        # Convert to joint commands
        joint_targets = self._action_to_joints(action)

        # Publish command
        self._publish_joint_command(joint_targets)

    def _build_observation(self) -> np.ndarray:
        """Build observation vector for policy."""
        return np.concatenate([
            self.current_state.joint_positions,
            self.current_state.joint_velocities,
            self.current_state.base_orientation,
            self.current_state.base_angular_velocity,
            self.target_velocity,
            np.sin(self.phase),  # Gait phase
            np.cos(self.phase),
        ])

    def _action_to_joints(self, action: np.ndarray) -> np.ndarray:
        """Convert policy action to joint positions."""
        # Action is delta from default pose
        default_pose = np.array([
            0, 0.5, -1.0, 0, 0.5, -1.0,  # Legs (6 DOF each)
            0, 0, 0, 0, 0, 0              # Arms
        ])
        return default_pose + action * 0.1  # Scale factor
```

---

## 6.6 System Integration

### State Machine Coordinator

```python
from transitions import Machine

class HumanoidCoordinator(Node):
    """Central coordinator using state machine."""

    STATES = [
        'idle', 'listening', 'planning',
        'navigating', 'manipulating', 'speaking', 'error'
    ]

    def __init__(self):
        super().__init__('coordinator')

        # Initialize state machine
        self.machine = Machine(
            model=self,
            states=self.STATES,
            initial='idle'
        )

        # Define transitions
        self.machine.add_transition('hear_command', 'idle', 'listening')
        self.machine.add_transition('understand', 'listening', 'planning')
        self.machine.add_transition('plan_ready', 'planning', 'navigating')
        self.machine.add_transition('arrived', 'navigating', 'manipulating')
        self.machine.add_transition('done', 'manipulating', 'speaking')
        self.machine.add_transition('finished', 'speaking', 'idle')
        self.machine.add_transition('fail', '*', 'error')
        self.machine.add_transition('recover', 'error', 'idle')

        # ROS interfaces
        self.setup_ros_interfaces()

        # Heartbeat
        self.timer = self.create_timer(0.1, self.heartbeat)

    def setup_ros_interfaces(self):
        """Set up all ROS subscribers and publishers."""
        # Status publisher
        self.status_pub = self.create_publisher(
            HumanoidState, '/humanoid_status', 10
        )

        # Monitor all subsystems
        self.nav_status_sub = self.create_subscription(
            GoalStatus, '/navigate_to_pose/_action/status',
            self.nav_status_callback, 10
        )
        self.manipulation_status_sub = self.create_subscription(
            String, '/manipulation_status',
            self.manipulation_status_callback, 10
        )

    def heartbeat(self):
        """Publish system status."""
        msg = HumanoidState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.current_state = self.state
        msg.current_task = str(self.current_task) if hasattr(self, 'current_task') else ""
        self.status_pub.publish(msg)

    def on_enter_error(self):
        """Handle error state."""
        self.get_logger().error("System entered error state")
        # Stop all motion
        self._emergency_stop()
        # Attempt recovery after delay
        self.create_timer(5.0, lambda: self.recover(), oneshot=True)
```

### System Launch

```python
# launch/full_system.launch.py
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, GroupAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import PushRosNamespace

def generate_launch_description():
    return LaunchDescription([
        # Hardware drivers
        GroupAction([
            PushRosNamespace('hardware'),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource('drivers.launch.py')
            ),
        ]),

        # Perception
        GroupAction([
            PushRosNamespace('perception'),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource('perception.launch.py')
            ),
        ]),

        # Navigation
        GroupAction([
            PushRosNamespace('navigation'),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource('nav2_bringup.launch.py'),
                launch_arguments={
                    'map': '/maps/lab.yaml',
                    'params_file': '/config/nav2_params.yaml'
                }.items()
            ),
        ]),

        # Control
        GroupAction([
            PushRosNamespace('control'),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource('control.launch.py')
            ),
        ]),

        # Coordinator
        Node(
            package='humanoid_coordination',
            executable='coordinator',
            name='coordinator',
            output='screen'
        ),
    ])
```

---

## 6.7 Testing and Evaluation

### Unit Tests

```python
import pytest
import rclpy
from humanoid_perception.perception_node import PerceptionNode

class TestPerception:
    """Test perception module."""

    @pytest.fixture
    def node(self):
        rclpy.init()
        node = PerceptionNode()
        yield node
        node.destroy_node()
        rclpy.shutdown()

    def test_object_detection(self, node):
        """Test object detection accuracy."""
        test_image = load_test_image("test_scene.png")
        detections = node.detect_objects(test_image)

        assert len(detections) > 0
        assert all(d.confidence > 0.5 for d in detections)

    def test_pose_estimation(self, node):
        """Test 3D pose estimation."""
        test_image = load_test_image("test_object.png")
        test_depth = load_test_depth("test_depth.png")

        pose = node.estimate_pose(test_image, test_depth)

        # Should be within 5cm of ground truth
        assert np.linalg.norm(pose - GROUND_TRUTH_POSE) < 0.05

class TestLocomotion:
    """Test locomotion controller."""

    def test_balance_recovery(self, simulation):
        """Test recovery from push disturbance."""
        # Apply push
        simulation.apply_force([0, 50, 0], duration=0.1)

        # Wait for recovery
        simulation.step(seconds=3.0)

        # Check upright
        orientation = simulation.get_robot_orientation()
        tilt = quaternion_to_euler(orientation)[1]  # Pitch
        assert abs(tilt) < 0.1  # Less than 6 degrees

    def test_walking_speed(self, simulation):
        """Test walking velocity tracking."""
        target_vel = 0.5  # m/s
        simulation.set_cmd_vel(target_vel, 0, 0)

        simulation.step(seconds=10.0)

        actual_vel = simulation.get_robot_velocity()[0]
        assert abs(actual_vel - target_vel) < 0.1
```

### Integration Tests

```python
class TestSystemIntegration:
    """End-to-end system tests."""

    def test_voice_to_action(self, system):
        """Test complete voice command pipeline."""
        # Send voice command
        system.send_audio("Pick up the red cup from the table")

        # Wait for execution
        system.wait_for_state("idle", timeout=60)

        # Verify cup was picked
        assert system.gripper_holding_object()
        assert system.last_object_picked == "red_cup"

    def test_navigation_and_manipulation(self, system):
        """Test combined navigation and manipulation."""
        # Start at origin
        system.reset_robot_pose([0, 0, 0])

        # Command to fetch object
        system.execute_command("Go to the kitchen and get a water bottle")

        # Should navigate to kitchen
        system.wait_for_state("manipulating", timeout=30)
        assert np.linalg.norm(system.robot_pose[:2] - KITCHEN_POS[:2]) < 0.5

        # Should pick up bottle
        system.wait_for_state("idle", timeout=30)
        assert system.gripper_holding_object()
```

### Performance Metrics

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| Task success rate | >90% | 100 trial average |
| Navigation accuracy | <10cm | RMS error to goal |
| Pick success | >85% | Grasp completion |
| Response latency | <500ms | Command to motion |
| Battery life | >2 hours | Full operation time |

---

## 6.8 Documentation and Presentation

### Project Documentation

```markdown
# Humanoid Robot System Documentation

## System Overview
[Architecture diagram and description]

## Installation
```bash
# Clone repository
git clone https://github.com/your-org/humanoid-system.git

# Install dependencies
cd humanoid-system
rosdep install --from-paths src --ignore-src -r -y

# Build
colcon build --symlink-install
```

## Configuration
[Parameter files and tuning guide]

## Operation Manual
1. Power on sequence
2. Calibration procedure
3. Starting the system
4. Voice commands reference
5. Emergency stop procedure

## Troubleshooting
[Common issues and solutions]
```

### Demo Presentation

1. **Introduction** (2 min)
   - Problem statement
   - System overview

2. **Technical Deep-Dive** (5 min)
   - Architecture walkthrough
   - Key innovations

3. **Live Demo** (10 min)
   - Voice command demonstration
   - Navigation and manipulation
   - Error recovery

4. **Results** (3 min)
   - Performance metrics
   - Comparison to baselines

5. **Future Work** (2 min)
   - Planned improvements
   - Research directions

---

## Chapter Summary

- Capstone integrates perception, planning, and control
- State machine coordinates all subsystems
- VLA models enable natural language control
- Thorough testing ensures reliability
- Documentation enables reproducibility

---

## Final Project Checklist

- [ ] Perception module detects >10 object classes
- [ ] Navigation works in 10x10m environment
- [ ] Manipulation succeeds >80% for tabletop objects
- [ ] Voice commands recognized with >95% accuracy
- [ ] System responds within 500ms
- [ ] Recovers from push disturbances
- [ ] Runs for >1 hour on battery
- [ ] Documentation complete
- [ ] Video demo recorded
- [ ] Code repository organized

---

## Congratulations!

You have completed the Physical AI & Humanoid Robotics curriculum. You now have the skills to:

- Design and build humanoid robot software systems
- Integrate perception, planning, and control
- Train and deploy neural network policies
- Use modern VLA models for intelligent control
- Test, document, and present robotics projects

**Continue your journey:**
- Contribute to open-source humanoid projects
- Explore latest research in embodied AI
- Build your own humanoid robot!
