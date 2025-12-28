# Chapter 2: ROS 2 Fundamentals for Humanoid Robotics

## Learning Objectives

By the end of this chapter, you will be able to:
- Install and configure ROS 2 Humble for humanoid development
- Understand the ROS 2 architecture and communication patterns
- Create nodes, topics, services, and actions
- Build a basic humanoid control package
- Use visualization and debugging tools

---

## 2.1 Introduction to ROS 2

### What is ROS 2?

ROS 2 (Robot Operating System 2) is a flexible framework for writing robot software. Despite its name, it's not an operating system but rather:

- A **middleware** for robot communication
- A **set of tools** for development and debugging
- A **collection of libraries** for common robotics tasks
- An **ecosystem** of packages and community support

### Why ROS 2 for Humanoids?

| Feature | Benefit for Humanoids |
|---------|----------------------|
| Real-time support | Critical for balance control |
| DDS middleware | Reliable inter-process communication |
| Lifecycle nodes | Managed startup/shutdown |
| Quality of Service | Configurable reliability |
| Multi-platform | Linux, Windows, macOS |

---

## 2.2 Installation and Setup

### Installing ROS 2 Humble

```bash
# Ubuntu 22.04 installation
sudo apt update && sudo apt install locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8

# Add ROS 2 repository
sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
  http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
  | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS 2
sudo apt update
sudo apt install ros-humble-desktop
```

### Environment Setup

```bash
# Add to ~/.bashrc
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc

# Verify installation
ros2 --version
```

---

## 2.3 ROS 2 Core Concepts

### Nodes

Nodes are the fundamental processing units in ROS 2:

```python
# humanoid_node.py
import rclpy
from rclpy.node import Node

class HumanoidController(Node):
    def __init__(self):
        super().__init__('humanoid_controller')
        self.get_logger().info('Humanoid controller initialized')

        # Create timer for control loop
        self.timer = self.create_timer(0.01, self.control_loop)  # 100 Hz

    def control_loop(self):
        # Main control logic here
        pass

def main(args=None):
    rclpy.init(args=args)
    node = HumanoidController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Topics

Topics enable publish/subscribe communication:

```python
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist

class HumanoidController(Node):
    def __init__(self):
        super().__init__('humanoid_controller')

        # Publisher for joint commands
        self.joint_pub = self.create_publisher(
            JointState,
            '/joint_commands',
            10
        )

        # Subscriber for velocity commands
        self.vel_sub = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.velocity_callback,
            10
        )

    def velocity_callback(self, msg):
        self.get_logger().info(f'Received velocity: {msg.linear.x}')
```

### Topic Communication Pattern

```
┌─────────────────┐      /joint_states       ┌─────────────────┐
│   Joint State   │ ─────────────────────────▶│   Visualizer    │
│   Publisher     │                           │   (RViz2)       │
└─────────────────┘                           └─────────────────┘

┌─────────────────┐      /cmd_vel            ┌─────────────────┐
│   Teleop        │ ─────────────────────────▶│   Motion        │
│   Controller    │                           │   Planner       │
└─────────────────┘                           └─────────────────┘
```

### Services

Services provide request/response communication:

```python
from std_srvs.srv import SetBool

class HumanoidController(Node):
    def __init__(self):
        super().__init__('humanoid_controller')

        # Create service
        self.enable_srv = self.create_service(
            SetBool,
            '/enable_motors',
            self.enable_motors_callback
        )

    def enable_motors_callback(self, request, response):
        if request.data:
            self.get_logger().info('Enabling motors')
            # Enable motor logic
            response.success = True
            response.message = 'Motors enabled'
        else:
            self.get_logger().info('Disabling motors')
            response.success = True
            response.message = 'Motors disabled'
        return response
```

### Actions

Actions handle long-running tasks with feedback:

```python
from rclpy.action import ActionServer
from humanoid_interfaces.action import WalkTo

class WalkController(Node):
    def __init__(self):
        super().__init__('walk_controller')

        self._action_server = ActionServer(
            self,
            WalkTo,
            'walk_to',
            self.execute_callback
        )

    async def execute_callback(self, goal_handle):
        self.get_logger().info('Executing walk goal...')

        feedback_msg = WalkTo.Feedback()
        result = WalkTo.Result()

        # Simulate walking with feedback
        for i in range(100):
            feedback_msg.progress = float(i) / 100.0
            goal_handle.publish_feedback(feedback_msg)
            await asyncio.sleep(0.1)

        goal_handle.succeed()
        result.success = True
        return result
```

---

## 2.4 Quality of Service (QoS)

QoS profiles control communication reliability:

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# Reliable QoS for critical data
reliable_qos = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10
)

# Best-effort QoS for sensor data
sensor_qos = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1
)

# Usage
self.joint_pub = self.create_publisher(
    JointState, '/joints', reliable_qos
)
```

### QoS Comparison

| Profile | Use Case | Trade-off |
|---------|----------|-----------|
| Reliable | Commands, safety | Higher latency |
| Best Effort | Sensors, video | Possible drops |
| Transient Local | Late subscribers | Memory usage |

---

## 2.5 Humanoid Robot Description

### URDF/Xacro Format

```xml
<?xml version="1.0"?>
<robot name="humanoid" xmlns:xacro="http://www.ros.org/wiki/xacro">

  <!-- Properties -->
  <xacro:property name="torso_height" value="0.5"/>
  <xacro:property name="leg_length" value="0.8"/>

  <!-- Base Link (Torso) -->
  <link name="torso">
    <visual>
      <geometry>
        <box size="0.3 0.2 ${torso_height}"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 0.8 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.3 0.2 ${torso_height}"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="20.0"/>
      <inertia ixx="0.5" iyy="0.5" izz="0.3"
               ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>

  <!-- Right Hip Joint -->
  <joint name="right_hip_pitch" type="revolute">
    <parent link="torso"/>
    <child link="right_thigh"/>
    <origin xyz="0.1 0 -0.25" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57"
           effort="100" velocity="3.14"/>
  </joint>

  <!-- Leg Macro -->
  <xacro:macro name="leg" params="prefix reflect">
    <link name="${prefix}_thigh">
      <visual>
        <geometry>
          <cylinder radius="0.05" length="0.4"/>
        </geometry>
      </visual>
    </link>
    <!-- Additional joints and links -->
  </xacro:macro>

  <!-- Instantiate legs -->
  <xacro:leg prefix="right" reflect="1"/>
  <xacro:leg prefix="left" reflect="-1"/>

</robot>
```

### Robot State Publisher

```python
# Launch file for robot state publisher
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import Command
import os

def generate_launch_description():
    urdf_file = os.path.join(
        get_package_share_directory('humanoid_description'),
        'urdf', 'humanoid.urdf.xacro'
    )

    return LaunchDescription([
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{
                'robot_description': Command(['xacro ', urdf_file])
            }]
        ),
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
        ),
    ])
```

---

## 2.6 Visualization with RViz2

### Basic RViz2 Configuration

```yaml
# config/humanoid.rviz
Panels:
  - Class: rviz_common/Displays
Visualization Manager:
  Displays:
    - Class: rviz_default_plugins/RobotModel
      Name: RobotModel
      Topic:
        Value: /robot_description
    - Class: rviz_default_plugins/TF
      Name: TF
      Show Names: true
    - Class: rviz_default_plugins/PointCloud2
      Name: LiDAR
      Topic:
        Value: /scan
  Global Options:
    Fixed Frame: base_link
```

### Launching RViz2

```bash
# Launch with configuration
ros2 run rviz2 rviz2 -d config/humanoid.rviz

# Or include in launch file
Node(
    package='rviz2',
    executable='rviz2',
    arguments=['-d', rviz_config_path]
)
```

---

## 2.7 Building a Humanoid Package

### Package Structure

```
humanoid_control/
├── CMakeLists.txt
├── package.xml
├── setup.py
├── humanoid_control/
│   ├── __init__.py
│   ├── controller.py
│   └── kinematics.py
├── launch/
│   └── humanoid.launch.py
├── config/
│   └── params.yaml
└── test/
    └── test_controller.py
```

### package.xml

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd"?>
<package format="3">
  <name>humanoid_control</name>
  <version>0.1.0</version>
  <description>Humanoid robot control package</description>
  <maintainer email="dev@example.com">Developer</maintainer>
  <license>MIT</license>

  <depend>rclpy</depend>
  <depend>sensor_msgs</depend>
  <depend>geometry_msgs</depend>
  <depend>tf2_ros</depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

### setup.py

```python
from setuptools import setup

package_name = 'humanoid_control'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    install_requires=['setuptools'],
    entry_points={
        'console_scripts': [
            'controller = humanoid_control.controller:main',
        ],
    },
)
```

---

## 2.8 TF2 Transform Library

### Broadcasting Transforms

```python
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped

class TransformPublisher(Node):
    def __init__(self):
        super().__init__('transform_publisher')
        self.tf_broadcaster = TransformBroadcaster(self)
        self.timer = self.create_timer(0.02, self.publish_transforms)

    def publish_transforms(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'world'
        t.child_frame_id = 'base_link'

        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 1.0

        t.transform.rotation.w = 1.0

        self.tf_broadcaster.sendTransform(t)
```

### Listening to Transforms

```python
from tf2_ros import Buffer, TransformListener

class TransformReader(Node):
    def __init__(self):
        super().__init__('transform_reader')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def get_hand_pose(self):
        try:
            trans = self.tf_buffer.lookup_transform(
                'base_link',
                'right_hand',
                rclpy.time.Time()
            )
            return trans.transform
        except Exception as e:
            self.get_logger().error(f'Transform error: {e}')
            return None
```

---

## Chapter Summary

- ROS 2 provides the foundation for humanoid robot software
- Core concepts include nodes, topics, services, and actions
- QoS profiles control communication reliability
- URDF/Xacro describes robot structure
- TF2 manages coordinate frames
- RViz2 enables 3D visualization

---

## Review Questions

1. What are the four main communication patterns in ROS 2?
2. When would you use a service versus an action?
3. What is the purpose of QoS profiles?
4. How does TF2 help with humanoid robot control?
5. What are the key components of a ROS 2 package?

---

## Hands-On Exercises

**Exercise 2.1**: Create a ROS 2 node that publishes simulated joint states for a 6-DOF arm at 100 Hz.

**Exercise 2.2**: Write a URDF file for a simple 2-link arm with revolute joints.

**Exercise 2.3**: Create a launch file that starts robot_state_publisher and RViz2 together.

---

## Lab: Simple Humanoid Simulation

```bash
# Create workspace
mkdir -p ~/humanoid_ws/src
cd ~/humanoid_ws/src

# Create package
ros2 pkg create --build-type ament_python humanoid_sim

# Build
cd ~/humanoid_ws
colcon build
source install/setup.bash

# Run
ros2 run humanoid_sim controller
```
