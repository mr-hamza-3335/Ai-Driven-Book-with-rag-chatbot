# Chapter 3: Gazebo Simulation for Humanoid Robots

## Learning Objectives

By the end of this chapter, you will be able to:
- Set up Gazebo simulation environments for humanoids
- Create and configure robot models with physics properties
- Implement sensor plugins for cameras, IMU, and force sensors
- Design test environments and scenarios
- Debug and optimize simulation performance

---

## 3.1 Introduction to Gazebo

### What is Gazebo?

Gazebo is a powerful 3D robotics simulator that provides:

- **Physics engines**: ODE, Bullet, DART, Simbody
- **Sensor simulation**: Cameras, LiDAR, IMU, contact sensors
- **Realistic rendering**: Lighting, shadows, materials
- **ROS integration**: Seamless communication with ROS 2

### Gazebo Versions

| Version | Status | ROS 2 Support |
|---------|--------|---------------|
| Gazebo Classic | Legacy | ROS 2 via bridges |
| Ignition Gazebo | Current | Native support |
| Gazebo Harmonic | Latest | Full ROS 2 Humble |

---

## 3.2 Installation and Setup

### Installing Gazebo Harmonic

```bash
# Add Gazebo repository
sudo wget https://packages.osrfoundation.org/gazebo.gpg \
  -O /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] \
  http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" \
  | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null

# Install Gazebo
sudo apt-get update
sudo apt-get install gz-harmonic

# Install ROS 2 integration
sudo apt-get install ros-humble-ros-gz
```

### Verify Installation

```bash
# Launch empty world
gz sim empty.sdf

# Check ROS 2 bridge
ros2 run ros_gz_bridge parameter_bridge --help
```

---

## 3.3 SDF Robot Models

### SDF vs URDF

| Feature | URDF | SDF |
|---------|------|-----|
| Origin | ROS | Gazebo |
| Physics | Basic | Advanced |
| Sensors | Limited | Full support |
| Plugins | Separate | Integrated |
| Worlds | No | Yes |

### Basic Humanoid SDF

```xml
<?xml version="1.0"?>
<sdf version="1.9">
  <model name="simple_humanoid">
    <static>false</static>

    <!-- Torso -->
    <link name="torso">
      <pose>0 0 1.0 0 0 0</pose>
      <inertial>
        <mass>30.0</mass>
        <inertia>
          <ixx>1.0</ixx>
          <iyy>1.0</iyy>
          <izz>0.5</izz>
        </inertia>
      </inertial>
      <visual name="torso_visual">
        <geometry>
          <box><size>0.3 0.4 0.5</size></box>
        </geometry>
        <material>
          <ambient>0.2 0.2 0.8 1</ambient>
        </material>
      </visual>
      <collision name="torso_collision">
        <geometry>
          <box><size>0.3 0.4 0.5</size></box>
        </geometry>
      </collision>
    </link>

    <!-- Right Thigh -->
    <link name="right_thigh">
      <pose>0.1 0 0.6 0 0 0</pose>
      <inertial>
        <mass>5.0</mass>
        <inertia>
          <ixx>0.1</ixx>
          <iyy>0.1</iyy>
          <izz>0.02</izz>
        </inertia>
      </inertial>
      <visual name="right_thigh_visual">
        <geometry>
          <cylinder>
            <radius>0.05</radius>
            <length>0.35</length>
          </cylinder>
        </geometry>
      </visual>
      <collision name="right_thigh_collision">
        <geometry>
          <cylinder>
            <radius>0.05</radius>
            <length>0.35</length>
          </cylinder>
        </geometry>
      </collision>
    </link>

    <!-- Right Hip Joint -->
    <joint name="right_hip_pitch" type="revolute">
      <parent>torso</parent>
      <child>right_thigh</child>
      <axis>
        <xyz>0 1 0</xyz>
        <limit>
          <lower>-1.57</lower>
          <upper>1.57</upper>
          <effort>200</effort>
          <velocity>5.0</velocity>
        </limit>
        <dynamics>
          <damping>0.1</damping>
          <friction>0.1</friction>
        </dynamics>
      </axis>
    </joint>

    <!-- Joint Controller Plugin -->
    <plugin
      filename="gz-sim-joint-position-controller-system"
      name="gz::sim::systems::JointPositionController">
      <joint_name>right_hip_pitch</joint_name>
      <p_gain>100</p_gain>
      <d_gain>10</d_gain>
    </plugin>

  </model>
</sdf>
```

---

## 3.4 Physics Properties

### Inertia Calculation

Proper inertia is critical for realistic simulation:

```python
# Calculate inertia for common shapes
import numpy as np

def box_inertia(mass, x, y, z):
    """Calculate inertia tensor for a box."""
    ixx = (1/12) * mass * (y**2 + z**2)
    iyy = (1/12) * mass * (x**2 + z**2)
    izz = (1/12) * mass * (x**2 + y**2)
    return ixx, iyy, izz

def cylinder_inertia(mass, radius, length):
    """Calculate inertia tensor for a cylinder (z-axis)."""
    ixx = (1/12) * mass * (3*radius**2 + length**2)
    iyy = ixx
    izz = (1/2) * mass * radius**2
    return ixx, iyy, izz

# Example: humanoid thigh
mass = 5.0  # kg
radius = 0.05  # m
length = 0.35  # m
print(cylinder_inertia(mass, radius, length))
# Output: (0.054, 0.054, 0.00625)
```

### Contact Properties

```xml
<collision name="foot_collision">
  <geometry>
    <box><size>0.15 0.08 0.03</size></box>
  </geometry>
  <surface>
    <friction>
      <ode>
        <mu>1.0</mu>
        <mu2>1.0</mu2>
      </ode>
    </friction>
    <contact>
      <ode>
        <kp>1e6</kp>
        <kd>100</kd>
        <max_vel>0.1</max_vel>
        <min_depth>0.001</min_depth>
      </ode>
    </contact>
  </surface>
</collision>
```

---

## 3.5 Sensor Plugins

### IMU Sensor

```xml
<link name="imu_link">
  <pose>0 0 1.0 0 0 0</pose>
  <sensor name="imu_sensor" type="imu">
    <always_on>true</always_on>
    <update_rate>400</update_rate>
    <imu>
      <angular_velocity>
        <x><noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.01</stddev>
        </noise></x>
      </angular_velocity>
      <linear_acceleration>
        <x><noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.1</stddev>
        </noise></x>
      </linear_acceleration>
    </imu>
    <plugin
      filename="gz-sim-imu-system"
      name="gz::sim::systems::Imu">
    </plugin>
  </sensor>
</link>
```

### RGB-D Camera

```xml
<link name="camera_link">
  <pose>0 0 1.5 0 0 0</pose>
  <sensor name="rgbd_camera" type="rgbd_camera">
    <update_rate>30</update_rate>
    <camera>
      <horizontal_fov>1.047</horizontal_fov>
      <image>
        <width>640</width>
        <height>480</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>10</far>
      </clip>
      <depth_camera>
        <clip>
          <near>0.1</near>
          <far>10</far>
        </clip>
      </depth_camera>
    </camera>
    <plugin
      filename="gz-sim-sensors-system"
      name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>
  </sensor>
</link>
```

### Force/Torque Sensor

```xml
<joint name="ankle_ft_sensor" type="fixed">
  <parent>shank</parent>
  <child>foot</child>
  <sensor name="ft_sensor" type="force_torque">
    <update_rate>1000</update_rate>
    <force_torque>
      <frame>child</frame>
      <measure_direction>child_to_parent</measure_direction>
    </force_torque>
  </sensor>
</joint>
```

---

## 3.6 World Creation

### Indoor Environment

```xml
<?xml version="1.0"?>
<sdf version="1.9">
  <world name="humanoid_lab">
    <!-- Physics -->
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
    </physics>

    <!-- Lighting -->
    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>-0.5 0.1 -0.9</direction>
    </light>

    <!-- Ground Plane -->
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <plane><normal>0 0 1</normal></plane>
          </geometry>
          <surface>
            <friction>
              <ode><mu>1.0</mu></ode>
            </friction>
          </surface>
        </collision>
        <visual name="visual">
          <geometry>
            <plane><normal>0 0 1</normal><size>20 20</size></plane>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
          </material>
        </visual>
      </link>
    </model>

    <!-- Stairs -->
    <model name="stairs">
      <static>true</static>
      <pose>3 0 0 0 0 0</pose>
      <link name="step1">
        <pose>0 0 0.1 0 0 0</pose>
        <collision name="collision">
          <geometry><box><size>0.3 1.0 0.2</size></box></geometry>
        </collision>
        <visual name="visual">
          <geometry><box><size>0.3 1.0 0.2</size></box></geometry>
        </visual>
      </link>
      <link name="step2">
        <pose>0.3 0 0.3 0 0 0</pose>
        <collision name="collision">
          <geometry><box><size>0.3 1.0 0.2</size></box></geometry>
        </collision>
        <visual name="visual">
          <geometry><box><size>0.3 1.0 0.2</size></box></geometry>
        </visual>
      </link>
    </model>

    <!-- Include humanoid -->
    <include>
      <uri>model://simple_humanoid</uri>
      <pose>0 0 0.05 0 0 0</pose>
    </include>

  </world>
</sdf>
```

---

## 3.7 ROS 2 Bridge Configuration

### Bridge Topics

```yaml
# bridge_config.yaml
- ros_topic_name: "/joint_states"
  gz_topic_name: "/world/humanoid_lab/model/simple_humanoid/joint_state"
  ros_type_name: "sensor_msgs/msg/JointState"
  gz_type_name: "gz.msgs.Model"
  direction: GZ_TO_ROS

- ros_topic_name: "/cmd_vel"
  gz_topic_name: "/model/simple_humanoid/cmd_vel"
  ros_type_name: "geometry_msgs/msg/Twist"
  gz_type_name: "gz.msgs.Twist"
  direction: ROS_TO_GZ

- ros_topic_name: "/imu/data"
  gz_topic_name: "/world/humanoid_lab/model/simple_humanoid/link/imu_link/sensor/imu_sensor/imu"
  ros_type_name: "sensor_msgs/msg/Imu"
  gz_type_name: "gz.msgs.IMU"
  direction: GZ_TO_ROS
```

### Launch with Bridge

```python
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # Start Gazebo
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('ros_gz_sim'),
                'launch',
                'gz_sim.launch.py'
            ])
        ]),
        launch_arguments={'gz_args': 'humanoid_lab.sdf'}.items()
    )

    # Start bridge
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/joint_states@sensor_msgs/msg/JointState@gz.msgs.Model',
            '/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist',
            '/imu/data@sensor_msgs/msg/Imu@gz.msgs.IMU',
        ],
        output='screen'
    )

    return LaunchDescription([gz_sim, bridge])
```

---

## 3.8 Performance Optimization

### Simulation Speed Tips

| Optimization | Benefit | Trade-off |
|--------------|---------|-----------|
| Larger time step | Faster simulation | Less accuracy |
| Simplified collision | Faster physics | Less realism |
| Lower sensor rate | Less computation | Coarser data |
| Headless mode | No rendering cost | No visualization |

### Headless Simulation

```bash
# Run without GUI
gz sim -s humanoid_lab.sdf

# Or with launch file
gz_args: '-r -s humanoid_lab.sdf'  # -s for server only
```

### Collision Optimization

```xml
<!-- Use simplified collision geometry -->
<collision name="torso_collision">
  <geometry>
    <!-- Use box instead of mesh -->
    <box><size>0.3 0.4 0.5</size></box>
  </geometry>
</collision>
```

---

## 3.9 Recording and Playback

### State Logging

```bash
# Record simulation state
gz sim -r humanoid_lab.sdf --record

# Playback
gz sim -p state.tlog
```

### ROS 2 Bag Recording

```bash
# Record all topics
ros2 bag record -a

# Record specific topics
ros2 bag record /joint_states /imu/data /camera/image_raw

# Playback
ros2 bag play rosbag2_2024_01_01-12_00_00
```

---

## Chapter Summary

- Gazebo provides realistic physics simulation for humanoids
- SDF format offers advanced features over URDF
- Proper physics properties are essential for realism
- Sensor plugins simulate cameras, IMU, and force sensors
- ROS 2 bridge enables seamless communication
- Performance optimization is crucial for real-time factors

---

## Review Questions

1. What are the advantages of SDF over URDF?
2. How do you calculate proper inertia values for robot links?
3. What sensor plugins are essential for humanoid simulation?
4. How does the ROS 2 bridge work with Gazebo?
5. What techniques optimize simulation performance?

---

## Hands-On Exercises

**Exercise 3.1**: Create an SDF model for a simple 2-link leg and simulate it falling under gravity.

**Exercise 3.2**: Add an IMU sensor to your model and visualize the data in ROS 2.

**Exercise 3.3**: Create a world with stairs and test your humanoid model climbing them.

---

## Lab: Walking Simulation

Build a complete simulation environment with:
1. Humanoid model with leg joints
2. IMU and foot force sensors
3. Flat ground and stair obstacles
4. ROS 2 bridge for joint control
5. Recording capability for analysis
