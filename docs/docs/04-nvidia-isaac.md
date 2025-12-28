# Chapter 4: NVIDIA Isaac for Humanoid Development

## Learning Objectives

By the end of this chapter, you will be able to:
- Set up NVIDIA Isaac Sim for humanoid robot simulation
- Use Isaac Gym for reinforcement learning training
- Implement sim-to-real transfer techniques
- Leverage GPU-accelerated physics for parallel training
- Deploy trained policies to real hardware

---

## 4.1 Introduction to NVIDIA Isaac

### What is NVIDIA Isaac?

NVIDIA Isaac is a comprehensive robotics platform that includes:

- **Isaac Sim**: High-fidelity simulation built on Omniverse
- **Isaac Gym**: GPU-accelerated RL training environment
- **Isaac ROS**: Optimized ROS 2 packages for NVIDIA hardware
- **Isaac SDK**: Tools for robot development and deployment

### Why Isaac for Humanoids?

| Feature | Benefit for Humanoids |
|---------|----------------------|
| PhysX GPU | Parallel physics for thousands of robots |
| RTX rendering | Photorealistic vision training |
| USD format | Industry-standard asset pipeline |
| RL tooling | End-to-end policy training |

---

## 4.2 Isaac Sim Setup

### System Requirements

```
Minimum Requirements:
- GPU: NVIDIA RTX 2070 or higher
- CPU: Intel Core i7 or AMD Ryzen 7
- RAM: 32 GB
- Storage: 50 GB SSD
- OS: Ubuntu 20.04/22.04

Recommended:
- GPU: NVIDIA RTX 4090 or A6000
- RAM: 64 GB
- Storage: 100 GB NVMe SSD
```

### Installation

```bash
# Install Omniverse Launcher
wget https://install.launcher.omniverse.nvidia.com/installers/\
omniverse-launcher-linux.AppImage

chmod +x omniverse-launcher-linux.AppImage
./omniverse-launcher-linux.AppImage

# Install Isaac Sim from Omniverse Launcher
# Navigate to Exchange -> Isaac Sim -> Install

# Verify installation
~/.local/share/ov/pkg/isaac_sim-2023.1.1/isaac-sim.sh --help
```

### Python Environment

```bash
# Create conda environment
conda create -n isaac python=3.10
conda activate isaac

# Install Isaac Sim Python packages
pip install isaacsim-rl isaacsim-robot isaacsim-sensor

# Or use the bundled Python
~/.local/share/ov/pkg/isaac_sim-2023.1.1/python.sh -m pip install torch
```

---

## 4.3 Humanoid Assets in USD

### USD Format Benefits

Universal Scene Description (USD) provides:

- Hierarchical scene composition
- Non-destructive editing
- Collaboration workflows
- Industry standard (Pixar)

### Converting URDF to USD

```python
from omni.isaac.urdf import _urdf
from pxr import Usd, UsdPhysics

def convert_humanoid_urdf(urdf_path: str, usd_path: str):
    """Convert URDF to USD with physics properties."""

    # Import URDF
    import_config = _urdf.ImportConfig()
    import_config.merge_fixed_joints = False
    import_config.fix_base = False
    import_config.make_default_prim = True
    import_config.create_physics_scene = True

    # Set physics properties
    import_config.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION
    import_config.default_drive_strength = 1000.0
    import_config.default_position_drive_damping = 100.0

    # Convert
    result = _urdf.import_robot(
        urdf_path,
        usd_path,
        import_config
    )

    return result
```

### Humanoid USD Structure

```
humanoid.usd
├── /World
│   ├── /PhysicsScene
│   ├── /GroundPlane
│   └── /Humanoid
│       ├── /torso (root link)
│       │   ├── /left_hip_joint (revolute)
│       │   │   └── /left_thigh
│       │   │       └── /left_knee_joint
│       │   │           └── /left_shin
│       │   └── /right_hip_joint
│       │       └── /right_thigh
│       └── /Sensors
│           ├── /imu_sensor
│           └── /camera_sensor
```

---

## 4.4 Isaac Gym for RL Training

### Parallel Environment Setup

```python
import isaacgym
from isaacgym import gymapi, gymtorch
import torch

class HumanoidEnv:
    """Parallel humanoid training environment."""

    def __init__(self, num_envs: int = 4096, device: str = "cuda"):
        self.num_envs = num_envs
        self.device = device

        # Initialize gym
        self.gym = gymapi.acquire_gym()

        # Simulation parameters
        sim_params = gymapi.SimParams()
        sim_params.dt = 1.0 / 60.0
        sim_params.substeps = 2
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

        # PhysX parameters
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.contact_offset = 0.01
        sim_params.physx.rest_offset = 0.0

        # Create sim
        self.sim = self.gym.create_sim(
            0, 0,  # GPU device IDs
            gymapi.SIM_PHYSX,
            sim_params
        )

        self._create_envs()

    def _create_envs(self):
        """Create parallel environments."""
        # Load humanoid asset
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.angular_damping = 0.01

        humanoid_asset = self.gym.load_asset(
            self.sim,
            "assets",
            "humanoid.urdf",
            asset_options
        )

        # Environment spacing
        env_spacing = 2.0
        env_lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
        env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

        self.envs = []
        self.actors = []

        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, env_lower, env_upper, int(self.num_envs ** 0.5))

            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0.0, 0.0, 1.0)

            actor = self.gym.create_actor(env, humanoid_asset, pose, f"humanoid_{i}", i, 0)

            self.envs.append(env)
            self.actors.append(actor)
```

### Observation and Action Spaces

```python
class HumanoidEnv:
    # ... continued

    def _get_observations(self) -> torch.Tensor:
        """Get observation tensor for all environments."""
        # Root state: position (3) + orientation (4) + linear vel (3) + angular vel (3)
        root_states = self.root_tensor[:, :13]

        # Joint states: positions (num_dof) + velocities (num_dof)
        dof_pos = self.dof_pos_tensor
        dof_vel = self.dof_vel_tensor

        # Concatenate observations
        obs = torch.cat([
            root_states[:, 2:3],    # Height
            root_states[:, 3:7],    # Orientation (quaternion)
            root_states[:, 7:10],   # Linear velocity
            root_states[:, 10:13],  # Angular velocity
            dof_pos,                 # Joint positions
            dof_vel,                 # Joint velocities
        ], dim=-1)

        return obs

    def _compute_reward(self) -> torch.Tensor:
        """Compute reward for walking task."""
        # Forward velocity reward
        forward_vel = self.root_tensor[:, 7]  # x velocity
        vel_reward = torch.exp(-torch.abs(forward_vel - 1.0))

        # Upright reward
        up_vec = self._get_up_vector()
        upright_reward = torch.sum(up_vec * torch.tensor([0, 0, 1], device=self.device), dim=-1)

        # Energy penalty
        energy_penalty = torch.sum(torch.abs(self.torques) * torch.abs(self.dof_vel_tensor), dim=-1)

        # Total reward
        reward = vel_reward + 0.5 * upright_reward - 0.01 * energy_penalty

        return reward

    def step(self, actions: torch.Tensor):
        """Step all environments."""
        # Apply actions as joint torques
        self.gym.set_dof_actuation_force_tensor(
            self.sim,
            gymtorch.unwrap_tensor(actions)
        )

        # Simulate
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        # Refresh tensors
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

        # Get new observations and rewards
        obs = self._get_observations()
        rewards = self._compute_reward()
        dones = self._check_termination()

        return obs, rewards, dones, {}
```

### PPO Training Loop

```python
from rl_games.algos_torch import ppo

class HumanoidTrainer:
    """PPO trainer for humanoid locomotion."""

    def __init__(self, env, config):
        self.env = env
        self.config = config

        # Policy network
        self.policy = ActorCritic(
            obs_dim=env.obs_dim,
            action_dim=env.action_dim,
            hidden_dims=[256, 256, 128]
        ).to(env.device)

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=3e-4
        )

    def train(self, num_iterations: int = 10000):
        """Train the policy."""
        for iteration in range(num_iterations):
            # Collect rollouts
            with torch.no_grad():
                obs = self.env.reset()

                for step in range(self.config.horizon):
                    actions, log_probs, values = self.policy(obs)
                    obs, rewards, dones, _ = self.env.step(actions)

                    # Store transition
                    self.buffer.add(obs, actions, rewards, dones, log_probs, values)

            # PPO update
            for epoch in range(self.config.num_epochs):
                for batch in self.buffer.get_batches():
                    loss = self._compute_ppo_loss(batch)

                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                    self.optimizer.step()

            # Logging
            if iteration % 100 == 0:
                print(f"Iteration {iteration}, Reward: {rewards.mean():.2f}")
```

---

## 4.5 Domain Randomization

### Physics Randomization

```python
class DomainRandomization:
    """Domain randomization for sim-to-real transfer."""

    def __init__(self, env):
        self.env = env

        # Randomization ranges
        self.mass_range = (0.8, 1.2)  # Scale factor
        self.friction_range = (0.5, 1.5)
        self.motor_strength_range = (0.9, 1.1)
        self.gravity_range = (-10.0, -9.5)

    def randomize(self):
        """Apply randomization to all environments."""
        for i, env in enumerate(self.env.envs):
            # Randomize mass
            mass_scale = np.random.uniform(*self.mass_range)
            self._scale_link_masses(env, mass_scale)

            # Randomize friction
            friction = np.random.uniform(*self.friction_range)
            self._set_friction(env, friction)

            # Randomize motor strength
            motor_scale = np.random.uniform(*self.motor_strength_range)
            self._scale_motor_strength(env, motor_scale)

    def _scale_link_masses(self, env, scale):
        """Scale masses of all links."""
        props = self.env.gym.get_actor_rigid_body_properties(env, 0)
        for prop in props:
            prop.mass *= scale
        self.env.gym.set_actor_rigid_body_properties(env, 0, props)
```

### Visual Randomization

```python
def randomize_visual(env, camera):
    """Randomize visual appearance for vision training."""
    # Lighting
    light_intensity = np.random.uniform(0.5, 2.0)
    light_color = np.random.uniform(0.8, 1.0, size=3)

    # Textures
    texture_options = ["wood", "metal", "concrete", "tile"]
    floor_texture = np.random.choice(texture_options)

    # Camera noise
    camera_noise_std = np.random.uniform(0.01, 0.05)

    return {
        "light_intensity": light_intensity,
        "light_color": light_color,
        "floor_texture": floor_texture,
        "camera_noise": camera_noise_std
    }
```

---

## 4.6 Sim-to-Real Transfer

### Teacher-Student Training

```python
class TeacherStudent:
    """Teacher-student framework for sim-to-real."""

    def __init__(self):
        # Teacher has access to privileged information
        self.teacher = ActorCritic(
            obs_dim=PRIVILEGED_OBS_DIM,  # Includes ground truth
            action_dim=ACTION_DIM
        )

        # Student only sees real-world observations
        self.student = ActorCritic(
            obs_dim=REAL_OBS_DIM,  # Only sensors
            action_dim=ACTION_DIM
        )

    def train_teacher(self, env):
        """Train teacher with privileged observations."""
        # Teacher sees: true contact states, terrain info, etc.
        pass

    def distill_to_student(self, env):
        """Distill teacher knowledge to student."""
        for batch in self.dataset:
            # Student tries to match teacher actions
            teacher_actions = self.teacher(batch.privileged_obs)
            student_actions = self.student(batch.real_obs)

            loss = F.mse_loss(student_actions, teacher_actions.detach())

            self.student_optimizer.zero_grad()
            loss.backward()
            self.student_optimizer.step()
```

### Policy Export

```python
def export_policy_for_deployment(policy, path: str):
    """Export trained policy for real robot deployment."""
    # Trace the model
    example_obs = torch.zeros(1, OBS_DIM)
    traced_model = torch.jit.trace(policy.actor, example_obs)

    # Save
    traced_model.save(path)

    print(f"Policy exported to {path}")

def create_ros2_node(policy_path: str):
    """Generate ROS 2 node for policy deployment."""
    template = '''
import rclpy
from rclpy.node import Node
import torch

class PolicyNode(Node):
    def __init__(self):
        super().__init__('policy_node')
        self.policy = torch.jit.load("{policy_path}")
        self.policy.eval()

        self.obs_sub = self.create_subscription(
            JointState, '/joint_states', self.obs_callback, 10
        )
        self.action_pub = self.create_publisher(
            JointTrajectory, '/joint_trajectory', 10
        )

    def obs_callback(self, msg):
        obs = self._msg_to_tensor(msg)
        with torch.no_grad():
            action = self.policy(obs)
        self._publish_action(action)
'''
    return template.format(policy_path=policy_path)
```

---

## 4.7 Isaac ROS Integration

### Isaac ROS Packages

```bash
# Install Isaac ROS
mkdir -p ~/workspaces/isaac_ros-dev/src
cd ~/workspaces/isaac_ros-dev/src

git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common.git
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_visual_slam.git
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_nvblox.git

# Build with Docker
cd ~/workspaces/isaac_ros-dev/src/isaac_ros_common
./scripts/run_dev.sh
colcon build --symlink-install
```

### Visual SLAM with Isaac ROS

```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    """Launch visual SLAM for humanoid navigation."""

    visual_slam = Node(
        package='isaac_ros_visual_slam',
        executable='isaac_ros_visual_slam',
        parameters=[{
            'denoise_input_images': True,
            'rectified_images': True,
            'enable_imu_fusion': True,
            'gyro_noise_density': 0.00016,
            'accelerometer_noise_density': 0.00024,
        }],
        remappings=[
            ('camera/image_raw', '/humanoid/camera/image'),
            ('camera/camera_info', '/humanoid/camera/info'),
            ('imu', '/humanoid/imu'),
        ]
    )

    return LaunchDescription([visual_slam])
```

---

## 4.8 Performance Optimization

### Multi-GPU Training

```python
class MultiGPUTrainer:
    """Distributed training across multiple GPUs."""

    def __init__(self, num_gpus: int = 4):
        self.num_gpus = num_gpus

        # Create environments on each GPU
        self.envs = []
        for gpu_id in range(num_gpus):
            env = HumanoidEnv(
                num_envs=4096,
                device=f"cuda:{gpu_id}"
            )
            self.envs.append(env)

        # Total environments
        self.total_envs = 4096 * num_gpus  # 16,384 parallel robots

    def collect_experience(self):
        """Collect experience from all GPUs."""
        all_obs = []
        all_rewards = []

        for env in self.envs:
            obs, rewards, _, _ = env.step(self.policy(env.obs))
            all_obs.append(obs)
            all_rewards.append(rewards)

        return torch.cat(all_obs), torch.cat(all_rewards)
```

### Memory Optimization

| Technique | Memory Saving | Implementation |
|-----------|---------------|----------------|
| Mixed precision | 50% | `torch.cuda.amp` |
| Gradient checkpointing | 30% | `torch.utils.checkpoint` |
| Shared memory tensors | 20% | `gymtorch.wrap_tensor` |
| Batch size tuning | Variable | Dynamic batching |

---

## Chapter Summary

- NVIDIA Isaac provides GPU-accelerated simulation for humanoids
- Isaac Gym enables massively parallel RL training
- Domain randomization improves sim-to-real transfer
- Teacher-student training bridges the reality gap
- Isaac ROS integrates with real robot deployment

---

## Review Questions

1. What are the main components of the NVIDIA Isaac platform?
2. How does Isaac Gym achieve parallel environment simulation?
3. What is domain randomization and why is it important?
4. Explain the teacher-student framework for sim-to-real transfer.
5. How do you export a trained policy for real robot deployment?

---

## Hands-On Exercises

**Exercise 4.1**: Set up Isaac Sim and load a humanoid model from URDF.

**Exercise 4.2**: Implement a parallel environment with 1024 humanoids in Isaac Gym.

**Exercise 4.3**: Train a walking policy using PPO with domain randomization.

**Exercise 4.4**: Export your trained policy and create a ROS 2 node for deployment.

---

## Lab: End-to-End RL Training Pipeline

Build a complete training pipeline:
1. Load humanoid asset in Isaac Gym
2. Define observation/action spaces for walking
3. Implement reward function for forward locomotion
4. Apply domain randomization
5. Train with PPO for 10,000 iterations
6. Export policy for ROS 2 deployment
