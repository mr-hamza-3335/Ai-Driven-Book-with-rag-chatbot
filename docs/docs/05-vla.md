# Chapter 5: Vision-Language-Action Models for Humanoid Control

## Learning Objectives

By the end of this chapter, you will be able to:
- Understand the VLA architecture and its components
- Implement vision-language models for robot perception
- Design action prediction networks for manipulation
- Train VLA models on robot demonstration data
- Deploy VLA systems on humanoid robots

---

## 5.1 Introduction to VLA Models

### What are Vision-Language-Action Models?

VLA models combine three modalities:

- **Vision**: Understanding visual scenes through images/video
- **Language**: Processing natural language instructions
- **Action**: Generating robot control commands

```
┌─────────────────────────────────────────────────────────────┐
│                    VLA Architecture                          │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│   ┌─────────┐   ┌─────────┐   ┌─────────────────────────┐   │
│   │  Image  │   │  Text   │   │     Action Tokens       │   │
│   │ Encoder │   │ Encoder │   │   (x, y, z, rx, ry, rz) │   │
│   └────┬────┘   └────┬────┘   └────────────┬────────────┘   │
│        │             │                      │                 │
│        └──────┬──────┘                      │                 │
│               ▼                             │                 │
│   ┌─────────────────────────┐               │                 │
│   │  Multimodal Transformer │◄──────────────┘                 │
│   │   (Cross-Attention)     │                                 │
│   └───────────┬─────────────┘                                 │
│               │                                               │
│               ▼                                               │
│   ┌─────────────────────────┐                                 │
│   │    Action Decoder       │                                 │
│   │  (7-DOF End-Effector)   │                                 │
│   └─────────────────────────┘                                 │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Why VLA for Humanoids?

| Challenge | VLA Solution |
|-----------|--------------|
| Complex instructions | Language understanding |
| Scene understanding | Visual perception |
| Skill generalization | Foundation model transfer |
| Multi-step tasks | Temporal reasoning |

---

## 5.2 Vision Encoders for Robotics

### Pre-trained Vision Models

```python
import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig

class RobotVisionEncoder(nn.Module):
    """Vision encoder for robot observations."""

    def __init__(self, pretrained: str = "google/vit-base-patch16-224"):
        super().__init__()

        # Load pre-trained ViT
        self.vit = ViTModel.from_pretrained(pretrained)
        self.hidden_dim = self.vit.config.hidden_size

        # Freeze early layers
        for param in self.vit.embeddings.parameters():
            param.requires_grad = False

        # Projection for robot features
        self.projection = nn.Linear(self.hidden_dim, 512)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, 3, 224, 224) RGB images
        Returns:
            features: (B, 197, 512) patch features
        """
        outputs = self.vit(pixel_values=images)
        features = outputs.last_hidden_state
        return self.projection(features)
```

### Multi-View Fusion

```python
class MultiViewEncoder(nn.Module):
    """Encode multiple camera views for humanoid perception."""

    def __init__(self, num_views: int = 3):
        super().__init__()
        self.num_views = num_views

        # Shared vision encoder
        self.vision_encoder = RobotVisionEncoder()

        # View position embeddings
        self.view_embeddings = nn.Embedding(num_views, 512)

        # Cross-view attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=512,
            num_heads=8,
            batch_first=True
        )

    def forward(self, views: torch.Tensor) -> torch.Tensor:
        """
        Args:
            views: (B, num_views, 3, 224, 224)
        Returns:
            fused_features: (B, seq_len, 512)
        """
        B = views.shape[0]

        # Encode each view
        all_features = []
        for i in range(self.num_views):
            features = self.vision_encoder(views[:, i])
            view_emb = self.view_embeddings(torch.tensor(i, device=views.device))
            features = features + view_emb
            all_features.append(features)

        # Concatenate views
        all_features = torch.cat(all_features, dim=1)

        # Cross-view attention
        fused, _ = self.cross_attention(
            all_features, all_features, all_features
        )

        return fused
```

---

## 5.3 Language Understanding

### Instruction Encoding

```python
from transformers import T5EncoderModel, T5Tokenizer

class InstructionEncoder(nn.Module):
    """Encode natural language instructions."""

    def __init__(self, model_name: str = "t5-base"):
        super().__init__()

        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.encoder = T5EncoderModel.from_pretrained(model_name)

        # Project to common dimension
        self.projection = nn.Linear(768, 512)

    def forward(self, instructions: list[str]) -> torch.Tensor:
        """
        Args:
            instructions: List of instruction strings
        Returns:
            features: (B, seq_len, 512)
        """
        # Tokenize
        tokens = self.tokenizer(
            instructions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )

        # Encode
        outputs = self.encoder(
            input_ids=tokens.input_ids.to(self.encoder.device),
            attention_mask=tokens.attention_mask.to(self.encoder.device)
        )

        return self.projection(outputs.last_hidden_state)
```

### Grounding Language to Vision

```python
class VisionLanguageGrounding(nn.Module):
    """Ground language instructions to visual features."""

    def __init__(self, hidden_dim: int = 512):
        super().__init__()

        # Cross-modal attention
        self.cross_attention = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, 8, batch_first=True)
            for _ in range(4)
        ])

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        vision_features: torch.Tensor,
        language_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            vision_features: (B, V, 512) visual patch features
            language_features: (B, L, 512) language token features
        Returns:
            grounded_features: (B, V, 512)
        """
        x = vision_features

        for attn in self.cross_attention:
            # Language attends to vision
            attended, _ = attn(language_features, x, x)

            # Vision attends to language
            x_new, _ = attn(x, attended, attended)
            x = self.norm1(x + x_new)
            x = self.norm2(x + self.ffn(x))

        return x
```

---

## 5.4 Action Prediction

### Action Tokenization

```python
class ActionTokenizer:
    """Convert continuous actions to discrete tokens."""

    def __init__(
        self,
        num_bins: int = 256,
        action_dim: int = 7  # 6 DOF + gripper
    ):
        self.num_bins = num_bins
        self.action_dim = action_dim

        # Action ranges for each dimension
        self.ranges = {
            'x': (-0.5, 0.5),      # Position (meters)
            'y': (-0.5, 0.5),
            'z': (0.0, 1.0),
            'rx': (-3.14, 3.14),   # Rotation (radians)
            'ry': (-3.14, 3.14),
            'rz': (-3.14, 3.14),
            'gripper': (0.0, 1.0)  # Gripper open/close
        }

    def encode(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            actions: (B, T, 7) continuous actions
        Returns:
            tokens: (B, T, 7) discrete action tokens
        """
        tokens = []
        for i, (name, (low, high)) in enumerate(self.ranges.items()):
            # Normalize to [0, 1]
            normalized = (actions[..., i] - low) / (high - low)
            normalized = torch.clamp(normalized, 0, 1)

            # Discretize
            token = (normalized * (self.num_bins - 1)).long()
            tokens.append(token)

        return torch.stack(tokens, dim=-1)

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """Convert tokens back to continuous actions."""
        actions = []
        for i, (name, (low, high)) in enumerate(self.ranges.items()):
            normalized = tokens[..., i].float() / (self.num_bins - 1)
            action = normalized * (high - low) + low
            actions.append(action)

        return torch.stack(actions, dim=-1)
```

### Autoregressive Action Decoder

```python
class ActionDecoder(nn.Module):
    """Decode actions autoregressively."""

    def __init__(
        self,
        hidden_dim: int = 512,
        action_dim: int = 7,
        num_bins: int = 256,
        max_steps: int = 10
    ):
        super().__init__()

        self.action_dim = action_dim
        self.num_bins = num_bins
        self.max_steps = max_steps

        # Action embedding
        self.action_embed = nn.Embedding(num_bins * action_dim, hidden_dim)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=2048,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=6)

        # Output heads for each action dimension
        self.action_heads = nn.ModuleList([
            nn.Linear(hidden_dim, num_bins)
            for _ in range(action_dim)
        ])

    def forward(
        self,
        context: torch.Tensor,
        target_actions: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            context: (B, C, 512) vision-language features
            target_actions: (B, T, 7) ground truth actions for training
        Returns:
            action_logits: (B, T, 7, num_bins)
        """
        B = context.shape[0]

        if target_actions is not None:
            # Training: teacher forcing
            T = target_actions.shape[1]

            # Embed actions
            action_indices = target_actions * self.action_dim + torch.arange(
                self.action_dim, device=target_actions.device
            )
            action_embeddings = self.action_embed(action_indices.flatten(-2))

            # Decode
            decoded = self.transformer(
                action_embeddings,
                context,
                tgt_mask=self._generate_causal_mask(T)
            )

            # Predict next action
            logits = []
            for i, head in enumerate(self.action_heads):
                logits.append(head(decoded))

            return torch.stack(logits, dim=-2)
        else:
            # Inference: autoregressive generation
            return self._generate(context)

    def _generate(self, context: torch.Tensor) -> torch.Tensor:
        """Generate actions autoregressively."""
        B = context.shape[0]
        device = context.device

        actions = []
        action_tokens = torch.zeros(B, 1, self.action_dim, dtype=torch.long, device=device)

        for t in range(self.max_steps):
            # Embed current actions
            action_embeddings = self.action_embed(
                action_tokens * self.action_dim + torch.arange(self.action_dim, device=device)
            )

            # Decode
            decoded = self.transformer(action_embeddings, context)

            # Predict next action
            next_action = []
            for i, head in enumerate(self.action_heads):
                logits = head(decoded[:, -1])
                token = torch.argmax(logits, dim=-1)
                next_action.append(token)

            next_action = torch.stack(next_action, dim=-1).unsqueeze(1)
            action_tokens = torch.cat([action_tokens, next_action], dim=1)
            actions.append(next_action)

        return torch.cat(actions, dim=1)
```

---

## 5.5 Complete VLA Model

### OpenVLA Architecture

```python
class OpenVLA(nn.Module):
    """Open-source Vision-Language-Action model."""

    def __init__(self, config):
        super().__init__()

        # Vision encoder
        self.vision_encoder = MultiViewEncoder(
            num_views=config.num_cameras
        )

        # Language encoder
        self.language_encoder = InstructionEncoder(
            model_name=config.language_model
        )

        # Vision-language grounding
        self.grounding = VisionLanguageGrounding(
            hidden_dim=config.hidden_dim
        )

        # Action decoder
        self.action_decoder = ActionDecoder(
            hidden_dim=config.hidden_dim,
            action_dim=config.action_dim,
            num_bins=config.num_bins
        )

        # Action tokenizer
        self.tokenizer = ActionTokenizer(
            num_bins=config.num_bins,
            action_dim=config.action_dim
        )

    def forward(
        self,
        images: torch.Tensor,
        instructions: list[str],
        actions: torch.Tensor = None
    ):
        """
        Args:
            images: (B, num_views, 3, 224, 224)
            instructions: List of instruction strings
            actions: (B, T, action_dim) target actions
        """
        # Encode vision
        vision_features = self.vision_encoder(images)

        # Encode language
        language_features = self.language_encoder(instructions)

        # Ground language to vision
        grounded_features = self.grounding(vision_features, language_features)

        # Tokenize actions for training
        if actions is not None:
            action_tokens = self.tokenizer.encode(actions)
        else:
            action_tokens = None

        # Decode actions
        action_logits = self.action_decoder(grounded_features, action_tokens)

        return action_logits

    def predict(
        self,
        images: torch.Tensor,
        instruction: str
    ) -> torch.Tensor:
        """Predict actions for a single instruction."""
        with torch.no_grad():
            logits = self.forward(images.unsqueeze(0), [instruction])
            tokens = torch.argmax(logits, dim=-1)
            actions = self.tokenizer.decode(tokens)
        return actions.squeeze(0)
```

---

## 5.6 Training VLA Models

### Data Collection

```python
class RobotDemonstrationDataset(torch.utils.data.Dataset):
    """Dataset of robot demonstrations."""

    def __init__(self, data_path: str, transform=None):
        self.data_path = data_path
        self.transform = transform

        # Load demonstration index
        self.demos = self._load_demos()

    def _load_demos(self):
        """Load demonstration metadata."""
        demos = []
        for demo_dir in Path(self.data_path).iterdir():
            if demo_dir.is_dir():
                metadata = json.load(open(demo_dir / "metadata.json"))
                demos.append({
                    "path": demo_dir,
                    "instruction": metadata["instruction"],
                    "num_steps": metadata["num_steps"]
                })
        return demos

    def __getitem__(self, idx):
        demo = self.demos[idx]

        # Load images
        images = []
        for cam in ["front", "left", "right"]:
            img_path = demo["path"] / f"{cam}_0.png"
            img = Image.open(img_path)
            if self.transform:
                img = self.transform(img)
            images.append(img)
        images = torch.stack(images)

        # Load actions
        actions = np.load(demo["path"] / "actions.npy")
        actions = torch.from_numpy(actions).float()

        return {
            "images": images,
            "instruction": demo["instruction"],
            "actions": actions
        }
```

### Training Loop

```python
class VLATrainer:
    """Train VLA model on demonstration data."""

    def __init__(self, model, dataset, config):
        self.model = model
        self.dataset = dataset
        self.config = config

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs
        )

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0

        for batch in tqdm(dataloader):
            images = batch["images"].to(self.config.device)
            instructions = batch["instruction"]
            actions = batch["actions"].to(self.config.device)

            # Forward pass
            action_logits = self.model(images, instructions, actions)

            # Compute loss (cross-entropy over action tokens)
            action_tokens = self.model.tokenizer.encode(actions)
            loss = F.cross_entropy(
                action_logits.reshape(-1, self.config.num_bins),
                action_tokens.reshape(-1)
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()

        self.scheduler.step()
        return total_loss / len(dataloader)

    def evaluate(self, dataloader):
        """Evaluate action prediction accuracy."""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in dataloader:
                images = batch["images"].to(self.config.device)
                instructions = batch["instruction"]
                actions = batch["actions"].to(self.config.device)

                # Predict
                action_logits = self.model(images, instructions)
                predicted = torch.argmax(action_logits, dim=-1)

                # Compare
                target_tokens = self.model.tokenizer.encode(actions)
                correct += (predicted == target_tokens).sum().item()
                total += target_tokens.numel()

        return correct / total
```

---

## 5.7 Deployment on Humanoids

### ROS 2 VLA Node

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String
from cv_bridge import CvBridge
import torch

class VLAControlNode(Node):
    """ROS 2 node for VLA-based humanoid control."""

    def __init__(self):
        super().__init__('vla_control')

        # Load model
        self.model = OpenVLA.load_pretrained("openvla-7b")
        self.model.eval()
        self.model.to("cuda")

        self.bridge = CvBridge()
        self.current_images = {}

        # Subscribers
        self.image_subs = {
            "front": self.create_subscription(
                Image, '/camera/front/image',
                lambda msg: self.image_callback(msg, "front"), 10
            ),
            "left": self.create_subscription(
                Image, '/camera/left/image',
                lambda msg: self.image_callback(msg, "left"), 10
            ),
        }

        self.instruction_sub = self.create_subscription(
            String, '/instruction', self.instruction_callback, 10
        )

        # Publisher
        self.action_pub = self.create_publisher(
            JointState, '/joint_commands', 10
        )

        self.current_instruction = None

        # Control loop
        self.timer = self.create_timer(0.1, self.control_loop)  # 10 Hz

    def image_callback(self, msg, camera_name):
        """Store latest image from each camera."""
        cv_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        self.current_images[camera_name] = cv_image

    def instruction_callback(self, msg):
        """Receive new instruction."""
        self.current_instruction = msg.data
        self.get_logger().info(f"New instruction: {msg.data}")

    def control_loop(self):
        """Execute VLA model and publish actions."""
        if self.current_instruction is None:
            return

        if len(self.current_images) < 2:
            return

        # Prepare images
        images = []
        for cam in ["front", "left"]:
            img = self.current_images.get(cam)
            if img is not None:
                img = self.preprocess(img)
                images.append(img)

        images = torch.stack(images).unsqueeze(0).to("cuda")

        # Predict action
        with torch.no_grad():
            action = self.model.predict(images, self.current_instruction)

        # Convert to joint command
        joint_msg = self.action_to_joint_state(action)
        self.action_pub.publish(joint_msg)

    def action_to_joint_state(self, action):
        """Convert VLA action to JointState message."""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [
            "shoulder_pan", "shoulder_lift", "elbow",
            "wrist_1", "wrist_2", "wrist_3", "gripper"
        ]
        msg.position = action.cpu().numpy().tolist()
        return msg

def main():
    rclpy.init()
    node = VLAControlNode()
    rclpy.spin(node)
    rclpy.shutdown()
```

---

## 5.8 Advanced Topics

### Chain-of-Thought Reasoning

```python
class VLAWithReasoning(OpenVLA):
    """VLA with intermediate reasoning steps."""

    def __init__(self, config):
        super().__init__(config)

        # Reasoning head
        self.reasoning_head = nn.Linear(config.hidden_dim, config.vocab_size)

    def forward_with_reasoning(self, images, instruction):
        """Generate reasoning chain before action."""
        # Encode inputs
        vision_features = self.vision_encoder(images)
        language_features = self.language_encoder([instruction])
        grounded = self.grounding(vision_features, language_features)

        # Generate reasoning tokens
        reasoning_tokens = self._generate_reasoning(grounded)

        # Decode reasoning to text
        reasoning_text = self.tokenizer.decode(reasoning_tokens)

        # Append reasoning to context
        enhanced_features = self._incorporate_reasoning(
            grounded, reasoning_tokens
        )

        # Generate actions
        actions = self.action_decoder(enhanced_features)

        return actions, reasoning_text
```

### Multi-Task Learning

```python
class MultiTaskVLA(OpenVLA):
    """VLA for multiple manipulation tasks."""

    TASKS = [
        "pick_and_place",
        "pour",
        "open_drawer",
        "press_button",
        "wipe_surface"
    ]

    def __init__(self, config):
        super().__init__(config)

        # Task-specific heads
        self.task_heads = nn.ModuleDict({
            task: ActionDecoder(
                hidden_dim=config.hidden_dim,
                action_dim=config.action_dim
            )
            for task in self.TASKS
        })

        # Task classifier
        self.task_classifier = nn.Linear(config.hidden_dim, len(self.TASKS))

    def forward(self, images, instruction, task=None):
        # Encode
        grounded = self._encode(images, instruction)

        # Classify task if not provided
        if task is None:
            task_logits = self.task_classifier(grounded.mean(dim=1))
            task_idx = torch.argmax(task_logits, dim=-1)
            task = self.TASKS[task_idx]

        # Use task-specific decoder
        actions = self.task_heads[task](grounded)

        return actions, task
```

---

## Chapter Summary

- VLA models combine vision, language, and action for robot control
- Vision encoders extract spatial features from camera images
- Language encoders understand natural language instructions
- Action decoders generate discretized robot commands
- Training requires diverse demonstration datasets
- Deployment uses ROS 2 for real-time control

---

## Review Questions

1. What are the three modalities in VLA models?
2. How does action tokenization work?
3. What is vision-language grounding?
4. How are VLA models trained on demonstration data?
5. What are the challenges in deploying VLA on real robots?

---

## Hands-On Exercises

**Exercise 5.1**: Implement a simple vision encoder using ResNet-18.

**Exercise 5.2**: Create an action tokenizer for a 7-DOF arm.

**Exercise 5.3**: Train a small VLA model on a pick-and-place dataset.

**Exercise 5.4**: Deploy your VLA model in a ROS 2 simulation.

---

## Lab: Building a Mini-VLA

Complete implementation project:
1. Create vision encoder with ViT-Small
2. Implement instruction encoder with DistilBERT
3. Build action decoder with 4-layer transformer
4. Train on CALVIN benchmark dataset
5. Evaluate on held-out manipulation tasks
