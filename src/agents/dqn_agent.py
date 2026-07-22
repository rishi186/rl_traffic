"""Deep Q-Network (DQN) agent implemented in PyTorch.

Supports standard and Dueling DQN, Double DQN, Prioritized Experience Replay
(PER), epsilon-greedy exploration, and periodic target-network updates.
"""

import os
from typing import Dict, Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.agents.replay_buffer import ReplayBuffer
from src.agents.prioritized_replay_buffer import PrioritizedReplayBuffer
from src.agents.nstep_buffer import NStepCollector
from src.agents.layers.noisy_linear import NoisyLinear
from src.utils.logger import get_logger

logger = get_logger(__name__)


class QNetwork(nn.Module):
    """Q-Network (standard or dueling architecture) in PyTorch.

    Args:
        obs_dim: Dimensionality of the observation/state vector.
        action_dim: Number of discrete actions.
        hidden_dims: List of hidden layer widths.
        dueling: Whether to use Dueling DQN architecture.
        noisy_net: Whether to use NoisyNet layers for exploration.
        noisy_sigma: Initial noise standard deviation for NoisyNet.
        use_attention: Whether to use self-attention on per-lane features.
        num_lanes: Number of lanes per observation (for attention reshaping).
        num_lane_features: Features per lane (for attention reshaping).
        num_agents: Number of agents for parameter sharing (0 = no sharing).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: List[int],
        dueling: bool,
        noisy_net: bool,
        noisy_sigma: float,
        use_attention: bool = False,
        num_lanes: int = 0,
        num_lane_features: int = 5,
        num_agents: int = 0,
    ) -> None:
        super().__init__()
        self.dueling = dueling
        self.noisy_net = noisy_net
        self.action_dim = action_dim
        self.use_attention = use_attention
        self.num_lanes = num_lanes
        self.num_lane_features = num_lane_features
        self.num_agents = num_agents

        if use_attention and num_lanes > 0:
            lane_feat_dim = num_lanes * num_lane_features
            global_feat_dim = obs_dim - lane_feat_dim

            num_heads = 2 if num_lane_features % 2 == 0 else 1
            self.lane_attention = nn.MultiheadAttention(
                embed_dim=num_lane_features,
                num_heads=num_heads,
                batch_first=True,
            )
            self.lane_norm = nn.LayerNorm(num_lane_features)

            attended_dim = num_lanes * num_lane_features + global_feat_dim
            if num_agents > 0:
                attended_dim += num_agents

            layers = []
            in_dim = attended_dim
            for units in hidden_dims:
                if noisy_net:
                    layers.append(NoisyLinear(in_dim, units, sigma_init=noisy_sigma))
                else:
                    layers.append(nn.Linear(in_dim, units))
                layers.append(nn.ReLU())
                if not noisy_net:
                    layers.append(nn.Dropout(0.1))
                in_dim = units
            self.feature_network = nn.Sequential(*layers)
        else:
            effective_obs_dim = obs_dim + (num_agents if num_agents > 0 else 0)
            layers = []
            in_dim = effective_obs_dim
            for units in hidden_dims:
                if noisy_net:
                    layers.append(NoisyLinear(in_dim, units, sigma_init=noisy_sigma))
                else:
                    layers.append(nn.Linear(in_dim, units))
                layers.append(nn.ReLU())
                if not noisy_net:
                    layers.append(nn.Dropout(0.1))
                in_dim = units
            self.feature_network = nn.Sequential(*layers)

        # Output heads
        if dueling:
            if noisy_net:
                self.value_hidden = nn.Sequential(
                    NoisyLinear(in_dim, 128, sigma_init=noisy_sigma),
                    nn.ReLU()
                )
                self.value = NoisyLinear(128, 1, sigma_init=noisy_sigma)

                self.advantage_hidden = nn.Sequential(
                    NoisyLinear(in_dim, 128, sigma_init=noisy_sigma),
                    nn.ReLU()
                )
                self.advantage = NoisyLinear(128, action_dim, sigma_init=noisy_sigma)
            else:
                self.value_hidden = nn.Sequential(
                    nn.Linear(in_dim, 128),
                    nn.ReLU()
                )
                self.value = nn.Linear(128, 1)

                self.advantage_hidden = nn.Sequential(
                    nn.Linear(in_dim, 128),
                    nn.ReLU()
                )
                self.advantage = nn.Linear(128, action_dim)
        else:
            if noisy_net:
                self.q_head = NoisyLinear(in_dim, action_dim, sigma_init=noisy_sigma)
            else:
                self.q_head = nn.Linear(in_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_attention and self.num_lanes > 0:
            batch_size = x.shape[0]
            lane_feat_dim = self.num_lanes * self.num_lane_features

            lane_feats = x[:, :lane_feat_dim].view(batch_size, self.num_lanes, self.num_lane_features)
            global_feats = x[:, lane_feat_dim:]

            attended, _ = self.lane_attention(lane_feats, lane_feats, lane_feats)
            attended = self.lane_norm(attended + lane_feats)
            attended_flat = attended.reshape(batch_size, -1)

            features_input = torch.cat([attended_flat, global_feats], dim=-1)

            if self.num_agents > 0:
                agent_id = x[:, -self.num_agents:]
                features_input = torch.cat([features_input, agent_id], dim=-1)

            features = self.feature_network(features_input)
        elif self.num_agents > 0:
            agent_id = x[:, -self.num_agents:]
            features_input = torch.cat([x[:, :-self.num_agents], agent_id], dim=-1)
            features = self.feature_network(features_input)
        else:
            features = self.feature_network(x)

        if self.dueling:
            v_hidden = self.value_hidden(features)
            values = self.value(v_hidden)

            a_hidden = self.advantage_hidden(features)
            advantages = self.advantage(a_hidden)

            return values + (advantages - advantages.mean(dim=-1, keepdim=True))
        else:
            return self.q_head(features)

    def reset_noise(self) -> None:
        """Reset noise for NoisyLinear layers."""
        if self.noisy_net:
            for module in self.modules():
                if isinstance(module, NoisyLinear):
                    module.reset_noise()


class DQNAgent:
    """Deep Q-Network agent for traffic signal control.

    Args:
        obs_dim: Dimensionality of the observation/state vector.
        action_dim: Number of discrete actions.
        config: Full experiment configuration dictionary.
    """

    def __init__(self, obs_dim: int, action_dim: int, config: dict) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config

        # Determine device
        device_name = config.get("experiment", {}).get("device", "cpu")
        if device_name == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif device_name == "mps" and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # DQN hyperparameters
        dqn_cfg = config["training"].get("dqn", {})
        self.lr = dqn_cfg.get("learning_rate", config["training"].get("learning_rate", 5e-4))
        self.gamma = config["training"]["gamma"]
        self.batch_size = config["training"]["batch_size"]

        self.epsilon = dqn_cfg.get("epsilon_start", 1.0)
        self.epsilon_end = dqn_cfg.get("epsilon_end", 0.01)
        self.epsilon_decay = dqn_cfg.get("epsilon_decay", 0.995)
        self.target_update_freq = dqn_cfg.get("target_update_freq", 10)
        self.dueling = dqn_cfg.get("dueling", False)
        self.double_dqn = dqn_cfg.get("double_dqn", True)
        self.use_per = dqn_cfg.get("per", False)
        self.per_alpha = dqn_cfg.get("per_alpha", 0.6)
        self.per_beta_start = dqn_cfg.get("per_beta_start", 0.4)
        self.per_beta_frames = dqn_cfg.get("per_beta_frames", 100_000)
        self.n_step = dqn_cfg.get("n_step", 1)
        self.noisy_net = dqn_cfg.get("noisy_net", False)
        self.noisy_sigma = dqn_cfg.get("noisy_sigma", 0.5)

        # Soft target update (Polyak averaging)
        self.soft_update_tau = dqn_cfg.get("soft_update_tau", 0.0)  # 0 = hard update
        self.use_soft_update = self.soft_update_tau > 0.0

        # Gradient clipping
        self.grad_clip = dqn_cfg.get("grad_clip", 0.0)  # 0 = no clipping

        # Learning rate scheduler
        self.lr_scheduler_type = dqn_cfg.get("lr_scheduler", "none")  # none, cosine, step
        self.lr_scheduler_eta_min = dqn_cfg.get("lr_scheduler_eta_min", 1e-6)
        self.lr_scheduler_step_size = dqn_cfg.get("lr_scheduler_step_size", 1000)
        self.lr_scheduler_gamma = dqn_cfg.get("lr_scheduler_gamma", 0.95)

        # Attention-based Q-Network
        self.use_attention = dqn_cfg.get("use_attention", False)
        self.num_lanes = dqn_cfg.get("num_lanes", 0)
        self.num_lane_features = dqn_cfg.get("num_lane_features", 5)

        # Multi-agent parameter sharing
        self.num_agents = dqn_cfg.get("num_agents", 0)  # 0 = no sharing

        hidden_dims: List[int] = dqn_cfg.get("hidden_dims", [256, 256, 128])
        replay_capacity: int = dqn_cfg.get("replay_buffer_size", 100_000)

        # Networks
        self.q_network = QNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            dueling=self.dueling,
            noisy_net=self.noisy_net,
            noisy_sigma=self.noisy_sigma,
            use_attention=self.use_attention,
            num_lanes=self.num_lanes,
            num_lane_features=self.num_lane_features,
            num_agents=self.num_agents,
        ).to(self.device)

        self.target_network = QNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            dueling=self.dueling,
            noisy_net=self.noisy_net,
            noisy_sigma=self.noisy_sigma,
            use_attention=self.use_attention,
            num_lanes=self.num_lanes,
            num_lane_features=self.num_lane_features,
            num_agents=self.num_agents,
        ).to(self.device)

        self._hard_update_target()

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.lr)

        # Learning rate scheduler
        if self.lr_scheduler_type == "cosine":
            total_steps = config["training"].get("total_episodes", 100) * config["training"].get("max_steps_per_episode", 1000)
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
                eta_min=self.lr_scheduler_eta_min,
            )
        elif self.lr_scheduler_type == "step":
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.lr_scheduler_step_size,
                gamma=self.lr_scheduler_gamma,
            )
        else:
            self.lr_scheduler = None

        # Replay buffer (uniform or prioritized)
        if self.use_per:
            self.replay_buffer = PrioritizedReplayBuffer(
                capacity=replay_capacity,
                obs_dim=obs_dim,
                alpha=self.per_alpha,
                beta_start=self.per_beta_start,
                beta_frames=self.per_beta_frames,
            )
            logger.info("Using Prioritized Experience Replay (alpha=%.2f, beta_start=%.2f)",
                        self.per_alpha, self.per_beta_start)
        else:
            self.replay_buffer = ReplayBuffer(capacity=replay_capacity, obs_dim=obs_dim)

        # Wrap with n-step collector if n_step > 1
        if self.n_step > 1:
            self.replay_buffer = NStepCollector(
                base_buffer=self.replay_buffer,
                n_step=self.n_step,
                gamma=self.gamma,
            )
            logger.info("Using %d-step returns", self.n_step)

        # Bookkeeping
        self.update_count: int = 0
        self.total_steps: int = 0

        logger.info(
            "DQN Agent initialized | obs_dim=%d | action_dim=%d | dueling=%s | "
            "double=%s | per=%s | hidden=%s | buffer=%d | lr=%.1e | device=%s",
            obs_dim, action_dim, self.dueling, self.double_dqn, self.use_per,
            hidden_dims, replay_capacity, self.lr, self.device,
        )

    def _hard_update_target(self) -> None:
        """Copy weights from Q-network to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
        logger.debug("Target network updated (hard copy)")

    def _soft_update_target(self) -> None:
        """Polyak averaging: target = tau * online + (1 - tau) * target."""
        with torch.no_grad():
            for target_param, online_param in zip(
                self.target_network.parameters(), self.q_network.parameters()
            ):
                target_param.data.mul_(1.0 - self.soft_update_tau)
                target_param.data.add_(self.soft_update_tau * online_param.data)
            # Also update buffers (BatchNorm, LayerNorm, etc.)
            for target_buf, online_buf in zip(
                self.target_network.buffers(), self.q_network.buffers()
            ):
                target_buf.data.copy_(online_buf.data)
        logger.debug("Target network updated (soft, tau=%.4f)", self.soft_update_tau)

    def select_action(
        self, obs: np.ndarray, deterministic: bool = False
    ) -> Tuple[int, float, float]:
        """Select an action using epsilon-greedy policy.

        Args:
            obs: Observation vector.
            deterministic: If True, always pick the greedy action (no exploration).

        Returns:
            Tuple of (action, q_value_of_action, max_q_value).
        """
        self.q_network.eval()
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            if self.noisy_net:
                self.q_network.reset_noise()
            q_values = self.q_network(obs_tensor).cpu().numpy()[0]

        if self.noisy_net:
            # NoisyNet explores via learned noise — always greedy
            action = int(np.argmax(q_values))
        elif not deterministic and np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            action = int(np.argmax(q_values))

        return action, float(q_values[action]), float(np.max(q_values))

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition in the replay buffer.

        Args:
            state: Current observation.
            action: Action taken.
            reward: Reward received.
            next_state: Resulting observation.
            done: Whether the episode ended.
        """
        self.replay_buffer.push(state, action, reward, next_state, done)
        self.total_steps += 1

    def update(self) -> Dict[str, float]:
        """Sample a mini-batch and update the Q-network.

        Returns:
            Dictionary with training metrics (loss, epsilon, avg_q, buffer_size).
            Returns empty dict if buffer is not ready.
        """
        if not self.replay_buffer.is_ready(self.batch_size):
            return {}

        self.q_network.train()
        self.target_network.eval()

        if self.use_per:
            (
                states, actions, rewards, next_states, dones,
                is_weights, tree_indices,
            ) = self.replay_buffer.sample(self.batch_size)

            states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
            actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device)
            rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
            next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
            dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)
            is_weights_t = torch.as_tensor(is_weights, dtype=torch.float32, device=self.device)

            if self.noisy_net:
                self.q_network.reset_noise()
                self.target_network.reset_noise()

            # Compute target Q-values (Double DQN or standard)
            with torch.no_grad():
                if self.double_dqn:
                    next_q_online = self.q_network(next_states_t)
                    best_actions = next_q_online.argmax(dim=1, keepdim=True)
                    next_q_target = self.target_network(next_states_t)
                    max_next_q = next_q_target.gather(1, best_actions).squeeze(1)
                else:
                    next_q = self.target_network(next_states_t)
                    max_next_q = next_q.max(dim=1)[0]
                targets = rewards_t + self.gamma * max_next_q * (1.0 - dones_t)

            q_values = self.q_network(states_t)
            predicted_q = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

            # Compute TD errors for priority updating
            td_errors = (targets - predicted_q).detach().cpu().numpy()

            # Huber Loss with Importance Sampling weights
            element_loss = F.huber_loss(predicted_q, targets, reduction='none')
            loss = (is_weights_t * element_loss).mean()

            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.grad_clip)
            self.optimizer.step()

            # Update priorities with new TD-errors
            self.replay_buffer.update_priorities(tree_indices, np.abs(td_errors))
        else:
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

            states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
            actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device)
            rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
            next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
            dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

            if self.noisy_net:
                self.q_network.reset_noise()
                self.target_network.reset_noise()

            # Compute target Q-values (Double DQN or standard)
            with torch.no_grad():
                if self.double_dqn:
                    next_q_online = self.q_network(next_states_t)
                    best_actions = next_q_online.argmax(dim=1, keepdim=True)
                    next_q_target = self.target_network(next_states_t)
                    max_next_q = next_q_target.gather(1, best_actions).squeeze(1)
                else:
                    next_q = self.target_network(next_states_t)
                    max_next_q = next_q.max(dim=1)[0]
                targets = rewards_t + self.gamma * max_next_q * (1.0 - dones_t)

            q_values = self.q_network(states_t)
            predicted_q = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)
            loss = F.huber_loss(predicted_q, targets, reduction='mean')

            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.grad_clip)
            self.optimizer.step()

        self.update_count += 1

        # Periodic target update
        if self.use_soft_update:
            self._soft_update_target()
        elif self.update_count % self.target_update_freq == 0:
            self._hard_update_target()

        # Learning rate scheduler step
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        # Decay epsilon (skip if using NoisyNet — exploration is learned)
        if not self.noisy_net:
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # Compute avg Q for logging
        self.q_network.eval()
        with torch.no_grad():
            sample_states = torch.as_tensor(states[:min(64, len(states))], dtype=torch.float32, device=self.device)
            avg_q = float(self.q_network(sample_states).max(dim=1)[0].mean().item())

        result = {
            "loss": float(loss.item()),
            "epsilon": self.epsilon,
            "avg_q": avg_q,
            "buffer_size": len(self.replay_buffer),
        }
        if self.lr_scheduler is not None:
            result["lr"] = self.optimizer.param_groups[0]["lr"]
        return result

    def save(self, path: str) -> None:
        """Save the Q-network weights and state to a file.

        Args:
            path: Destination file path.
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        save_dict = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'update_count': self.update_count,
            'total_steps': self.total_steps,
        }
        if self.lr_scheduler is not None:
            save_dict['lr_scheduler_state_dict'] = self.lr_scheduler.state_dict()
        torch.save(save_dict, path)
        logger.info("DQN model saved to %s", path)

    def load(self, path: str) -> None:
        """Load Q-network weights and state from a file.

        Args:
            path: Source file path.
        """
        checkpoint = torch.load(path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'q_network_state_dict' in checkpoint:
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            if 'target_network_state_dict' in checkpoint:
                self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            else:
                self._hard_update_target()
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'epsilon' in checkpoint:
                self.epsilon = checkpoint['epsilon']
            if 'update_count' in checkpoint:
                self.update_count = checkpoint['update_count']
            if 'total_steps' in checkpoint:
                self.total_steps = checkpoint['total_steps']
            if 'lr_scheduler_state_dict' in checkpoint and self.lr_scheduler is not None:
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        else:
            self.q_network.load_state_dict(checkpoint)
            self._hard_update_target()
        logger.info("DQN model loaded from %s", path)

    def get_config_summary(self) -> Dict[str, object]:
        """Return a summary of agent configuration for logging.

        Returns:
            Dictionary of key hyperparameters.
        """
        return {
            "algorithm": "DQN" + (" (Dueling)" if self.dueling else ""),
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "learning_rate": self.lr,
            "gamma": self.gamma,
            "epsilon_start": self.config["training"].get("dqn", {}).get("epsilon_start", 1.0),
            "epsilon_end": self.epsilon_end,
            "epsilon_decay": self.epsilon_decay,
            "target_update_freq": self.target_update_freq,
            "batch_size": self.batch_size,
            "replay_buffer_capacity": self.replay_buffer.capacity,
            "dueling": self.dueling,
            "double_dqn": self.double_dqn,
            "per": self.use_per,
            "n_step": self.n_step,
            "noisy_net": self.noisy_net,
            "soft_update_tau": self.soft_update_tau,
            "grad_clip": self.grad_clip,
            "lr_scheduler": self.lr_scheduler_type,
            "use_attention": self.use_attention,
            "num_agents": self.num_agents,
        }
