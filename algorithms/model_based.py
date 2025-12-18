"""
Model-Based Reinforcement Learning

Implements world models and model-based RL algorithms for
efficient learning in industrial robotics applications.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Independent, Normal
from collections import deque
import math

from ..core.mdp import Transition, State, Action, Reward
from ..core.policy import Policy


class WorldModel(ABC):
    """Abstract base class for world models"""

    def __init__(self, state_dim: int, action_dim: int, config: Dict[str, Any]):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.device = torch.device(config.get("device", "cpu"))

    @abstractmethod
    def predict(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict next states and rewards"""
        pass

    @abstractmethod
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Train world model on batch of transitions"""
        pass


class EnsembleDynamicsModel(WorldModel, nn.Module):
    """Ensemble of neural networks for dynamics modeling"""

    def __init__(self, state_dim: int, action_dim: int, config: Dict[str, Any]):
        WorldModel.__init__(self, state_dim, action_dim, config)
        nn.Module.__init__(self)

        self.num_ensemble = config.get("num_ensemble", 7)
        self.hidden_dims = config.get("hidden_dims", [200, 200, 200])
        self.learning_rate = config.get("learning_rate", 1e-3)

        self.models = nn.ModuleList(
            [self._create_network() for _ in range(self.num_ensemble)]
        )

        self.optimizers = [
            torch.optim.Adam(model.parameters(), lr=self.learning_rate)
            for model in self.models
        ]

        self.max_logvar = nn.Parameter(torch.ones(1) * config.get("max_logvar", 0.5))
        self.min_logvar = nn.Parameter(torch.ones(1) * config.get("min_logvar", -10.0))

        self.to(self.device)

    def _create_network(self) -> nn.Module:
        """Create individual network in ensemble"""
        input_dim = self.state_dim + self.action_dim
        output_dim = self.state_dim * 2  # mean and logvar

        layers = []
        prev_dim = input_dim

        for hidden_dim in self.hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1)])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)

    def forward(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through ensemble"""
        inputs = torch.cat([states, actions], dim=-1)

        outputs = []
        for model in self.models:
            output = model(inputs)
            mean, logvar = torch.chunk(output, 2, dim=-1)

            logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
            logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

            outputs.append((mean, logvar))

        return outputs

    def predict(
        self, states: torch.Tensor, actions: torch.Tensor, num_samples: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict next states with uncertainty"""
        with torch.no_grad():
            ensemble_outputs = self.forward(states, actions)

            model_idx = torch.randint(0, self.num_ensemble, (num_samples,))

            next_states = []
            rewards = []

            for idx in model_idx:
                mean, logvar = ensemble_outputs[idx]
                std = torch.exp(0.5 * logvar)

                eps = torch.randn_like(mean)
                sample = mean + std * eps

                next_states.append(sample)

            next_states_tensor = torch.stack(next_states, dim=0)

            if num_samples == 1:
                next_states_tensor = next_states_tensor.squeeze(0)

            return next_states_tensor, torch.zeros(num_samples, states.shape[0], 1).to(
                self.device
            )

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Train ensemble on batch"""
        states = batch["states"]
        actions = batch["actions"]
        next_states = batch["next_states"]

        losses = []

        for model, optimizer in zip(self.models, self.optimizers):
            optimizer.zero_grad()

            input_tensor = torch.cat([states, actions], dim=-1)
            output = model(input_tensor)
            pred_mean, pred_logvar = torch.chunk(output, 2, dim=-1)

            pred_logvar = self.max_logvar - F.softplus(self.max_logvar - pred_logvar)
            pred_logvar = self.min_logvar + F.softplus(pred_logvar - self.min_logvar)

            inv_var = torch.exp(-pred_logvar)

            loss = torch.mean((pred_mean - next_states) ** 2 * inv_var + pred_logvar)

            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        return {"ensemble_loss": np.mean(losses)}


class RewardModel(nn.Module):
    """Neural network for reward prediction"""

    def __init__(self, state_dim: int, action_dim: int, config: Dict[str, Any]):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        hidden_dims = config.get("hidden_dims", [200, 200, 200])
        learning_rate = config.get("learning_rate", 1e-3)

        layers = []
        input_dim = state_dim + action_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1)]
            )
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.device = torch.device(config.get("device", "cpu"))
        self.to(self.device)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat([states, actions], dim=-1)
        return self.network(inputs)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        states = batch["states"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)

        self.optimizer.zero_grad()
        pred_rewards = self.forward(states, actions)
        loss = F.mse_loss(pred_rewards, rewards)

        loss.backward()
        self.optimizer.step()

        return {"reward_loss": loss.item()}


class MBPO:
    """Model-Based Policy Optimization"""

    def __init__(
        self,
        world_model: WorldModel,
        reward_model: RewardModel,
        policy: Policy,
        config: Dict[str, Any],
    ):
        self.world_model = world_model
        self.reward_model = reward_model
        self.policy = policy

        self.config = config
        self.device = torch.device(config.get("device", "cpu"))

        self.batch_size = config.get("batch_size", 256)
        self.model_rollouts = config.get("model_rollouts", 5)
        self.rollout_horizon = config.get("rollout_horizon", 5)
        self.real_ratio = config.get("real_ratio", 0.5)

        self.replay_buffer = deque(maxlen=config.get("buffer_size", 100000))
        self.model_buffer = deque(maxlen=config.get("model_buffer_size", 100000))

        self.update_count = 0

    def collect_real_data(self, environment, num_episodes: int) -> List[Transition]:
        """Collect real environment data"""
        trajectories = []

        for episode in range(num_episodes):
            state = environment.reset()
            episode_transitions = []

            while True:
                action = self.policy.get_action(state)
                transition = environment.step(action)

                episode_transitions.append(transition)
                self.replay_buffer.append(transition)

                if transition.done:
                    break

                state = transition.next_state

            trajectories.append(episode_transitions)

        return trajectories

    def generate_model_rollouts(self, num_rollouts: int):
        """Generate synthetic data using world model"""
        if len(self.replay_buffer) < self.batch_size:
            return

        real_batch = list(
            np.random.choice(
                list(self.replay_buffer),
                size=min(num_rollouts, len(self.replay_buffer)),
                replace=False,
            )
        )

        for start_transition in real_batch:
            state = start_transition.state
            rollout_transitions = []

            for _ in range(self.rollout_horizon):
                action = self.policy.get_action(state)

                state_tensor = (
                    torch.FloatTensor(state.values).unsqueeze(0).to(self.device)
                )
                action_tensor = (
                    torch.FloatTensor(action.values).unsqueeze(0).to(self.device)
                )

                with torch.no_grad():
                    next_state_samples, reward_samples = self.world_model.predict(
                        state_tensor, action_tensor, num_samples=1
                    )

                next_state_values = next_state_samples.squeeze(0).cpu().numpy()
                reward_value = reward_samples.squeeze(0).cpu().numpy()[0, 0]

                next_state = State(values=next_state_values, type=state.type)
                reward = Reward(value=reward_value)

                done = self._check_terminal(next_state)

                transition = Transition(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                )

                rollout_transitions.append(transition)
                self.model_buffer.append(transition)

                if done:
                    break

                state = next_state

    def train_world_model(self, num_epochs: int = 10):
        """Train world model on real data"""
        if len(self.replay_buffer) < self.batch_size:
            return {}

        losses = []

        for _ in range(num_epochs):
            batch = list(
                np.random.choice(
                    list(self.replay_buffer), size=self.batch_size, replace=False
                )
            )

            states = torch.FloatTensor([t.state.values for t in batch]).to(self.device)
            actions = torch.FloatTensor([t.action.values for t in batch]).to(
                self.device
            )
            next_states = torch.FloatTensor([t.next_state.values for t in batch]).to(
                self.device
            )

            model_batch = {
                "states": states,
                "actions": actions,
                "next_states": next_states,
            }

            loss = self.world_model.train_step(model_batch)
            losses.append(loss)

        return {"world_model_loss": np.mean([l["ensemble_loss"] for l in losses])}

    def train_reward_model(self, num_epochs: int = 10):
        """Train reward model on real data"""
        if len(self.replay_buffer) < self.batch_size:
            return {}

        losses = []

        for _ in range(num_epochs):
            batch = list(
                np.random.choice(
                    list(self.replay_buffer), size=self.batch_size, replace=False
                )
            )

            states = torch.FloatTensor([t.state.values for t in batch]).to(self.device)
            actions = torch.FloatTensor([t.action.values for t in batch]).to(
                self.device
            )
            rewards = torch.FloatTensor([[t.reward.value] for t in batch]).to(
                self.device
            )

            model_batch = {"states": states, "actions": actions, "rewards": rewards}

            loss = self.reward_model.train_step(model_batch)
            losses.append(loss)

        return {"reward_model_loss": np.mean([l["reward_loss"] for l in losses])}

    def update_policy(self) -> Dict[str, float]:
        """Update policy using mixed real and model data"""
        real_batch_size = int(self.batch_size * self.real_ratio)
        model_batch_size = self.batch_size - real_batch_size

        real_data = []
        model_data = []

        if real_batch_size > 0 and len(self.replay_buffer) >= real_batch_size:
            real_data = list(
                np.random.choice(
                    list(self.replay_buffer), size=real_batch_size, replace=False
                )
            )

        if model_batch_size > 0 and len(self.model_buffer) >= model_batch_size:
            model_data = list(
                np.random.choice(
                    list(self.model_buffer), size=model_batch_size, replace=False
                )
            )

        mixed_batch = real_data + model_data

        if len(mixed_batch) < self.batch_size:
            return {}

        # Here you would typically call the policy's update method
        # This depends on the specific policy optimizer being used

        return {"policy_updated": True}

    def _check_terminal(self, state: State) -> bool:
        """Check if state is terminal"""
        # This should be customized for specific environments
        return False


class PETS:
    """Probabilistic Ensembles with Trajectory Sampling"""

    def __init__(
        self, world_model: WorldModel, reward_model: RewardModel, config: Dict[str, Any]
    ):
        self.world_model = world_model
        self.reward_model = reward_model
        self.config = config
        self.device = torch.device(config.get("device", "cpu"))

        self.horizon = config.get("horizon", 15)
        self.num_particles = config.get("num_particles", 20)
        self.num_samples = config.get("num_samples", 100)

        self.action_dim = world_model.action_dim

    def plan(self, initial_state: State) -> Action:
        """Plan optimal action sequence using MPC"""
        state_tensor = (
            torch.FloatTensor(initial_state.values).unsqueeze(0).to(self.device)
        )

        best_action_sequence = None
        best_return = -float("inf")

        for _ in range(self.num_samples):
            action_sequence = torch.randn(self.horizon, self.action_dim).to(self.device)
            action_sequence = torch.tanh(action_sequence)

            total_return = 0
            current_state = state_tensor.repeat(self.num_particles, 1, 1)

            for t in range(self.horizon):
                action = (
                    action_sequence[t].unsqueeze(0).repeat(self.num_particles, 1, 1)
                )

                with torch.no_grad():
                    next_states, rewards = self.world_model.predict(
                        current_state.squeeze(1),
                        action.squeeze(1),
                        num_samples=self.num_particles,
                    )
                    pred_rewards = self.reward_model(
                        current_state.squeeze(1), action.squeeze(1)
                    )

                total_return += pred_rewards.mean().item()
                current_state = next_states.unsqueeze(1)

            if total_return > best_return:
                best_return = total_return
                best_action_sequence = action_sequence

        return Action(
            values=best_action_sequence[0].cpu().numpy(), type=initial_state.type
        )
