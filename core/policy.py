"""
Policy and Value Function Framework

Implements policy representations and value function approximators
for reinforcement learning in industrial control systems.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal, Independent


class Policy(ABC):
    """Abstract base class for policies"""

    def __init__(self, state_dim: int, action_dim: int, config: Dict[str, Any]):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.device = torch.device(config.get("device", "cpu"))

    @abstractmethod
    def forward(self, states: torch.Tensor) -> Any:
        """Forward pass through policy network"""
        pass

    @abstractmethod
    def sample(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample actions and return log probabilities"""
        pass

    @abstractmethod
    def get_action(self, state: np.ndarray) -> np.ndarray:
        """Get action for single state (deterministic)"""
        pass


class ValueFunction(ABC):
    """Abstract base class for value functions"""

    def __init__(self, state_dim: int, config: Dict[str, Any]):
        self.state_dim = state_dim
        self.config = config
        self.device = torch.device(config.get("device", "cpu"))

    @abstractmethod
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Forward pass through value network"""
        pass

    @abstractmethod
    def get_value(self, state: np.ndarray) -> float:
        """Get value for single state"""
        pass


class MLP(nn.Module):
    """Multi-layer perceptron for neural network architectures"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        activation: str = "relu",
        dropout: float = 0.0,
        layer_norm: bool = False,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))

            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "gelu":
                layers.append(nn.GELU())

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class GaussianPolicy(Policy, nn.Module):
    """Gaussian policy for continuous control"""

    def __init__(self, state_dim: int, action_dim: int, config: Dict[str, Any]):
        Policy.__init__(self, state_dim, action_dim, config)
        nn.Module.__init__(self)

        hidden_dims = config.get("hidden_dims", [256, 256])
        self.action_scale = torch.tensor(config.get("action_scale", 1.0)).to(
            self.device
        )
        self.action_bias = torch.tensor(config.get("action_bias", 0.0)).to(self.device)

        self.mean_net = MLP(state_dim, action_dim, hidden_dims)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        self.to(self.device)

    def forward(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.mean_net(states)
        std = torch.exp(self.log_std.clamp(-20, 2))

        return mean, std

    def sample(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, std = self.forward(states)
        dist = Independent(Normal(mean, std), 1)

        actions = dist.rsample()
        log_probs = dist.log_prob(actions)

        actions = torch.tanh(actions) * self.action_scale + self.action_bias

        return actions, log_probs

    def get_action(self, state: np.ndarray) -> np.ndarray:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mean, _ = self.forward(state_tensor)
            action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action.cpu().numpy()[0]


class CategoricalPolicy(Policy, nn.Module):
    """Categorical policy for discrete control"""

    def __init__(self, state_dim: int, num_actions: int, config: Dict[str, Any]):
        Policy.__init__(self, state_dim, num_actions, config)
        nn.Module.__init__(self)

        hidden_dims = config.get("hidden_dims", [256, 256])
        self.policy_net = MLP(state_dim, num_actions, hidden_dims)

        self.to(self.device)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        logits = self.policy_net(states)
        return logits

    def sample(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(states)
        dist = Categorical(logits=logits)

        actions = dist.sample()
        log_probs = dist.log_prob(actions)

        return actions, log_probs

    def get_action(self, state: np.ndarray) -> np.ndarray:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.forward(state_tensor)
            action = torch.argmax(logits, dim=-1)
        return action.cpu().numpy()[0]


class ValueNetwork(ValueFunction, nn.Module):
    """Value function approximator"""

    def __init__(self, state_dim: int, config: Dict[str, Any]):
        ValueFunction.__init__(self, state_dim, config)
        nn.Module.__init__(self)

        hidden_dims = config.get("hidden_dims", [256, 256])
        self.value_net = MLP(state_dim, 1, hidden_dims)

        self.to(self.device)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.value_net(states).squeeze(-1)

    def get_value(self, state: np.ndarray) -> float:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            value = self.forward(state_tensor)
        return value.cpu().numpy()[0]


class QNetwork(ValueFunction, nn.Module):
    """Q-value function approximator"""

    def __init__(self, state_dim: int, action_dim: int, config: Dict[str, Any]):
        ValueFunction.__init__(self, state_dim, config)
        nn.Module.__init__(self)

        self.action_dim = action_dim
        hidden_dims = config.get("hidden_dims", [256, 256])
        self.q_net = MLP(state_dim + action_dim, 1, hidden_dims)

        self.to(self.device)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([states, actions], dim=-1)
        return self.q_net(x).squeeze(-1)

    def get_value(self, state: np.ndarray, action: np.ndarray) -> float:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_tensor = torch.FloatTensor(action).unsqueeze(0).to(self.device)
        with torch.no_grad():
            value = self.forward(state_tensor, action_tensor)
        return value.cpu().numpy()[0]


class ActorCritic(nn.Module):
    """Actor-Critic architecture combining policy and value function"""

    def __init__(self, state_dim: int, action_dim: int, config: Dict[str, Any]):
        super().__init__()

        self.action_type = config.get("action_type", "continuous")
        hidden_dims = config.get("hidden_dims", [256, 256])

        if self.action_type == "continuous":
            self.actor = GaussianPolicy(state_dim, action_dim, config)
        else:
            self.actor = CategoricalPolicy(state_dim, action_dim, config)

        self.critic = ValueNetwork(state_dim, config)

    def forward(self, states: torch.Tensor) -> Dict[str, torch.Tensor]:
        actions, log_probs = self.actor.sample(states)
        values = self.critic(states)

        return {"actions": actions, "log_probs": log_probs, "values": values}

    def evaluate_actions(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Evaluate actions for PPO-style updates"""
        if self.action_type == "continuous":
            mean, std = self.actor.forward(states)
            dist = Independent(Normal(mean, std), 1)
        else:
            logits = self.actor.forward(states)
            dist = Categorical(logits=logits)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        values = self.critic(states)

        return {"log_probs": log_probs, "values": values, "entropy": entropy}
