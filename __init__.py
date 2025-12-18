"""
Industrial Reinforcement Learning Framework for Robotics & Control

Core Components:
- Markov Decision Processes
- Policy Optimization
- Model-based RL
- Robotics Environments
- Control Systems Integration

Engineering Design Principles:
- Modular Architecture
- Production Ready
- Scalable Deployment
- Real-time Performance
"""

__version__ = "1.0.0"
__author__ = "Industrial AI Systems"

from .core.mdp import MDP, State, Action, Reward, Transition
from .core.policy import Policy, ValueFunction
from .algorithms.policy_gradient import PPO, TRPO
from .algorithms.value_based import DQN, DDQN
from .algorithms.model_based import MBPO, PETS
from .environments.robots import RobotEnvironment
from .control.controllers import RLController, AdaptiveController

__all__ = [
    "MDP",
    "State",
    "Action",
    "Reward",
    "Transition",
    "Policy",
    "ValueFunction",
    "PPO",
    "TRPO",
    "DQN",
    "DDPG",
    "MBPO",
    "PETS",
    "RobotEnvironment",
    "RLController",
    "AdaptiveController",
]
