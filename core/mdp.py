"""
Core Markov Decision Process Framework

Implements the mathematical foundation for reinforcement learning
in industrial robotics and control systems.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np


class StateType(Enum):
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    HYBRID = "hybrid"


@dataclass
class State:
    """State representation in the MDP"""

    values: np.ndarray
    type: StateType
    timestamp: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.values.shape

    @property
    def dimension(self) -> int:
        return self.values.size


@dataclass
class Action:
    """Action representation in the MDP"""

    values: np.ndarray
    type: StateType
    constraints: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.values.shape

    def clip(self, lower: np.ndarray, upper: np.ndarray) -> "Action":
        """Clip action to bounds"""
        return Action(
            values=np.clip(self.values, lower, upper),
            type=self.type,
            constraints=self.constraints,
            metadata=self.metadata,
        )


@dataclass
class Reward:
    """Reward signal with additional context"""

    value: float
    components: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None

    def __add__(self, other: Union[float, "Reward"]) -> "Reward":
        if isinstance(other, Reward):
            new_value = self.value + other.value
            new_components = {}
            if self.components:
                new_components.update(self.components)
            if other.components:
                for k, v in other.components.items():
                    new_components[k] = new_components.get(k, 0) + v
            return Reward(value=new_value, components=new_components or None)
        else:
            return Reward(value=self.value + other, components=self.components)


@dataclass
class Transition:
    """Complete transition tuple in MDP"""

    state: State
    action: Action
    reward: Reward
    next_state: State
    done: bool
    info: Optional[Dict[str, Any]] = None


class MDP(ABC):
    """
    Abstract base class for Markov Decision Process

    Implements the mathematical framework for sequential decision making
    in industrial control systems.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._state_space = None
        self._action_space = None
        self._current_state = None
        self._step_count = 0
        self._episode_count = 0

    @property
    @abstractmethod
    def state_space(self) -> Dict[str, Any]:
        """State space specification"""
        pass

    @property
    @abstractmethod
    def action_space(self) -> Dict[str, Any]:
        """Action space specification"""
        pass

    @abstractmethod
    def reset(self) -> State:
        """Reset environment to initial state"""
        pass

    @abstractmethod
    def step(self, action: Action) -> Transition:
        """Execute one step in the environment"""
        pass

    @abstractmethod
    def is_terminal(self, state: State) -> bool:
        """Check if state is terminal"""
        pass

    @abstractmethod
    def transition_probability(
        self, state: State, action: Action, next_state: State
    ) -> float:
        """Transition probability P(s'|s,a)"""
        pass

    @abstractmethod
    def reward_function(
        self, state: State, action: Action, next_state: State
    ) -> Reward:
        """Reward function R(s,a,s')"""
        pass

    def get_state_representation(self, raw_state: Any) -> State:
        """Convert raw state to State object"""
        if isinstance(raw_state, np.ndarray):
            return State(values=raw_state, type=StateType.CONTINUOUS)
        elif isinstance(raw_state, (int, float)):
            return State(values=np.array([raw_state]), type=StateType.DISCRETE)
        else:
            raise ValueError(f"Unsupported state type: {type(raw_state)}")

    def get_action_representation(self, raw_action: Any) -> Action:
        """Convert raw action to Action object"""
        if isinstance(raw_action, np.ndarray):
            return Action(values=raw_action, type=StateType.CONTINUOUS)
        elif isinstance(raw_action, (int, float)):
            return Action(values=np.array([raw_action]), type=StateType.DISCRETE)
        else:
            raise ValueError(f"Unsupported action type: {type(raw_action)}")

    def validate_action(self, action: Action) -> bool:
        """Validate action against action space constraints"""
        if self.action_space.get("type") == "continuous":
            lower = self.action_space.get("low", -np.inf)
            upper = self.action_space.get("high", np.inf)
            return np.all(action.values >= lower) and np.all(action.values <= upper)
        return True

    @property
    def current_state(self) -> Optional[State]:
        """Current environment state"""
        return self._current_state

    @property
    def step_count(self) -> int:
        """Current step count in episode"""
        return self._step_count

    @property
    def episode_count(self) -> int:
        """Total episode count"""
        return self._episode_count


class DiscreteMDP(MDP):
    """Discrete state and action MDP"""

    def __init__(self, num_states: int, num_actions: int, config: Dict[str, Any]):
        super().__init__(config)
        self.num_states = num_states
        self.num_actions = num_actions
        self._transition_matrix = np.zeros((num_states, num_actions, num_states))
        self._reward_matrix = np.zeros((num_states, num_actions, num_states))

    @property
    def state_space(self) -> Dict[str, Any]:
        return {
            "type": "discrete",
            "size": self.num_states,
            "values": list(range(self.num_states)),
        }

    @property
    def action_space(self) -> Dict[str, Any]:
        return {
            "type": "discrete",
            "size": self.num_actions,
            "values": list(range(self.num_actions)),
        }

    def set_transition_probability(
        self, state: int, action: int, next_state: int, prob: float
    ):
        """Set transition probability"""
        self._transition_matrix[state, action, next_state] = prob

    def set_reward(self, state: int, action: int, next_state: int, reward: float):
        """Set reward value"""
        self._reward_matrix[state, action, next_state] = reward

    def transition_probability(
        self, state: State, action: Action, next_state: State
    ) -> float:
        s_idx = int(state.values[0])
        a_idx = int(action.values[0])
        ns_idx = int(next_state.values[0])
        return self._transition_matrix[s_idx, a_idx, ns_idx]

    def reward_function(
        self, state: State, action: Action, next_state: State
    ) -> Reward:
        s_idx = int(state.values[0])
        a_idx = int(action.values[0])
        ns_idx = int(next_state.values[0])
        return Reward(value=self._reward_matrix[s_idx, a_idx, ns_idx])


class ContinuousMDP(MDP):
    """Continuous state and action MDP"""

    def __init__(self, state_dim: int, action_dim: int, config: Dict[str, Any]):
        super().__init__(config)
        self.state_dim = state_dim
        self.action_dim = action_dim

    @property
    def state_space(self) -> Dict[str, Any]:
        return {
            "type": "continuous",
            "dimension": self.state_dim,
            "low": self.config.get("state_low", -np.inf),
            "high": self.config.get("state_high", np.inf),
        }

    @property
    def action_space(self) -> Dict[str, Any]:
        return {
            "type": "continuous",
            "dimension": self.action_dim,
            "low": self.config.get("action_low", -1.0),
            "high": self.config.get("action_high", 1.0),
        }
