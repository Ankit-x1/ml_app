"""
Control Systems Integration

Bridges reinforcement learning with traditional control systems
for industrial robotics and automation applications.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import numpy as np
import scipy.linalg as la
import scipy.signal as signal
from dataclasses import dataclass
from enum import Enum
import time

from ..core.mdp import State, Action, Reward
from ..core.policy import Policy
from ..environments.robots import RobotEnvironment


class ControllerType(Enum):
    PID = "pid"
    LQR = "lqr"
    MPC = "mpc"
    ADAPTIVE = "adaptive"
    ROBUST = "robust"
    HYBRID = "hybrid"


@dataclass
class ControllerConfig:
    """Controller configuration parameters"""

    controller_type: ControllerType
    sampling_time: float
    state_dimension: int
    action_dimension: int
    gains: Optional[Dict[str, np.ndarray]] = None
    constraints: Optional[Dict[str, Any]] = None
    adaptive_params: Optional[Dict[str, Any]] = None
    robust_params: Optional[Dict[str, Any]] = None


class Controller(ABC):
    """Abstract base class for controllers"""

    def __init__(self, config: ControllerConfig):
        self.config = config
        self.state_dim = config.state_dimension
        self.action_dim = config.action_dimension
        self.dt = config.sampling_time
        self.is_initialized = False

    @abstractmethod
    def initialize(self, initial_state: State) -> None:
        """Initialize controller with initial conditions"""
        pass

    @abstractmethod
    def compute_control(
        self, state: State, reference: Optional[State] = None
    ) -> Action:
        """Compute control action"""
        pass

    @abstractmethod
    def update_parameters(self, state: State, action: Action, **kwargs) -> None:
        """Update controller parameters"""
        pass


class PIDController(Controller):
    """PID controller for single and multi-variable systems"""

    def __init__(self, config: ControllerConfig):
        super().__init__(config)

        # PID gains [Kp, Ki, Kd] for each DOF
        self.kp = config.gains.get("kp", np.ones(self.action_dim))
        self.ki = config.gains.get("ki", np.zeros(self.action_dim))
        self.kd = config.gains.get("kd", np.zeros(self.action_dim))

        # State variables
        self.integral_error = np.zeros(self.action_dim)
        self.prev_error = np.zeros(self.action_dim)
        self.prev_time = 0

        # Constraints
        self.action_limits = config.constraints.get("action_limits", None)
        self.integral_limits = config.constraints.get("integral_limits", None)

    def initialize(self, initial_state: State) -> None:
        """Initialize PID controller"""
        self.integral_error = np.zeros(self.action_dim)
        self.prev_error = np.zeros(self.action_dim)
        self.prev_time = time.time()
        self.is_initialized = True

    def compute_control(
        self, state: State, reference: Optional[State] = None
    ) -> Action:
        """Compute PID control action"""
        if not self.is_initialized:
            self.initialize(state)

        if reference is None:
            reference = State(values=np.zeros(self.state_dim), type=state.type)

        current_time = time.time()
        dt = current_time - self.prev_time

        error = reference.values[: self.action_dim] - state.values[: self.action_dim]

        # Proportional term
        p_term = self.kp * error

        # Integral term
        self.integral_error += error * dt
        if self.integral_limits:
            self.integral_error = np.clip(
                self.integral_error,
                self.integral_limits[:, 0],
                self.integral_limits[:, 1],
            )
        i_term = self.ki * self.integral_error

        # Derivative term
        if dt > 0:
            derivative = (error - self.prev_error) / dt
            d_term = self.kd * derivative
        else:
            d_term = np.zeros(self.action_dim)

        # Total control action
        control_action = p_term + i_term + d_term

        # Apply action limits
        if self.action_limits:
            control_action = np.clip(
                control_action, self.action_limits[:, 0], self.action_limits[:, 1]
            )

        # Update states
        self.prev_error = error
        self.prev_time = current_time

        return Action(values=control_action, type=state.type)

    def update_parameters(self, state: State, action: Action, **kwargs) -> None:
        """Update PID gains (adaptive functionality)"""
        if "new_gains" in kwargs:
            new_gains = kwargs["new_gains"]
            if "kp" in new_gains:
                self.kp = new_gains["kp"]
            if "ki" in new_gains:
                self.ki = new_gains["ki"]
            if "kd" in new_gains:
                self.kd = new_gains["kd"]


class LQRController(Controller):
    """Linear Quadratic Regulator controller"""

    def __init__(self, config: ControllerConfig):
        super().__init__(config)

        # Weight matrices
        self.Q = config.gains.get("Q", np.eye(self.state_dim))
        self.R = config.gains.get("R", np.eye(self.action_dim))

        # System matrices (should be set via system identification)
        self.A = None
        self.B = None

        # LQR gain matrix
        self.K = None

        # Constraints
        self.action_limits = config.constraints.get("action_limits", None)

    def set_system_matrices(self, A: np.ndarray, B: np.ndarray) -> None:
        """Set linear system matrices"""
        self.A = A
        self.B = B
        self._solve_lqr()

    def _solve_lqr(self) -> None:
        """Solve Algebraic Riccati Equation for LQR gains"""
        if self.A is None or self.B is None:
            raise ValueError("System matrices A and B must be set first")

        try:
            # Solve ARE using scipy
            X = la.solve_continuous_are(self.A, self.B, self.Q, self.R)
            self.K = la.solve(self.R, self.B.T @ X)
        except Exception as e:
            print(f"LQR solution failed: {e}")
            self.K = np.zeros((self.action_dim, self.state_dim))

    def initialize(self, initial_state: State) -> None:
        """Initialize LQR controller"""
        if self.K is None:
            self._solve_lqr()
        self.is_initialized = True

    def compute_control(
        self, state: State, reference: Optional[State] = None
    ) -> Action:
        """Compute LQR control action"""
        if not self.is_initialized:
            self.initialize(state)

        if reference is None:
            reference = State(values=np.zeros(self.state_dim), type=state.type)

        error = reference.values - state.values
        control_action = -self.K @ error

        # Apply action limits
        if self.action_limits:
            control_action = np.clip(
                control_action, self.action_limits[:, 0], self.action_limits[:, 1]
            )

        return Action(values=control_action, type=state.type)

    def update_parameters(self, state: State, action: Action, **kwargs) -> None:
        """Update LQR parameters for adaptive control"""
        if "new_Q" in kwargs:
            self.Q = kwargs["new_Q"]
            self._solve_lqr()
        if "new_R" in kwargs:
            self.R = kwargs["new_R"]
            self._solve_lqr()


class MPCController(Controller):
    """Model Predictive Controller"""

    def __init__(self, config: ControllerConfig):
        super().__init__(config)

        # MPC parameters
        self.horizon = config.adaptive_params.get("horizon", 10)
        self.Q = config.gains.get("Q", np.eye(self.state_dim))
        self.R = config.gains.get("R", np.eye(self.action_dim))

        # System model
        self.A = None
        self.B = None

        # Constraints
        self.action_limits = config.constraints.get("action_limits", None)
        self.state_limits = config.constraints.get("state_limits", None)

    def set_system_model(
        self, dynamics_model: Callable[[np.ndarray, np.ndarray], np.ndarray]
    ) -> None:
        """Set system dynamics model"""
        self.dynamics_model = dynamics_model

    def initialize(self, initial_state: State) -> None:
        """Initialize MPC controller"""
        self.is_initialized = True

    def compute_control(
        self, state: State, reference: Optional[State] = None
    ) -> Action:
        """Compute MPC control action"""
        if not self.is_initialized:
            self.initialize(state)

        if reference is None:
            reference = State(values=np.zeros(self.state_dim), type=state.type)

        # Solve optimization problem (simplified implementation)
        optimal_actions = self._solve_mpc_problem(state, reference)

        # Apply first action in sequence
        control_action = optimal_actions[0]

        # Apply action limits
        if self.action_limits:
            control_action = np.clip(
                control_action, self.action_limits[:, 0], self.action_limits[:, 1]
            )

        return Action(values=control_action, type=state.type)

    def _solve_mpc_problem(self, current_state: State, reference: State) -> np.ndarray:
        """Solve MPC optimization problem (simplified)"""
        # Simplified solution - in practice would use optimization solver
        actions = np.zeros((self.horizon, self.action_dim))
        predicted_state = current_state.values.copy()

        for k in range(self.horizon):
            # Simple LQR-style control
            error = reference.values - predicted_state
            actions[k] = (
                -la.solve(self.R, self.B.T @ self.Q @ error)
                if self.B is not None
                else np.zeros(self.action_dim)
            )

            # Predict next state
            if self.dynamics_model:
                predicted_state = self.dynamics_model(predicted_state, actions[k])

        return actions

    def update_parameters(self, state: State, action: Action, **kwargs) -> None:
        """Update MPC parameters"""
        pass


class AdaptiveController(Controller):
    """Adaptive controller combining RL with traditional control"""

    def __init__(
        self,
        config: ControllerConfig,
        base_controller: Controller,
        rl_policy: Optional[Policy] = None,
    ):
        super().__init__(config)

        self.base_controller = base_controller
        self.rl_policy = rl_policy
        self.adaptation_rate = config.adaptive_params.get("adaptation_rate", 0.01)
        self.blending_factor = 0.0  # Controls mix of base vs RL

        # Adaptive parameters
        self.parameter_history = []
        self.performance_history = []

    def initialize(self, initial_state: State) -> None:
        """Initialize adaptive controller"""
        self.base_controller.initialize(initial_state)
        self.blending_factor = 0.0
        self.is_initialized = True

    def compute_control(
        self, state: State, reference: Optional[State] = None
    ) -> Action:
        """Compute adaptive control action"""
        if not self.is_initialized:
            self.initialize(state)

        # Base controller action
        base_action = self.base_controller.compute_control(state, reference)

        # RL policy action (if available)
        if self.rl_policy:
            rl_action = self.rl_policy.get_action(state)

            # Blend actions
            adaptive_action = Action(
                values=(1 - self.blending_factor) * base_action.values
                + self.blending_factor * rl_action.values,
                type=state.type,
            )
        else:
            adaptive_action = base_action

        return adaptive_action

    def update_parameters(self, state: State, action: Action, **kwargs) -> None:
        """Update adaptive parameters based on performance"""
        performance = kwargs.get("performance", 0.0)
        self.performance_history.append(performance)

        # Update blending factor based on performance
        if len(self.performance_history) > 10:
            recent_performance = np.mean(self.performance_history[-10:])

            if recent_performance > 0:
                # Increase RL contribution if performance is good
                self.blending_factor = min(
                    1.0, self.blending_factor + self.adaptation_rate
                )
            else:
                # Decrease RL contribution if performance is poor
                self.blending_factor = max(
                    0.0, self.blending_factor - self.adaptation_rate
                )

        # Update base controller parameters
        self.base_controller.update_parameters(state, action, **kwargs)


class RLController(Controller):
    """Pure RL-based controller"""

    def __init__(self, config: ControllerConfig, rl_policy: Policy):
        super().__init__(config)
        self.rl_policy = rl_policy

    def initialize(self, initial_state: State) -> None:
        """Initialize RL controller"""
        self.is_initialized = True

    def compute_control(
        self, state: State, reference: Optional[State] = None
    ) -> Action:
        """Compute RL control action"""
        if reference is not None:
            # Combine state and reference for goal-conditioned policies
            combined_state = np.concatenate([state.values, reference.values])
            combined_state_obj = State(values=combined_state, type=state.type)
            return self.rl_policy.get_action(combined_state_obj)
        else:
            return self.rl_policy.get_action(state)

    def update_parameters(self, state: State, action: Action, **kwargs) -> None:
        """Update RL policy parameters"""
        # This would typically be handled by the RL training loop
        pass


class HybridController(Controller):
    """Hybrid controller combining multiple control strategies"""

    def __init__(self, config: ControllerConfig, controllers: List[Controller]):
        super().__init__(config)

        self.controllers = controllers
        self.num_controllers = len(controllers)

        # Switching strategy
        self.switching_strategy = config.robust_params.get(
            "switching_strategy", "performance_based"
        )
        self.controller_weights = np.ones(self.num_controllers) / self.num_controllers

    def initialize(self, initial_state: State) -> None:
        """Initialize all controllers"""
        for controller in self.controllers:
            controller.initialize(initial_state)
        self.is_initialized = True

    def compute_control(
        self, state: State, reference: Optional[State] = None
    ) -> Action:
        """Compute hybrid control action"""
        if not self.is_initialized:
            self.initialize(state)

        actions = []
        for controller in self.controllers:
            action = controller.compute_control(state, reference)
            actions.append(action.values)

        # Combine actions based on weights
        combined_action = np.zeros_like(actions[0])
        for i, action in enumerate(actions):
            combined_action += self.controller_weights[i] * action

        return Action(values=combined_action, type=state.type)

    def update_parameters(self, state: State, action: Action, **kwargs) -> None:
        """Update hybrid controller parameters"""
        # Update individual controllers
        for controller in self.controllers:
            controller.update_parameters(state, action, **kwargs)

        # Update weights based on performance
        if "controller_performances" in kwargs:
            performances = kwargs["controller_performances"]
            self._update_weights(performances)

    def _update_weights(self, performances: List[float]) -> None:
        """Update controller weights based on performance"""
        if self.switching_strategy == "performance_based":
            # Convert performance to positive weights
            weights = np.maximum(0.1, np.array(performances))
            # Normalize weights
            self.controller_weights = weights / np.sum(weights)


class ControllerFactory:
    """Factory for creating controllers"""

    @staticmethod
    def create_controller(
        controller_type: ControllerType, config: ControllerConfig, **kwargs
    ) -> Controller:
        """Create controller based on type"""
        if controller_type == ControllerType.PID:
            return PIDController(config)
        elif controller_type == ControllerType.LQR:
            return LQRController(config)
        elif controller_type == ControllerType.MPC:
            return MPCController(config)
        elif controller_type == ControllerType.ADAPTIVE:
            base_controller = kwargs.get("base_controller")
            rl_policy = kwargs.get("rl_policy")
            return AdaptiveController(config, base_controller, rl_policy)
        elif controller_type == ControllerType.HYBRID:
            controllers = kwargs.get("controllers", [])
            return HybridController(config, controllers)
        else:
            raise ValueError(f"Unknown controller type: {controller_type}")


class ControllerIntegrator:
    """Integrates RL policies with traditional control systems"""

    def __init__(self, environment: RobotEnvironment):
        self.environment = environment
        self.controllers = {}
        self.active_controller = None

    def add_controller(self, name: str, controller: Controller) -> None:
        """Add a controller to the integrator"""
        self.controllers[name] = controller

    def set_active_controller(self, name: str) -> None:
        """Set the active controller"""
        if name in self.controllers:
            self.active_controller = self.controllers[name]
        else:
            raise ValueError(f"Controller {name} not found")

    def step(self, state: State, reference: Optional[State] = None) -> Action:
        """Execute control step with active controller"""
        if self.active_controller is None:
            raise ValueError("No active controller set")

        return self.active_controller.compute_control(state, reference)

    def update_all_controllers(self, state: State, action: Action, **kwargs) -> None:
        """Update all controllers with current state and action"""
        for controller in self.controllers.values():
            controller.update_parameters(state, action, **kwargs)
