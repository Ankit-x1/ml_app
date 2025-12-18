"""
Main Application Entry Point

Industrial Reinforcement Learning Framework for Robotics & Control
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.mdp import MDP, State, Action, Reward
from core.policy import ActorCritic, Policy, ValueFunction
from algorithms.policy_gradient import PPO, SAC
from algorithms.model_based import MBPO, EnsembleDynamicsModel, RewardModel
from environments.robots import (
    ManipulatorEnvironment,
    MobileRobotEnvironment,
    RobotType,
    RobotConfig,
    ControlMode,
)
from control.controllers import (
    ControllerFactory,
    ControllerType,
    ControllerConfig,
    ControllerIntegrator,
)
from deployment.deploy import (
    DeploymentManager,
    MonitoringManager,
    SecurityManager,
    create_deployment_configs,
)


class IndustrialRLFramework:
    """Main application class for industrial RL framework"""

    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.logger = self._setup_logging()
        self.config = self._load_config()

        # Initialize components
        self.environment = None
        self.policy = None
        self.value_function = None
        self.controller = None
        self.algorithm = None

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger("IndustrialRL")
        logger.setLevel(logging.INFO)

        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        import yaml

        try:
            with open(self.config_path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            return {}

    def setup_environment(self) -> None:
        """Setup robot environment"""
        env_config = self.config.get("environment", {})

        if env_config.get("type") == "manipulator":
            robot_config = RobotConfig(
                robot_type=RobotType.MANIPULATOR,
                control_mode=ControlMode.POSITION,
                num_joints=env_config.get("num_joints", 6),
                joint_limits=env_config.get("joint_limits", [(-3.14, 3.14)] * 6),
                max_velocity=env_config.get("max_velocity", [1.0] * 6),
                max_acceleration=env_config.get("max_acceleration", [2.0] * 6),
            )
            self.environment = ManipulatorEnvironment(robot_config, env_config)

        elif env_config.get("type") == "mobile":
            robot_config = RobotConfig(
                robot_type=RobotType.MOBILE,
                control_mode=ControlMode.VELOCITY,
                num_joints=env_config.get("num_joints", 2),
                joint_limits=env_config.get("joint_limits", [(-10, 10)] * 2),
                max_velocity=env_config.get("max_velocity", [5.0] * 2),
                max_acceleration=env_config.get("max_acceleration", [10.0] * 2),
            )
            self.environment = MobileRobotEnvironment(robot_config, env_config)

        self.logger.info(
            f"Environment setup complete: {type(self.environment).__name__}"
        )

    def setup_policy(self) -> None:
        """Setup neural network policy and value function"""
        if not self.environment:
            raise ValueError("Environment must be setup first")

        policy_config = self.config.get("policy", {})
        state_dim = self.environment.state_space["dimension"]
        action_dim = self.environment.action_space["dimension"]

        # Create actor-critic architecture
        self.policy = ActorCritic(state_dim, action_dim, policy_config)
        self.value_function = self.policy.critic

        self.logger.info(
            f"Policy setup complete: {policy_config.get('algorithm', 'PPO')}"
        )

    def setup_controller(self) -> None:
        """Setup control system"""
        if not self.policy:
            raise ValueError("Policy must be setup first")

        control_config = self.config.get("control", {})

        # Create controller configuration
        controller_config = ControllerConfig(
            controller_type=ControllerType(control_config.get("type", "adaptive")),
            sampling_time=control_config.get("sampling_time", 0.01),
            state_dimension=self.environment.state_space["dimension"],
            action_dimension=self.environment.action_space["dimension"],
            gains=control_config.get("gains", {}),
            constraints=control_config.get("constraints", {}),
        )

        # Create base controller
        base_controller = ControllerFactory.create_controller(
            ControllerType.PID, controller_config
        )

        # Create adaptive controller with RL policy
        adaptive_controller = ControllerFactory.create_controller(
            ControllerType.ADAPTIVE,
            controller_config,
            base_controller=base_controller,
            rl_policy=self.policy.actor,
        )

        # Setup controller integrator
        self.controller = ControllerIntegrator(self.environment)
        self.controller.add_controller("adaptive", adaptive_controller)
        self.controller.set_active_controller("adaptive")

        self.logger.info(
            f"Controller setup complete: {control_config.get('type', 'adaptive')}"
        )

    def setup_algorithm(self) -> None:
        """Setup RL algorithm"""
        if not self.policy or not self.environment:
            raise ValueError("Policy and environment must be setup first")

        alg_config = self.config.get("algorithm", {})
        algorithm_type = alg_config.get("type", "PPO")

        if algorithm_type == "PPO":
            self.algorithm = PPO(self.policy.actor, self.policy.critic, alg_config)
        elif algorithm_type == "SAC":
            self.algorithm = SAC(self.policy.actor, self.policy.critic, alg_config)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm_type}")

        self.logger.info(f"Algorithm setup complete: {algorithm_type}")

    def train(self, num_episodes: int = 1000) -> Dict[str, Any]:
        """Train the RL agent"""
        if not self.algorithm or not self.environment:
            raise ValueError("Algorithm and environment must be setup first")

        self.logger.info(f"Starting training for {num_episodes} episodes")

        training_stats = {"episode_rewards": [], "episode_lengths": [], "losses": []}

        for episode in range(num_episodes):
            state = self.environment.reset()
            episode_transitions = []
            episode_reward = 0
            episode_length = 0

            while True:
                action = self.algorithm.get_action(state)
                transition = self.environment.step(action)

                episode_transitions.append(transition)
                episode_reward += transition.reward.value
                episode_length += 1

                if transition.done:
                    break

                state = transition.next_state

            training_stats["episode_rewards"].append(episode_reward)
            training_stats["episode_lengths"].append(episode_length)

            # Update policy
            if len(episode_transitions) > 1:
                losses = self.algorithm.update([episode_transitions])
                training_stats["losses"].append(losses)

            if episode % 100 == 0:
                avg_reward = sum(training_stats["episode_rewards"][-100:]) / 100
                self.logger.info(
                    f"Episode {episode}: Average reward (last 100): {avg_reward:.2f}"
                )

        self.logger.info("Training completed")
        return training_stats

    def evaluate(self, num_episodes: int = 10) -> Dict[str, Any]:
        """Evaluate the trained policy"""
        if not self.algorithm or not self.environment:
            raise ValueError("Algorithm and environment must be setup first")

        self.logger.info(f"Starting evaluation for {num_episodes} episodes")

        evaluation_stats = {
            "episode_rewards": [],
            "episode_lengths": [],
            "success_rate": 0.0,
        }

        successful_episodes = 0

        for episode in range(num_episodes):
            state = self.environment.reset()
            episode_reward = 0
            episode_length = 0

            while True:
                action = self.algorithm.get_action(state)
                transition = self.environment.step(action)

                episode_reward += transition.reward.value
                episode_length += 1

                if transition.done:
                    if episode_reward > 0:  # Assuming positive reward means success
                        successful_episodes += 1
                    break

                state = transition.next_state

            evaluation_stats["episode_rewards"].append(episode_reward)
            evaluation_stats["episode_lengths"].append(episode_length)

        evaluation_stats["success_rate"] = successful_episodes / num_episodes
        avg_reward = sum(evaluation_stats["episode_rewards"]) / num_episodes

        self.logger.info(
            f"Evaluation complete - Average reward: {avg_reward:.2f}, Success rate: {evaluation_stats['success_rate']:.2%}"
        )

        return evaluation_stats

    def deploy(self, environment_type: str = "production") -> bool:
        """Deploy the system"""
        self.logger.info(f"Starting deployment for {environment_type}")

        # Create deployment configurations
        create_deployment_configs(str(self.config_path.parent))

        # Setup deployment
        deploy_config_path = (
            self.config_path.parent / "config" / f"{environment_type}.yaml"
        )
        deployment_manager = DeploymentManager(deploy_config_path)

        success = deployment_manager.deploy()

        if success:
            # Setup monitoring
            monitoring_manager = MonitoringManager(deploy_config_path)
            monitoring_manager.setup_monitoring()

            # Setup security
            security_manager = SecurityManager(deploy_config_path)
            security_manager.setup_security()

            self.logger.info(f"Deployment to {environment_type} completed successfully")

        return success


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Industrial RL Framework for Robotics & Control"
    )
    parser.add_argument(
        "--config", type=str, default="config/app.yaml", help="Configuration file path"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "evaluate", "deploy"],
        default="train",
        help="Operation mode",
    )
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes")
    parser.add_argument(
        "--environment", type=str, default="production", help="Deployment environment"
    )

    args = parser.parse_args()

    # Initialize framework
    framework = IndustrialRLFramework(args.config)

    try:
        # Setup components
        framework.setup_environment()
        framework.setup_policy()
        framework.setup_controller()
        framework.setup_algorithm()

        # Execute mode
        if args.mode == "train":
            stats = framework.train(args.episodes)
            print(f"Training completed: {stats}")

        elif args.mode == "evaluate":
            stats = framework.evaluate(args.episodes)
            print(f"Evaluation completed: {stats}")

        elif args.mode == "deploy":
            success = framework.deploy(args.environment)
            print(f"Deployment {'successful' if success else 'failed'}")

    except Exception as e:
        logging.error(f"Application error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
