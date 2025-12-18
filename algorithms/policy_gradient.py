"""
Policy Optimization Algorithms

Implements state-of-the-art policy gradient methods for
industrial robotics and control systems.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.kl import kl_divergence
from collections import deque
import time

from ..core.mdp import Transition, State, Action, Reward
from ..core.policy import ActorCritic, Policy, ValueFunction


class PolicyOptimizer(ABC):
    """Abstract base class for policy optimization algorithms"""

    def __init__(
        self,
        policy: Policy,
        value_function: Optional[ValueFunction] = None,
        config: Dict[str, Any] = None,
    ):
        self.policy = policy
        self.value_function = value_function
        self.config = config or {}
        self.device = torch.device(config.get("device", "cpu"))
        self.training_stats = {}

    @abstractmethod
    def update(self, trajectories: List[List[Transition]]) -> Dict[str, float]:
        """Update policy based on collected trajectories"""
        pass

    @abstractmethod
    def get_action(self, state: State) -> Action:
        """Get action from current policy"""
        pass


class PPO(PolicyOptimizer):
    """Proximal Policy Optimization"""

    def __init__(
        self, policy: Policy, value_function: ValueFunction, config: Dict[str, Any]
    ):
        super().__init__(policy, value_function, config)

        self.policy = policy.to(self.device)
        self.value_function = value_function.to(self.device)

        self.learning_rate = config.get("learning_rate", 3e-4)
        self.gamma = config.get("gamma", 0.99)
        self.gae_lambda = config.get("gae_lambda", 0.95)
        self.clip_epsilon = config.get("clip_epsilon", 0.2)
        self.value_loss_coef = config.get("value_loss_coef", 0.5)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        self.max_grad_norm = config.get("max_grad_norm", 0.5)
        self.ppo_epochs = config.get("ppo_epochs", 4)
        self.mini_batch_size = config.get("mini_batch_size", 64)

        self.policy_optimizer = optim.Adam(
            self.policy.parameters(), lr=self.learning_rate
        )
        self.value_optimizer = optim.Adam(
            self.value_function.parameters(), lr=self.learning_rate
        )

    def compute_gae(
        self, rewards: List[float], values: List[float], dones: List[bool]
    ) -> Tuple[List[float], List[float]]:
        """Compute Generalized Advantage Estimation"""
        advantages = []
        returns = []

        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])

        return advantages, returns

    def update(self, trajectories: List[List[Transition]]) -> Dict[str, float]:
        """Update policy using PPO"""
        all_states = []
        all_actions = []
        all_rewards = []
        all_dones = []
        all_old_log_probs = []

        for traj in trajectories:
            states = [torch.FloatTensor(t.state.values) for t in traj]
            actions = [torch.FloatTensor(t.action.values) for t in traj]
            rewards = [t.reward.value for t in traj]
            dones = [t.done for t in traj]

            with torch.no_grad():
                actor_output = self.policy.evaluate_actions(
                    torch.stack(states), torch.stack(actions)
                )
                old_log_probs = actor_output["log_probs"]
                values = self.value_function(torch.stack(states))

            all_states.extend(states)
            all_actions.extend(actions)
            all_rewards.extend(rewards)
            all_dones.extend(dones)
            all_old_log_probs.extend(old_log_probs)

        states_tensor = torch.stack(all_states)
        actions_tensor = torch.stack(all_actions)
        old_log_probs_tensor = torch.stack(all_old_log_probs)

        values_list = []
        with torch.no_grad():
            for state in all_states:
                values_list.append(self.value_function(state.unsqueeze(0)).item())

        advantages, returns = self.compute_gae(all_rewards, values_list, all_dones)

        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)

        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (
            advantages_tensor.std() + 1e-8
        )

        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0

        dataset_size = len(states_tensor)
        indices = np.random.permutation(dataset_size)

        for epoch in range(self.ppo_epochs):
            for start in range(0, dataset_size, self.mini_batch_size):
                end = start + self.mini_batch_size
                batch_indices = indices[start:end]

                batch_states = states_tensor[batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]

                actor_output = self.policy.evaluate_actions(batch_states, batch_actions)
                current_log_probs = actor_output["log_probs"]
                current_values = self.value_function(batch_states)
                entropy = actor_output["entropy"]

                ratio = torch.exp(current_log_probs - batch_old_log_probs)

                surr1 = ratio * batch_advantages
                surr2 = (
                    torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                    * batch_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(current_values, batch_returns)
                entropy_loss = -entropy.mean()

                self.policy_optimizer.zero_grad()
                (policy_loss + self.entropy_coef * entropy_loss).backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy_optimizer.step()

                self.value_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.value_function.parameters(), self.max_grad_norm
                )
                self.value_optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()

        num_updates = self.ppo_epochs * (dataset_size // self.mini_batch_size)

        return {
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy_loss": total_entropy_loss / num_updates,
            "avg_return": np.mean(returns),
            "avg_advantage": np.mean(advantages),
        }

    def get_action(self, state: State) -> Action:
        """Get action from policy"""
        state_tensor = torch.FloatTensor(state.values).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_values, _ = self.policy.sample(state_tensor)
        action_values_np = action_values.cpu().numpy()[0]

        return Action(values=action_values_np, type=state.type)


class TRPO(PolicyOptimizer):
    """Trust Region Policy Optimization"""

    def __init__(
        self, policy: Policy, value_function: ValueFunction, config: Dict[str, Any]
    ):
        super().__init__(policy, value_function, config)

        self.policy = policy.to(self.device)
        self.value_function = value_function.to(self.device)

        self.gamma = config.get("gamma", 0.99)
        self.gae_lambda = config.get("gae_lambda", 0.95)
        self.damping = config.get("damping", 0.1)
        self.max_kl = config.get("max_kl", 0.01)
        self.cg_iters = config.get("cg_iters", 10)

        self.value_optimizer = optim.Adam(self.value_function.parameters(), lr=3e-4)

    def conjugate_gradient(
        self, grad: torch.Tensor, states: torch.Tensor
    ) -> torch.Tensor:
        """Conjugate gradient solver for TRPO"""
        x = torch.zeros_like(grad)
        r = grad.clone()
        p = grad.clone()
        rdotr = torch.dot(r, r)

        for i in range(self.cg_iters):
            Ap = self.hessian_vector_product(p, states)
            alpha = rdotr / (torch.dot(p, Ap) + 1e-8)
            x += alpha * p
            r -= alpha * Ap
            new_rdotr = torch.dot(r, r)
            if new_rdotr < 1e-10:
                break
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr

        return x

    def hessian_vector_product(
        self, vector: torch.Tensor, states: torch.Tensor
    ) -> torch.Tensor:
        """Compute Hessian-vector product"""
        with torch.no_grad():
            actor_output = self.policy.evaluate_actions(states, None)
            old_log_probs = actor_output["log_probs"]
            old_dist = actor_output.get("distribution")

        def new_loss_fn():
            actor_output = self.policy.evaluate_actions(states, None)
            new_log_probs = actor_output["log_probs"]
            new_dist = actor_output.get("distribution")

            kl = kl_divergence(old_dist, new_dist).mean()
            return kl

        grad = torch.autograd.grad(
            new_loss_fn(), self.policy.parameters(), create_graph=True
        )
        flat_grad = torch.cat([g.view(-1) for g in grad])

        grad_vector_product = torch.dot(flat_grad, vector)
        hessian = torch.autograd.grad(grad_vector_product, self.policy.parameters())
        flat_hessian = torch.cat([g.contiguous().view(-1) for g in hessian])

        return flat_hessian + self.damping * vector

    def update(self, trajectories: List[List[Transition]]) -> Dict[str, float]:
        """Update policy using TRPO"""
        states, actions, rewards, dones, old_log_probs = self._process_trajectories(
            trajectories
        )

        advantages, returns = self._compute_advantages(states, rewards, dones)

        def compute_loss():
            actor_output = self.policy.evaluate_actions(states, actions)
            current_log_probs = actor_output["log_probs"]
            ratio = torch.exp(current_log_probs - old_log_probs)

            policy_loss = -(ratio * advantages).mean()
            return policy_loss

        self._update_value_function(states, returns)

        grad = torch.autograd.grad(compute_loss(), self.policy.parameters())
        flat_grad = torch.cat([g.view(-1) for g in grad])

        step_dir = self.conjugate_gradient(flat_grad, states)

        step_size = torch.sqrt(
            2
            * self.max_kl
            / (
                torch.dot(step_dir, self.hessian_vector_product(step_dir, states))
                + 1e-8
            )
        )
        full_step = step_size * step_dir

        self._line_search(
            states, actions, advantages, old_log_probs, full_grad, full_step
        )

        return {
            "policy_loss": compute_loss().item(),
            "avg_return": returns.mean().item(),
            "kl_divergence": self._compute_kl(states).item(),
        }

    def get_action(self, state: State) -> Action:
        """Get action from policy"""
        state_tensor = torch.FloatTensor(state.values).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_values, _ = self.policy.sample(state_tensor)
        action_values_np = action_values.cpu().numpy()[0]

        return Action(values=action_values_np, type=state.type)


class SAC(PolicyOptimizer):
    """Soft Actor-Critic for continuous control"""

    def __init__(
        self, policy: Policy, value_function: ValueFunction, config: Dict[str, Any]
    ):
        super().__init__(policy, value_function, config)

        self.policy = policy.to(self.device)
        self.value_function = value_function.to(self.device)

        self.learning_rate = config.get("learning_rate", 3e-4)
        self.gamma = config.get("gamma", 0.99)
        self.tau = config.get("tau", 0.005)
        self.alpha = config.get("alpha", 0.2)
        self.target_update_interval = config.get("target_update_interval", 1)

        self.policy_optimizer = optim.Adam(
            self.policy.parameters(), lr=self.learning_rate
        )
        self.q1_optimizer = optim.Adam(
            self.value_function.parameters(), lr=self.learning_rate
        )

        self.q_target = type(self.value_function)(
            self.value_function.state_dim, self.value_function.config
        ).to(self.device)
        self.q_target.load_state_dict(self.value_function.state_dict())

        self.update_count = 0

    def update(self, trajectories: List[List[Transition]]) -> Dict[str, float]:
        """Update policy using SAC"""
        batch = self._sample_batch(trajectories)

        states = torch.FloatTensor([t.state.values for t in batch]).to(self.device)
        actions = torch.FloatTensor([t.action.values for t in batch]).to(self.device)
        rewards = torch.FloatTensor([t.reward.value for t in batch]).to(self.device)
        next_states = torch.FloatTensor([t.next_state.values for t in batch]).to(
            self.device
        )
        dones = torch.FloatTensor([t.done for t in batch]).to(self.device)

        with torch.no_grad():
            next_actions, next_log_probs = self.policy.sample(next_states)
            target_q1_next = self.q_target(next_states, next_actions)
            target_q_next = target_q1_next - self.alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * target_q_next

        current_q1 = self.value_function(states, actions)
        q1_loss = F.mse_loss(current_q1, target_q)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        new_actions, new_log_probs = self.policy.sample(states)
        q_new = self.value_function(states, new_actions)

        policy_loss = (self.alpha * new_log_probs - q_new).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        if self.update_count % self.target_update_interval == 0:
            for target_param, param in zip(
                self.q_target.parameters(), self.value_function.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

        self.update_count += 1

        return {
            "q_loss": q1_loss.item(),
            "policy_loss": policy_loss.item(),
            "avg_q": current_q1.mean().item(),
            "entropy": -new_log_probs.mean().item(),
        }

    def get_action(self, state: State) -> Action:
        """Get action from policy"""
        state_tensor = torch.FloatTensor(state.values).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_values, _ = self.policy.sample(state_tensor)
        action_values_np = action_values.cpu().numpy()[0]

        return Action(values=action_values_np, type=state.type)
