import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional

from .planner import SymbolicPlanner
from .icm import ICMModule
from .rnd import RNDModule
from .pseudocount import PseudoCountExploration


class PPOPolicy(nn.Module):
    def __init__(self, input_dim: int, action_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.actor(x), self.critic(x)

    def act(self, x: torch.Tensor):
        logits, value = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value


def compute_gae(
        rewards: List[float],
        values: List[float],
        gamma: float = 0.99,
        lam: float = 0.95):
    advantages: List[float] = []
    gae = 0
    values = values + [0]
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    return advantages


def train_agent(
    env,
    policy: PPOPolicy,
    icm: ICMModule,
    planner: SymbolicPlanner,
    optimizer_policy: torch.optim.Optimizer,
    optimizer_icm: torch.optim.Optimizer,
    use_icm=True,
    use_planner=True,
    num_episodes: int = 500,
    beta: float = 0.1,
    final_beta: Optional[float] = None,
    gamma: float = 0.99,
    planner_weights: Optional[dict] = None,
    rnd: Optional[RNDModule] = None,
    pseudo: Optional[PseudoCountExploration] = None,
    seed: int = 42,
    add_noise: bool = False,
    logger=None,
    initial_bonus: float = 0.5,
    reset_env: bool = True,
):
    """Train a PPO agent with optional curiosity and planning.

    The ``beta`` parameter controls the strength of the intrinsic reward.
    When ``final_beta`` is provided its value linearly decays from ``beta`` to
    ``final_beta`` over the first two thirds of training episodes and then
    remains constant for the rest of training.

    Parameters
    ----------
    reset_env : bool, optional
        If ``True`` the environment is reset and its map saved before
        training begins. Set to ``False`` to use the environment state and
        maps provided by the caller.
    """

    reward_log = []
    paths_log = []
    planner_usage = []
    ppo_usage = []
    intrinsic_rewards = []
    extrinsic_rewards = []
    step_counts = []
    success_flags = []
    planner_usage_rate = []

    initial_bonus = max(0.1, float(initial_bonus))

    benchmark_map = "maps/map_00.npz"
    os.makedirs(os.path.dirname(benchmark_map), exist_ok=True)
    if reset_env:
        env.reset(seed=seed)
        env.save_map(benchmark_map)

    for episode in range(num_episodes):
        obs, _ = env.reset(
            seed=seed, load_map_path=benchmark_map, add_noise=add_noise)

        done = False
        total_ext_reward = 0
        step_count = 0
        info = {}
        terrain_decay = max(0.1, 1 - (episode / 1000))
        obs_buf = []
        action_buf = []
        logprob_buf = []
        val_buf = []
        reward_buf = []
        agent_path = []
        planner_decisions = 0
        if planner_weights is not None:
            planner = SymbolicPlanner(
                env.cost_map,
                env.risk_map,
                env.np_random,
                cost_weight=planner_weights.get("cost_weight", 2.0),
                risk_weight=planner_weights.get("risk_weight", 3.0),
                revisit_penalty=planner_weights.get("revisit_penalty", 1.0),
            )
        else:
            planner.cost_map = env.cost_map
            planner.risk_map = env.risk_map
            planner.np_random = env.np_random
            planner.grid_size = env.grid_size
            planner.reset()

        ppo_decisions = 0
        intrinsic_log = 0
        visit_count = np.zeros((env.grid_size, env.grid_size))
        planner_bonus = max(0.1, initial_bonus *
                            (1 - (episode / num_episodes)))

        if final_beta is not None:
            decay_end = max(1, int(num_episodes * 2 / 3))
            progress = min(episode, decay_end) / decay_end
            beta_val = max(final_beta, beta - (beta - final_beta) * progress)
        else:
            beta_val = beta

        while not done:
            state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            risk = env.risk_map[env.agent_pos[0]][env.agent_pos[1]]
            if risk > 0.7:
                planner.risk_weight = 5.0
                planner.cost_weight = 1.0
                use_planner_now = use_planner
            elif risk > 0.5:
                planner.risk_weight = 3.0
                planner.cost_weight = 2.0
                use_planner_now = use_planner
            elif risk > 0.2:
                planner.risk_weight = 1.0
                planner.cost_weight = 3.0
                use_planner_now = use_planner
            else:
                use_planner_now = False
            used_planner = False
            if use_planner_now:
                action = planner.get_safe_subgoal(env.agent_pos)
                used_planner = True
                logprob = torch.tensor([0.0])
                value = torch.tensor(0.0)
                planner_decisions += 1
            else:
                action, logprob, value = policy.act(state_tensor)
                ppo_decisions += 1

            next_obs, ext_reward, cost_t, done, _, info = env.step(
                action, terrain_decay=terrain_decay
            )
            x, y = env.agent_pos
            visit_count[x][y] += 1
            count_reward = 1.0 / np.sqrt(visit_count[x][y])

            if used_planner and env.risk_map[env.agent_pos[0]
                                             ][env.agent_pos[1]] < 0.5:
                ext_reward += planner_bonus

            next_tensor = torch.tensor(
                next_obs, dtype=torch.float32).unsqueeze(0)
            action_tensor = torch.tensor([action])

            if use_icm == "count":
                total_reward = ext_reward + beta_val * count_reward
                curiosity = torch.tensor([count_reward])
            elif use_icm == "rnd" and rnd is not None:
                r_int, pred, target = rnd(state_tensor)
                total_reward = ext_reward + beta_val * r_int.item()
                curiosity = r_int
                loss = F.mse_loss(pred, target.detach())
                optimizer_icm.zero_grad()
                loss.backward()
                optimizer_icm.step()
            elif use_icm == "pseudo" and pseudo is not None:
                p_bonus = pseudo.bonus(state_tensor.numpy())
                total_reward = ext_reward + beta_val * p_bonus
                curiosity = torch.tensor([p_bonus])
            elif use_icm is True:
                curiosity, f_loss, i_loss = icm(
                    state_tensor, next_tensor, action_tensor)
                total_reward = ext_reward + beta_val * curiosity.item()
                icm_loss = f_loss + i_loss
                optimizer_icm.zero_grad()
                icm_loss.backward()
                optimizer_icm.step()
            else:
                curiosity = torch.tensor([0.0])
                total_reward = ext_reward

            intrinsic_log += curiosity.item()

            agent_path.append(tuple(env.agent_pos))
            if episode > 200 and step_count % 10 == 0:
                cx, cy = env.grid_size // 2, env.grid_size // 2
                env.risk_map[cx][cy] = min(1.0, env.risk_map[cx][cy] + 0.05)

            obs = next_obs
            total_ext_reward += ext_reward
            step_count += 1

            obs_buf.append(obs)
            action_buf.append(action)
            logprob_buf.append(logprob)
            val_buf.append(value.item())
            reward_buf.append(total_reward)

        reward_range = max(reward_buf) - min(reward_buf) + 1e-8
        reward_buf = [
            (r - min(reward_buf)) / reward_range
            for r in reward_buf
        ]
        advantages = compute_gae(reward_buf, val_buf, gamma=gamma, lam=0.95)

        adv_tensor = torch.tensor(advantages, dtype=torch.float32)
        adv_tensor = (adv_tensor - adv_tensor.mean()) / \
            (adv_tensor.std() + 1e-8)

        obs_tensor = torch.tensor(np.array(obs_buf), dtype=torch.float32)
        action_tensor = torch.tensor(action_buf)
        logprob_tensor = torch.stack(logprob_buf)
        logits, new_vals = policy(obs_tensor)
        dist = torch.distributions.Categorical(logits=logits)
        new_logprob = dist.log_prob(action_tensor)
        entropy = dist.entropy()

        ratio = torch.exp(new_logprob - logprob_tensor)
        surr1 = ratio * adv_tensor
        surr2 = torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * adv_tensor
        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = F.mse_loss(
            new_vals.squeeze(), torch.tensor(
                reward_buf, dtype=torch.float32))
        entropy_loss = -0.01 * entropy.mean()

        total_loss = policy_loss + 0.5 * value_loss + entropy_loss
        optimizer_policy.zero_grad()
        total_loss.backward()
        optimizer_policy.step()

        if episode % 50 == 0:
            paths_log.append(agent_path)

        reward_log.append(total_ext_reward)
        planner_usage.append(planner_decisions)
        ppo_usage.append(ppo_decisions)
        intrinsic_rewards.append(intrinsic_log)
        extrinsic_rewards.append(total_ext_reward)
        step_counts.append(step_count)
        survived = step_count >= env.max_steps and not info.get("dead", False)
        success_flags.append(1 if survived else 0)
        if (planner_decisions + ppo_decisions) > 0:
            planner_percent = planner_decisions / \
                (planner_decisions + ppo_decisions)
        else:
            planner_percent = 0
        planner_usage_rate.append(planner_percent)

        success_rate = np.mean(success_flags)
        if logger is not None:
            if hasattr(logger, "add_scalar"):
                logger.add_scalar("reward", total_ext_reward, episode)
                logger.add_scalar("intrinsic_reward", intrinsic_log, episode)
                logger.add_scalar("success_rate", success_rate, episode)
            elif hasattr(logger, "log"):
                logger.log(
                    {
                        "reward": total_ext_reward,
                        "intrinsic_reward": intrinsic_log,
                        "success_rate": success_rate,
                        "episode": episode,
                    }
                )

        print(
            f"Episode {episode + 1:03d}/{num_episodes} | "
            f"Reward: {total_ext_reward:.2f} | "
            f"Steps: {step_count} | "
            f"PPO: {ppo_decisions} | "
            f"Planner: {planner_decisions} | "
            f"Success: {success_rate * 100:.1f}%"
        )

    return (
        reward_log,
        intrinsic_rewards,
        extrinsic_rewards,
        planner_usage,
        paths_log,
        step_counts,
        success_flags,
        planner_usage_rate,
    )
