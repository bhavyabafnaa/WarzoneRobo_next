import os
from typing import List, Tuple

import gym
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from scipy import stats

from .planner import SymbolicPlanner


class GridWorldICM:
    """Simple grid world used by training script."""

    def __init__(
        self,
        grid_size: int = 6,
        dynamic_risk: bool = False,
        dynamic_cost: bool = False,
        reward_clip: Tuple[float, float] = (-10, 100),
        max_steps: int = 100,
        survival_reward: float = 0.05,
        enemy_speed: int = 1,
        enemy_speed_range: Tuple[int, int] | None = None,
        enemy_policy: str = "random",
        enemy_policy_options: List[str] | None = None,
        mine_density: float = 0.05,
        mine_density_range: Tuple[float, float] | None = None,
        hazard_density_range: Tuple[float, float] | None = None,
        seed: int | None = None,
    ) -> None:
        self.grid_size = grid_size
        self.dynamic_risk = dynamic_risk
        self.dynamic_cost = dynamic_cost
        self.reward_clip = reward_clip
        self.max_steps = max_steps
        self.survival_reward = survival_reward
        self.enemy_speed = enemy_speed
        self.enemy_speed_range = enemy_speed_range
        self.enemy_policy = enemy_policy
        self.enemy_policy_options = enemy_policy_options
        self.mine_density = mine_density
        self.mine_density_range = mine_density_range
        self.hazard_density_range = hazard_density_range
        # Track cumulative cost over an episode
        self.episode_cost = 0.0
        if seed is None:
            seed = 0
        self.seed(seed)
        self.reset()

    def seed(self, seed: int | None = None) -> int:
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return seed

    def reset(
        self,
        seed: int | None = None,
        load_map_path: str | None = None,
        add_noise: bool = False,
    ) -> Tuple[np.ndarray, dict]:
        if seed is not None:
            self.seed(seed)

        # randomize episode-specific parameters
        if self.mine_density_range is not None:
            self.mine_density = float(
                self.np_random.uniform(*self.mine_density_range)
            )
        if self.enemy_speed_range is not None:
            self.enemy_speed = int(
                self.np_random.integers(
                    self.enemy_speed_range[0],
                    self.enemy_speed_range[1] + 1,
                )
            )
        if self.enemy_policy_options:
            self.enemy_policy = str(self.np_random.choice(self.enemy_policy_options))

        if load_map_path:
            self.load_map(load_map_path)
        else:
            self.cost_map = np.ones((self.grid_size, self.grid_size)) * 0.3
            # initialize hazard (risk) map
            self.hazard_density = 0.0
            if self.hazard_density_range is not None:
                self.hazard_density = float(
                    self.np_random.uniform(*self.hazard_density_range)
                )
            self.risk_map = np.zeros((self.grid_size, self.grid_size))
            if self.hazard_density > 0:
                hazard_cells = (
                    self.np_random.random((self.grid_size, self.grid_size))
                    < self.hazard_density
                )
                hazard_vals = self.np_random.uniform(
                    0.3, 0.7, size=(self.grid_size, self.grid_size)
                )
                self.risk_map = np.where(hazard_cells, hazard_vals, 0.0)
            self.terrain_map = np.full(
                (self.grid_size, self.grid_size), "normal")
            self.mine_map = self.np_random.random(
                (self.grid_size, self.grid_size)) < self.mine_density
            self.terrain_map[self.np_random.random(
                (self.grid_size, self.grid_size)) < 0.15] = "mud"
            self.terrain_map[self.np_random.random(
                (self.grid_size, self.grid_size)) < 0.15] = "water"
            self.enemy_positions = [[
                int(self.np_random.integers(0, self.grid_size)),
                int(self.np_random.integers(0, self.grid_size)),
            ]]

        if add_noise:
            noise = np.random.normal(0, 0.05, self.risk_map.shape)
            self.risk_map = np.clip(self.risk_map + noise, 0.0, 1.0)
            noise = np.random.normal(0, 0.05, self.cost_map.shape)
            self.cost_map = np.clip(self.cost_map + noise, 0.0, 1.0)

        self.agent_pos = (
            self.np_random.integers(0, self.grid_size, size=2).tolist()
        )
        self.steps = 0
        self.episode_cost = 0.0

        return self._get_obs(), {}

    def load_map(self, filepath: str = "map_data.npz") -> None:
        data = np.load(filepath, allow_pickle=True)
        self.cost_map = data["cost_map"]
        self.risk_map = data["risk_map"]
        self.terrain_map = data["terrain_map"]
        self.mine_map = data["mine_map"]
        self.enemy_positions = data["enemy_positions"].tolist()
        self.grid_size = self.cost_map.shape[0]
        self.enemy_speed = int(data["enemy_speed"]) if "enemy_speed" in data else 1
        self.enemy_policy = (
            str(data["enemy_policy"])
            if "enemy_policy" in data
            else "random"
        )
        self.mine_density = (
            float(data["mine_density"]) if "mine_density" in data else 0.05
        )
        self.hazard_density = (self.risk_map > 0).mean()

    def save_map(self, filepath: str = "map_data.npz") -> None:
        np.savez_compressed(
            filepath,
            cost_map=self.cost_map,
            risk_map=self.risk_map,
            terrain_map=self.terrain_map,
            mine_map=self.mine_map,
            enemy_positions=np.array(self.enemy_positions),
            enemy_speed=self.enemy_speed,
            enemy_policy=self.enemy_policy,
            mine_density=self.mine_density,
        )

    def update_cost_map(self) -> None:
        """Apply decay and increase costs around mines when enabled."""
        if not getattr(self, "dynamic_cost", False):
            return

        # decay existing costs slightly each step
        self.cost_map *= 0.98

        # raise costs near mines to discourage paths through them
        mine_indices = np.argwhere(self.mine_map)
        for mx, my in mine_indices:
            self.cost_map[mx, my] = min(1.0, self.cost_map[mx, my] + 0.5)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = mx + dx, my + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    self.cost_map[nx, ny] = min(
                        1.0, self.cost_map[nx, ny] + 0.25
                    )

    def step(self, action: int, terrain_decay: float = 1.0):
        x, y = self.agent_pos
        if action == 0 and x > 0:
            x -= 1
        elif action == 1 and x < self.grid_size - 1:
            x += 1
        elif action == 2 and y > 0:
            y -= 1
        elif action == 3 and y < self.grid_size - 1:
            y += 1
        self.agent_pos = [x, y]

        self.steps += 1

        for enemy in self.enemy_positions:
            if self.enemy_policy == "aggressive":
                dx = np.sign(self.agent_pos[0] - enemy[0]) * self.enemy_speed
                dy = np.sign(self.agent_pos[1] - enemy[1]) * self.enemy_speed
            elif self.enemy_policy == "stationary":
                dx = dy = 0
            else:
                dx, dy = self.np_random.choice([-1, 0, 1], size=2)
                dx *= self.enemy_speed
                dy *= self.enemy_speed
            enemy[0] = int(np.clip(enemy[0] + dx, 0, self.grid_size - 1))
            enemy[1] = int(np.clip(enemy[1] + dy, 0, self.grid_size - 1))

        if self.dynamic_risk:
            self.risk_map *= 0.95
            for ex, ey in self.enemy_positions:
                self.risk_map[ex, ey] = min(1.0, self.risk_map[ex, ey] + 0.5)
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = ex + dx, ey + dy
                    if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                        self.risk_map[nx, ny] = min(
                            1.0, self.risk_map[nx, ny] + 0.25
                        )

        self.update_cost_map()
        cost_t = np.clip(self.cost_map[x][y], 0, 1)
        risk = np.clip(self.risk_map[x][y], 0, 1)

        reward = self.survival_reward - cost_t - risk
        terrain = self.terrain_map[x][y]
        if terrain == "mud":
            reward -= 0.2 * terrain_decay
        elif terrain == "water":
            reward -= 0.1 * terrain_decay

        # accumulate episode cost
        self.episode_cost += cost_t

        if self.mine_map[x][y]:
            reward -= 2
            return (
                self._get_obs(),
                self._clip_reward(reward),
                cost_t,
                True,
                False,
                {"dead": True},
            )

        for ex, ey in self.enemy_positions:
            if [x, y] == [ex, ey]:
                reward -= 3
                return (
                    self._get_obs(),
                    self._clip_reward(reward),
                    cost_t,
                    True,
                    False,
                    {"dead": True},
                )

        done = self.steps >= self.max_steps

        return (
            self._get_obs(),
            self._clip_reward(reward),
            cost_t,
            done,
            False,
            {},
        )

    def _clip_reward(self, r: float) -> float:
        return np.clip(r, self.reward_clip[0], self.reward_clip[1])

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros((4, self.grid_size, self.grid_size), dtype=np.float32)
        obs[0, self.agent_pos[0], self.agent_pos[1]] = 1.0
        obs[1] = self.cost_map
        obs[2] = self.risk_map
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.terrain_map[i][j] == "mud":
                    obs[3, i, j] = 0.5
                elif self.terrain_map[i][j] == "water":
                    obs[3, i, j] = 0.8
                if self.mine_map[i][j]:
                    obs[3, i, j] = 1.0
        return obs.flatten()

    def render(self) -> np.ndarray:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.backends.backend_agg import (
            FigureCanvasAgg as FigureCanvas,
        )

        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)
        canvas = FigureCanvas(fig)
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        ax.set_aspect("equal")
        ax.axis("off")

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                color = "white"
                if self.terrain_map[i][j] == "mud":
                    color = "#C2B280"
                elif self.terrain_map[i][j] == "water":
                    color = "#99ccff"
                if self.mine_map[i][j]:
                    color = "#ff9999"
                if self.risk_map[i][j] > 0:
                    color = "#ffcccb"
                ax.add_patch(
                    patches.Rectangle(
                        (j, self.grid_size - 1 - i),
                        1,
                        1,
                        facecolor=color,
                        edgecolor="gray",
                    )
                )

        for ex, ey in self.enemy_positions:
            ax.add_patch(
                patches.Circle(
                    (ey + 0.5, self.grid_size - 1 - ex + 0.5), 0.3, color="red"
                )
            )
        ax.add_patch(
            patches.Circle(
                (
                    self.agent_pos[1] + 0.5,
                    self.grid_size - 1 - self.agent_pos[0] + 0.5,
                ),
                0.3,
                color="black",
            )
        )

        canvas.draw()
        width, height = fig.get_size_inches() * fig.get_dpi()
        img = (
            np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
            .reshape(int(height), int(width), 4)[..., :3]
        )
        plt.close(fig)
        return img


def visualize_paths_on_benchmark_maps(
    env: GridWorldICM,
    policy,
    map_folder: str = "maps/",
    num_maps: int = 9,
    grid_cols: int = 3,
    save: bool = False,
    H: int = 8,
):
    from matplotlib.collections import LineCollection
    import torch

    grid_rows = int(np.ceil(num_maps / grid_cols))
    fig, axs = plt.subplots(
        grid_rows, grid_cols, figsize=(
            grid_cols * 4, grid_rows * 4))
    axs = axs.flatten()

    for i in range(num_maps):
        map_path = os.path.join(map_folder, f"map_{i:02d}.npz")
        obs, _ = env.reset(load_map_path=map_path)
        planner = SymbolicPlanner(env.cost_map, env.risk_map, env.np_random)
        g = planner.get_subgoal(env.agent_pos, H)
        subgoal_timer = H
        dx, dy = g[0] - env.agent_pos[0], g[1] - env.agent_pos[1]
        obs = np.concatenate([obs, np.array([dx, dy], dtype=np.float32)])
        path = [tuple(env.agent_pos)]
        rewards: List[float] = []

        done = False
        while not done:
            state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            action, _, _, _ = policy.act(state_tensor)
            obs_base, reward, cost_t, done, _, _ = env.step(action)
            subgoal_timer -= 1
            if subgoal_timer <= 0 or tuple(env.agent_pos) == g:
                g = planner.get_subgoal(env.agent_pos, H)
                subgoal_timer = H
            dx, dy = g[0] - env.agent_pos[0], g[1] - env.agent_pos[1]
            obs = np.concatenate([obs_base, np.array([dx, dy], dtype=np.float32)])
            path.append(tuple(env.agent_pos))
            rewards.append(reward)

        segments = [
            [(path[j][1], path[j][0]), (path[j + 1][1], path[j + 1][0])]
            for j in range(len(path) - 1)
        ]
        rewards_arr = np.array(rewards)
        norm = plt.Normalize(rewards_arr.min(), rewards_arr.max())
        lc = LineCollection(segments, cmap="RdYlGn", norm=norm, linewidth=2)
        lc.set_array(rewards_arr)

        ax = axs[i]
        ax.imshow(
            np.zeros(
                (env.grid_size,
                 env.grid_size)),
            cmap="Greys",
            origin="lower")
        ax.add_collection(lc)
        ax.set_title(f"Map {i}")
        ax.axis("off")

        if save:
            plt.savefig(f"benchmark_visuals/map_{i:02d}.png")

    for j in range(num_maps, len(axs)):
        axs[j].axis("off")

    plt.tight_layout()
    plt.suptitle("Reward-Weighted Agent Paths", fontsize=16, y=1.02)
    plt.show()


def export_benchmark_maps(
    env: GridWorldICM,
    num_train: int = 20,
    num_test: int = 10,
    num_ood: int = 10,
    train_folder: str = "train_maps/",
    test_folder: str = "test_maps/",
    ood_folder: str = "ood_maps/",
) -> None:
    """Generate benchmark maps for training, testing, and OOD evaluation.

    Parameters
    ----------
    env : GridWorldICM
        Environment instance used to generate the maps.
    num_train : int, optional
        Number of training maps to create. Defaults to ``20``.
    num_test : int, optional
        Number of test maps to create. Defaults to ``10``.
    num_ood : int, optional
        Number of OOD maps to create. Defaults to ``10``.
    train_folder : str, optional
        Directory where training maps are saved.
    test_folder : str, optional
        Directory where test maps are saved.
    ood_folder : str, optional
        Directory where OOD maps are saved.
    """

    os.makedirs(train_folder, exist_ok=True)
    for i in range(num_train):
        env.reset(seed=i)
        env.save_map(os.path.join(train_folder, f"map_{i:02d}.npz"))

    os.makedirs(test_folder, exist_ok=True)
    for i in range(num_test):
        env.reset(seed=num_train + i)
        env.save_map(os.path.join(test_folder, f"map_{i:02d}.npz"))

    os.makedirs(ood_folder, exist_ok=True)
    # preserve original parameters
    orig_mine_density = env.mine_density
    orig_mine_density_range = env.mine_density_range
    orig_enemy_speed = env.enemy_speed
    orig_enemy_speed_range = env.enemy_speed_range
    orig_enemy_policy = env.enemy_policy
    orig_enemy_policy_options = env.enemy_policy_options

    for i in range(num_ood):
        if i % 2 == 0:
            env.mine_density = 0.1
            env.mine_density_range = None
            env.enemy_speed = 2
            env.enemy_speed_range = None
            env.enemy_policy = "aggressive"
            env.enemy_policy_options = None
        else:
            env.mine_density = 0.02
            env.mine_density_range = None
            env.enemy_speed = 1
            env.enemy_speed_range = None
            env.enemy_policy = "stationary"
            env.enemy_policy_options = None
        env.reset(seed=num_train + num_test + i)
        env.save_map(os.path.join(ood_folder, f"map_{i:02d}.npz"))

    env.mine_density = orig_mine_density
    env.mine_density_range = orig_mine_density_range
    env.enemy_speed = orig_enemy_speed
    env.enemy_speed_range = orig_enemy_speed_range
    env.enemy_policy = orig_enemy_policy
    env.enemy_policy_options = orig_enemy_policy_options


def evaluate_on_benchmarks(
    env: GridWorldICM,
    policy,
    map_folder: str = "maps/",
    num_maps: int = 10,
    H: int = 8,
    ood_map_folder: str | None = None,
    num_ood_maps: int = 0,
) -> Tuple[float, float] | Tuple[Tuple[float, float], Tuple[float, float]]:
    import torch
    def _evaluate(folder: str, count: int) -> Tuple[float, float]:
        rewards = []
        for i in range(count):
            map_path = f"{folder}/map_{i:02d}.npz"
            obs, _ = env.reset(load_map_path=map_path)
            planner = SymbolicPlanner(env.cost_map, env.risk_map, env.np_random)
            g = planner.get_subgoal(env.agent_pos, H)
            subgoal_timer = H
            dx, dy = g[0] - env.agent_pos[0], g[1] - env.agent_pos[1]
            obs = np.concatenate([obs, np.array([dx, dy], dtype=np.float32)])
            done = False
            total_reward = 0.0
            while not done:
                state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                action, _, _, _ = policy.act(state_tensor)
                obs_base, reward, cost_t, done, _, _ = env.step(action)
                subgoal_timer -= 1
                if subgoal_timer <= 0 or tuple(env.agent_pos) == g:
                    g = planner.get_subgoal(env.agent_pos, H)
                    subgoal_timer = H
                dx, dy = g[0] - env.agent_pos[0], g[1] - env.agent_pos[1]
                obs = np.concatenate([obs_base, np.array([dx, dy], dtype=np.float32)])
                total_reward += reward
            rewards.append(total_reward)
        arr = np.asarray(rewards)
        mean = float(arr.mean())
        if len(arr) < 2:
            ci = 0.0
        else:
            ci = float(stats.sem(arr) * stats.t.ppf(0.975, len(arr) - 1))
        return mean, ci

    id_result = _evaluate(map_folder, num_maps)
    if ood_map_folder and num_ood_maps > 0:
        ood_result = _evaluate(ood_map_folder, num_ood_maps)
        return id_result, ood_result
    return id_result


def plot_model_performance(
    models: List,
    model_names: List[str],
    env: GridWorldICM,
    map_folder: str = "maps/",
    num_maps: int = 10,
    ood_map_folder: str | None = None,
    num_ood_maps: int = 0,
):
    avg_rewards: List[float] = []
    ci_vals: List[float] = []
    avg_rewards_ood: List[float] = []
    ci_vals_ood: List[float] = []

    for policy in models:
        if ood_map_folder and num_ood_maps > 0:
            id_res, ood_res = evaluate_on_benchmarks(
                env,
                policy,
                map_folder,
                num_maps,
                ood_map_folder=ood_map_folder,
                num_ood_maps=num_ood_maps,
            )
            mean_r, ci_r = id_res
            mean_o, ci_o = ood_res
            avg_rewards.append(mean_r)
            ci_vals.append(ci_r)
            avg_rewards_ood.append(mean_o)
            ci_vals_ood.append(ci_o)
        else:
            mean_r, ci_r = evaluate_on_benchmarks(
                env, policy, map_folder, num_maps)
            avg_rewards.append(mean_r)
            ci_vals.append(ci_r)

    x = np.arange(len(model_names))
    plt.figure(figsize=(8, 5))
    if ood_map_folder and num_ood_maps > 0:
        width = 0.35
        plt.bar(
            x - width / 2,
            avg_rewards,
            width,
            yerr=ci_vals,
            capsize=5,
            label="ID",
        )
        plt.bar(
            x + width / 2,
            avg_rewards_ood,
            width,
            yerr=ci_vals_ood,
            capsize=5,
            label="OOD",
            color="orange",
        )
        plt.legend()
    else:
        plt.bar(
            x,
            avg_rewards,
            yerr=ci_vals,
            capsize=5,
            color=["gray", "skyblue", "green"],
        )
    plt.xticks(x, model_names)
    plt.ylabel("Average Reward")
    plt.title("Model Performance on Benchmark Maps")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.show()
