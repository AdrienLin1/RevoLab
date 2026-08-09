import torch
from torch.utils.data import Dataset


def transform_op(arr):
    # swap axes 0↔1 then flatten
    if arr is None:
        return arr
    s = arr.size()
    return arr.transpose(0, 1).reshape(s[0] * s[1], *s[2:])


class ExperienceBuffer(Dataset):
    def __init__(
        self,
        num_envs,
        horizon_length,
        batch_size,
        minibatch_size,
        obs_dim,
        act_dim,
        priv_dim,
        device,
        tactile_hist_shape=None,
        separate_coord_advantage=False,
    ):
        self.device = device
        self.num_envs = num_envs
        self.transitions_per_env = horizon_length
        self.priv_info_dim = priv_dim
        self.tactile_hist_shape = tactile_hist_shape
        self.separate_coord_advantage = bool(separate_coord_advantage)
        self.rollout_stats = {}

        self.data_dict = None
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.priv_dim = priv_dim
        self.storage_dict = {
            'obses': torch.zeros((self.transitions_per_env, self.num_envs, self.obs_dim), dtype=torch.float32, device=self.device),
            'priv_info': torch.zeros((self.transitions_per_env, self.num_envs, self.priv_dim), dtype=torch.float32, device=self.device),
            'rewards': torch.zeros((self.transitions_per_env, self.num_envs, 1), dtype=torch.float32, device=self.device),
            'values': torch.zeros((self.transitions_per_env, self.num_envs,  1), dtype=torch.float32, device=self.device),
            'neglogpacs': torch.zeros((self.transitions_per_env, self.num_envs), dtype=torch.float32, device=self.device),
            'dones': torch.zeros((self.transitions_per_env, self.num_envs), dtype=torch.uint8, device=self.device),
            'actions': torch.zeros((self.transitions_per_env, self.num_envs, self.act_dim), dtype=torch.float32, device=self.device),
            'mus': torch.zeros((self.transitions_per_env, self.num_envs, self.act_dim), dtype=torch.float32, device=self.device),
            'sigmas': torch.zeros((self.transitions_per_env, self.num_envs, self.act_dim), dtype=torch.float32, device=self.device),
            'returns': torch.zeros((self.transitions_per_env, self.num_envs,  1), dtype=torch.float32, device=self.device),
        }
        if self.separate_coord_advantage:
            self.storage_dict.update(
                {
                    'coord_rewards': torch.zeros(
                        (self.transitions_per_env, self.num_envs, 1),
                        dtype=torch.float32,
                        device=self.device,
                    ),
                    'coord_values': torch.zeros(
                        (self.transitions_per_env, self.num_envs, 1),
                        dtype=torch.float32,
                        device=self.device,
                    ),
                    'coord_returns': torch.zeros(
                        (self.transitions_per_env, self.num_envs, 1),
                        dtype=torch.float32,
                        device=self.device,
                    ),
                }
            )
        if tactile_hist_shape is not None:
            self.storage_dict['tactile_hist'] = torch.zeros(
                (self.transitions_per_env, self.num_envs, *tactile_hist_shape),
                dtype=torch.float32,
                device=self.device,
            )

        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.length = self.batch_size // self.minibatch_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        start = idx * self.minibatch_size
        end = (idx + 1) * self.minibatch_size
        self.last_range = (start, end)
        input_dict = {}
        for k, v in self.data_dict.items():
            if type(v) is dict:
                v_dict = {kd: vd[start:end] for kd, vd in v.items()}
                input_dict[k] = v_dict
            else:
                input_dict[k] = v[start:end]
        return input_dict

    def update_mu_sigma(self, mu, sigma):
        start = self.last_range[0]
        end = self.last_range[1]
        self.data_dict['mus'][start:end] = mu
        self.data_dict['sigmas'][start:end] = sigma

    def update_data(self, name, index, val):
        if type(val) is dict:
            for k, v in val.items():
                self.storage_dict[name][k][index,:] = v
        else:
            self.storage_dict[name][index,:] = val

    def _compute_return(self, reward_key, value_key, return_key, last_values, gamma, tau):
        """Compute GAE returns for one reward and value stream.

        Args:
            reward_key: Storage key containing per-step rewards.
            value_key: Storage key containing per-step value predictions.
            return_key: Storage key receiving bootstrapped returns.
            last_values: Value prediction after the final rollout transition.
            gamma: Reward discount factor.
            tau: GAE trace decay.
        """
        last_gae_lam = 0
        mb_advs = torch.zeros_like(self.storage_dict[reward_key])
        for t in reversed(range(self.transitions_per_env)):
            if t == self.transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.storage_dict[value_key][t + 1]
            next_nonterminal = 1.0 - self.storage_dict['dones'].float()[t]
            next_nonterminal = next_nonterminal.unsqueeze(1)
            delta = (
                self.storage_dict[reward_key][t]
                + gamma * next_values * next_nonterminal
                - self.storage_dict[value_key][t]
            )
            mb_advs[t] = last_gae_lam = delta + gamma * tau * next_nonterminal * last_gae_lam
            self.storage_dict[return_key][t, :] = (
                mb_advs[t] + self.storage_dict[value_key][t]
            )

    def computer_return(self, last_values, gamma, tau, coord_last_values=None):
        """Compute task and optional coordination GAE returns.

        Args:
            last_values: Task-value prediction after the rollout.
            gamma: Reward discount factor.
            tau: GAE trace decay.
            coord_last_values: Coordination-value prediction after the rollout.
        """
        self._compute_return('rewards', 'values', 'returns', last_values, gamma, tau)
        if self.separate_coord_advantage:
            if coord_last_values is None:
                raise ValueError(
                    "coord_last_values is required when separate_coord_advantage is enabled"
                )
            self._compute_return(
                'coord_rewards',
                'coord_values',
                'coord_returns',
                coord_last_values,
                gamma,
                tau,
            )

    @staticmethod
    def _normalize_advantage(advantage):
        """Standardize an advantage tensor without amplifying constant inputs.

        Args:
            advantage: Raw rollout advantage tensor.

        Returns:
            Zero-mean unit-variance advantages, or zeros for a constant tensor.
        """
        std = advantage.std(unbiased=False)
        if not torch.isfinite(std) or std <= 1.0e-8:
            return torch.zeros_like(advantage)
        return (advantage - advantage.mean()) / std

    def prepare_training(self, normalize_advantage=True, coord_advantage_coef=0.0):
        """Flatten rollout data and build separate task/coord advantages.

        Args:
            normalize_advantage: Whether to standardize each advantage stream.
            coord_advantage_coef: Coordination actor-loss coefficient for the rollout.

        Returns:
            Flattened training tensors keyed by storage field.
        """
        self.data_dict = {}
        for k, v in self.storage_dict.items():
            self.data_dict[k] = transform_op(v)
        task_raw = self.data_dict['returns'] - self.data_dict['values']
        task_advantages = (
            self._normalize_advantage(task_raw) if normalize_advantage else task_raw
        ).squeeze(1)
        self.data_dict['task_advantages'] = task_advantages
        self.data_dict['advantages'] = task_advantages
        self.rollout_stats = {
            'task_adv_raw_std': task_raw.std(unbiased=False).item(),
        }

        if self.separate_coord_advantage:
            coord_raw = self.data_dict['coord_returns'] - self.data_dict['coord_values']
            coord_advantages = (
                self._normalize_advantage(coord_raw)
                if normalize_advantage
                else coord_raw
            ).squeeze(1)
            self.data_dict['coord_advantages'] = coord_advantages
            self.data_dict['advantages'] = (
                task_advantages + float(coord_advantage_coef) * coord_advantages
            )
            self.rollout_stats.update(
                {
                    'coord_adv_raw_std': coord_raw.std(unbiased=False).item(),
                    'coord_reward_nonzero_ratio': (
                        self.data_dict['coord_rewards'].abs() > 1.0e-8
                    ).float().mean().item(),
                }
            )
        return self.data_dict
