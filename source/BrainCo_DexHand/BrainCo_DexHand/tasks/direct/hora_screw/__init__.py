# Copyright (c) 2026, BrainCo.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""HORA-style Revo3 screw manipulation tasks (ported from dexscrew)."""

import gymnasium as gym


gym.register(
    id="BrainCo-Direct-Revo3-HoraNutBolt-v0",
    entry_point=f"{__name__}.revo3_hand_screw_env:Revo3HandScrewEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.revo3_hand_screw_env_cfg:Revo3HandScrewNutBoltEnvCfg",
    },
)

gym.register(
    id="BrainCo-Direct-Revo3-HoraScrewDriver-v0",
    entry_point=f"{__name__}.revo3_hand_screw_env:Revo3HandScrewEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.revo3_hand_screw_env_cfg:Revo3HandScrewDriverEnvCfg",
    },
)

gym.register(
    id="BrainCo-Direct-Revo3-HoraVavleDriver-v0",
    entry_point=f"{__name__}.revo3_hand_screw_env:Revo3HandScrewEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.revo3_hand_screw_env_cfg:Revo3HandVavleDriverEnvCfg",
    },
)

gym.register(
    id="BrainCo-Direct-Revo3-HoraValveDriver-v0",
    entry_point=f"{__name__}.revo3_hand_screw_env:Revo3HandScrewEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.revo3_hand_screw_env_cfg:Revo3HandVavleDriverEnvCfg",
    },
)

gym.register(
    id="BrainCo-Direct-Revo3-HoraValveDriver25-v0",
    entry_point=f"{__name__}.revo3_hand_screw_env:Revo3HandScrewEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.revo3_hand_screw_env_cfg:Revo3HandValveDriver25EnvCfg",
    },
)

gym.register(
    id="BrainCo-Direct-Revo3-HoraValveDriver40-v0",
    entry_point=f"{__name__}.revo3_hand_screw_env:Revo3HandScrewEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.revo3_hand_screw_env_cfg:Revo3HandValveDriver40EnvCfg",
    },
)

# Tactile variants: base tasks + TacSL fingertip arrays in the teacher (priv) obs.
gym.register(
    id="BrainCo-Direct-Revo3-HoraNutBoltTactile-v0",
    entry_point=f"{__name__}.revo3_hand_screw_tactile_env:Revo3HandScrewTactileEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.revo3_hand_screw_tactile_env_cfg:Revo3HandScrewNutBoltTactileEnvCfg",
    },
)

gym.register(
    id="BrainCo-Direct-Revo3-HoraScrewDriverTactile-v0",
    entry_point=f"{__name__}.revo3_hand_screw_tactile_env:Revo3HandScrewTactileEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.revo3_hand_screw_tactile_env_cfg:Revo3HandScrewDriverTactileEnvCfg",
    },
)

gym.register(
    id="BrainCo-Direct-Revo3-HoraVavleDriverTactile-v0",
    entry_point=f"{__name__}.revo3_hand_screw_tactile_env:Revo3HandScrewTactileEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.revo3_hand_screw_tactile_env_cfg:Revo3HandVavleDriverTactileEnvCfg",
    },
)

gym.register(
    id="BrainCo-Direct-Revo3-HoraValveDriverTactile-v0",
    entry_point=f"{__name__}.revo3_hand_screw_tactile_env:Revo3HandScrewTactileEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.revo3_hand_screw_tactile_env_cfg:Revo3HandVavleDriverTactileEnvCfg",
    },
)

gym.register(
    id="BrainCo-Direct-Revo3-HoraValveDriverTactile25-v0",
    entry_point=f"{__name__}.revo3_hand_screw_tactile_env:Revo3HandScrewTactileEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.revo3_hand_screw_tactile_env_cfg:Revo3HandValveDriver25TactileEnvCfg",
    },
)

gym.register(
    id="BrainCo-Direct-Revo3-HoraValveDriverTactile40-v0",
    entry_point=f"{__name__}.revo3_hand_screw_tactile_env:Revo3HandScrewTactileEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.revo3_hand_screw_tactile_env_cfg:Revo3HandValveDriver40TactileEnvCfg",
    },
)

# Short aliases matching the requested task names.
gym.register(
    id="RevoHoraNutBolt-v0",
    entry_point=f"{__name__}.revo3_hand_screw_env:Revo3HandScrewEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.revo3_hand_screw_env_cfg:Revo3HandScrewNutBoltEnvCfg",
    },
)

gym.register(
    id="RevoHoraScrewDriver-v0",
    entry_point=f"{__name__}.revo3_hand_screw_env:Revo3HandScrewEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.revo3_hand_screw_env_cfg:Revo3HandScrewDriverEnvCfg",
    },
)

gym.register(
    id="RevoHoraVavleDriver-v0",
    entry_point=f"{__name__}.revo3_hand_screw_env:Revo3HandScrewEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.revo3_hand_screw_env_cfg:Revo3HandVavleDriverEnvCfg",
    },
)

gym.register(
    id="RevoHoraValveDriver-v0",
    entry_point=f"{__name__}.revo3_hand_screw_env:Revo3HandScrewEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.revo3_hand_screw_env_cfg:Revo3HandVavleDriverEnvCfg",
    },
)

gym.register(
    id="RevoHoraValveDriver25-v0",
    entry_point=f"{__name__}.revo3_hand_screw_env:Revo3HandScrewEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.revo3_hand_screw_env_cfg:Revo3HandValveDriver25EnvCfg",
    },
)

gym.register(
    id="RevoHoraValveDriver40-v0",
    entry_point=f"{__name__}.revo3_hand_screw_env:Revo3HandScrewEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.revo3_hand_screw_env_cfg:Revo3HandValveDriver40EnvCfg",
    },
)

gym.register(
    id="RevoHoraNutBoltTactile-v0",
    entry_point=f"{__name__}.revo3_hand_screw_tactile_env:Revo3HandScrewTactileEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.revo3_hand_screw_tactile_env_cfg:Revo3HandScrewNutBoltTactileEnvCfg",
    },
)

gym.register(
    id="RevoHoraScrewDriverTactile-v0",
    entry_point=f"{__name__}.revo3_hand_screw_tactile_env:Revo3HandScrewTactileEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.revo3_hand_screw_tactile_env_cfg:Revo3HandScrewDriverTactileEnvCfg",
    },
)

gym.register(
    id="RevoHoraVavleDriverTactile-v0",
    entry_point=f"{__name__}.revo3_hand_screw_tactile_env:Revo3HandScrewTactileEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.revo3_hand_screw_tactile_env_cfg:Revo3HandVavleDriverTactileEnvCfg",
    },
)

gym.register(
    id="RevoHoraValveDriverTactile-v0",
    entry_point=f"{__name__}.revo3_hand_screw_tactile_env:Revo3HandScrewTactileEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.revo3_hand_screw_tactile_env_cfg:Revo3HandVavleDriverTactileEnvCfg",
    },
)

gym.register(
    id="RevoHoraValveDriverTactile25-v0",
    entry_point=f"{__name__}.revo3_hand_screw_tactile_env:Revo3HandScrewTactileEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.revo3_hand_screw_tactile_env_cfg:Revo3HandValveDriver25TactileEnvCfg",
    },
)

gym.register(
    id="RevoHoraValveDriverTactile40-v0",
    entry_point=f"{__name__}.revo3_hand_screw_tactile_env:Revo3HandScrewTactileEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.revo3_hand_screw_tactile_env_cfg:Revo3HandValveDriver40TactileEnvCfg",
    },
)
