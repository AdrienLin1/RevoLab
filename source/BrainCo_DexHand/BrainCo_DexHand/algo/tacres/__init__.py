# Copyright (c) 2026, BrainCo.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""TacRes: contact-event-gated tactile residual policies (frozen base + residual + gate)."""

from .tacres_actor import TacResActor
from .tacres_ppo import TacResPPO

__all__ = ["TacResActor", "TacResPPO"]
