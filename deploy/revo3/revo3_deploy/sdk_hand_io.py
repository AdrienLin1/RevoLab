from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Sequence
from typing import Any

import numpy as np

from revo3_deploy.robot_profile import JOINT_DIM

DEG_TO_RAD = math.pi / 180.0
RAD_TO_DEG = 180.0 / math.pi
RPM_TO_RAD_S = 2.0 * math.pi / 60.0
RAD_S_TO_RPM = 60.0 / (2.0 * math.pi)

REVO3_TOUCH_MODULES = {
    0: ("palm", 36),
    1: ("thumb_tip", 31),
    2: ("thumb_pad", 57),
    3: ("index_tip", 21),
    4: ("index_pad", 52),
    5: ("middle_tip", 21),
    6: ("middle_pad", 52),
    7: ("ring_tip", 21),
    8: ("ring_pad", 52),
    9: ("little_tip", 21),
    10: ("little_pad", 52),
}
# These counts are the current Revo3 pressure-module hardware contract. Runtime
# reads validate every requested array so a different sensor vendor/layout fails
# closed instead of silently changing the policy input width.
REVO3_FINGERTIP_MODULE_IDS = (1, 3, 5, 7, 9)


def _load_sdk():
    try:
        from bc_stark_sdk import main_mod as sdk
    except ImportError:
        try:
            from bc_stark_sdk import bc_stark_sdk as sdk
        except ImportError as exc:
            raise RuntimeError(
                "bc-stark-sdk is not installed. Install the hardware extra with "
                '`pip install -e ".[hardware]"` from deploy/revo3, or install the '
                "bc-stark-sdk wheel provided for your platform."
            ) from exc
    return sdk


@dataclass
class Revo3SdkConfig:
    port: str | None = None
    baudrate: int = 5000000
    slave_id: int = 126
    auto_detect: bool = True


class Revo3SdkHandIO:
    """Thin async adapter around bc-stark-sdk using radians internally."""

    def __init__(self, config: Revo3SdkConfig) -> None:
        self.config = config
        self.sdk = _load_sdk()
        self.ctx: Any | None = None
        self.slave_id = int(config.slave_id)

    async def open(self) -> None:
        self.sdk.init_logging()
        port = self.config.port
        baudrate = self.config.baudrate
        slave_id = self.config.slave_id

        if self.config.auto_detect and port is None:
            _, port, baudrate, slave_id = await self.sdk.auto_detect_modbus_revo3()
        if port is None:
            raise ValueError("A serial port is required when Revo3 auto-detect is disabled.")

        self.slave_id = int(slave_id)
        self.ctx = await self.sdk.modbus_open(port, self._baudrate_enum(int(baudrate)))

    @property
    def sdk_version(self) -> str:
        """Return the loaded bc-stark-sdk version string."""

        return str(self.sdk.get_sdk_version())

    def close(self) -> None:
        if self.ctx is not None:
            self.sdk.modbus_close(self.ctx)
            self.ctx = None

    async def read_position_rad(self) -> np.ndarray:
        status = await self._ctx.v3_get_motor_status_data(self.slave_id)
        return self._status_vector(status.positions, "positions") * DEG_TO_RAD

    async def read_state_rad(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        status = await self._ctx.v3_get_motor_status_data(self.slave_id)
        pos = self._status_vector(status.positions, "positions") * DEG_TO_RAD
        vel = self._status_vector(status.velocities, "velocities") * RPM_TO_RAD_S
        cur = self._status_vector(status.currents, "currents")
        return pos, vel, cur

    async def prepare_touch(
        self,
        module_ids: Sequence[int],
        data_type: int | None = None,
        enable_modules: bool = False,
        reset_pressure: bool = False,
    ) -> int:
        """Validate and optionally configure the touch modules used by a policy.

        ``bc-stark-sdk==1.4.5`` documents touch data type ``0`` as AD values and
        ``1`` as calibrated values. The method does not reset pressure unless the
        caller explicitly requests it because resetting changes the hardware
        baseline.

        Args:
            module_ids: Revo3 module IDs required by the policy.
            data_type: Optional SDK data type written before reading.
            enable_modules: Whether to enable required modules automatically.
            reset_pressure: Whether to clear required modules' pressure baselines.

        Returns:
            The 11-bit enabled-module mask reported by the hand.
        """

        module_ids = self._touch_module_ids(module_ids)
        await self._ctx.get_device_info(self.slave_id)
        if data_type is not None:
            if int(data_type) not in (0, 1):
                raise ValueError("bc-stark-sdk 1.4.5 touch data_type must be 0 or 1.")
            await self._ctx.v3_set_touch_data_type(self.slave_id, int(data_type))

        required_mask = sum(1 << module_id for module_id in module_ids)
        enabled_mask = int(await self._ctx.v3_get_all_touch_modules_enabled(self.slave_id))
        missing_mask = required_mask & ~enabled_mask
        if missing_mask and enable_modules:
            enabled_mask |= required_mask
            await self._ctx.v3_set_all_touch_modules_enabled(self.slave_id, enabled_mask)
            enabled_mask = int(
                await self._ctx.v3_get_all_touch_modules_enabled(self.slave_id)
            )
            missing_mask = required_mask & ~enabled_mask
        if missing_mask:
            missing = [module_id for module_id in module_ids if missing_mask & (1 << module_id)]
            raise RuntimeError(
                f"Required Revo3 touch modules are disabled: {missing}. "
                "Enable them on the hand or pass --enable-touch-modules."
            )
        if reset_pressure:
            for module_id in module_ids:
                await self._ctx.v3_reset_touch_pressure(self.slave_id, module_id)
        return enabled_mask

    async def read_touch_module(self, module_id: int) -> np.ndarray:
        """Read one physical touch module in SDK-returned channel order.

        Args:
            module_id: Revo3 physical module ID in ``0..10``.

        Returns:
            One float32 vector in the order returned by bc-stark-sdk. This order is
            not assumed to equal the sensor IDs printed in the layout diagram.
        """

        module_id = self._touch_module_ids((module_id,))[0]
        values = np.asarray(
            await self._ctx.v3_get_touch_module_data(self.slave_id, module_id),
            dtype=np.float32,
        ).reshape(-1)
        expected_count = REVO3_TOUCH_MODULES[module_id][1]
        if values.shape != (expected_count,):
            raise RuntimeError(
                f"Touch module {module_id} returned {values.shape[0]} values; "
                f"bc-stark-sdk 1.4.5 Pressure layout expects {expected_count}. "
                "The connected hand may use a different touch vendor/layout."
            )
        if not np.isfinite(values).all():
            raise RuntimeError(f"Touch module {module_id} contains non-finite values.")
        return values

    async def read_touch_modules(
        self, module_ids: Sequence[int]
    ) -> dict[int, np.ndarray]:
        """Read one bulk touch snapshot and select the requested modules.

        Args:
            module_ids: Physical module IDs required for the current policy.

        Returns:
            Mapping from module ID to its per-sensor values.
        """

        module_ids = self._touch_module_ids(module_ids)
        _, modules = await self.read_all_touch_data(module_ids)
        return {module_id: modules[module_id].copy() for module_id in module_ids}

    async def read_all_touch_data(
        self, validate_module_ids: Sequence[int] | None = None
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
        """Read the SDK bulk touch structure for diagnostics.

        Args:
            validate_module_ids: Modules whose array sizes must match the current
                pressure layout. By default all 11 modules are validated.

        Returns:
            The 16-value summary and all 11 module arrays in SDK module order.
        """

        data = await self._ctx.v3_get_all_touch_data(self.slave_id)
        summary = np.asarray(data.summary, dtype=np.float32).reshape(-1)
        modules = tuple(
            np.asarray(values, dtype=np.float32).reshape(-1) for values in data.modules
        )
        if summary.shape != (16,) or len(modules) != len(REVO3_TOUCH_MODULES):
            raise RuntimeError(
                f"Unexpected V3TouchData shape: summary={summary.shape}, modules={len(modules)}."
            )
        if not np.isfinite(summary).all():
            raise RuntimeError("V3TouchData summary contains non-finite values.")
        if validate_module_ids is None:
            module_ids = tuple(REVO3_TOUCH_MODULES)
        else:
            module_ids = self._touch_module_ids(validate_module_ids)
        for module_id in module_ids:
            values = modules[module_id]
            expected_count = REVO3_TOUCH_MODULES[module_id][1]
            if values.shape != (expected_count,):
                raise RuntimeError(
                    f"Touch module {module_id} returned {values.shape[0]} values; "
                    f"expected {expected_count}."
                )
            if not np.isfinite(values).all():
                raise RuntimeError(f"Touch module {module_id} contains non-finite values.")
        return summary, modules

    async def read_position_and_touch_rad(
        self, module_ids: Sequence[int]
    ) -> tuple[np.ndarray, dict[int, np.ndarray]]:
        """Read measured joints followed by the selected tactile arrays.

        Args:
            module_ids: Physical touch module IDs required by the policy.

        Returns:
            SDK-order joint positions in radians and touch arrays by module ID.
        """

        positions = await self.read_position_rad()
        touch_modules = await self.read_touch_modules(module_ids)
        return positions, touch_modules

    async def send_mit_command_rad(
        self,
        position_rad: np.ndarray,
        velocity_rad_s: np.ndarray | None = None,
        kp: float | list[float] | np.ndarray = 1.0,
        kd: float | list[float] | np.ndarray = 0.1,
        effort_ma: float | list[float] | np.ndarray = 0.0,
    ) -> None:
        pos_deg = self._vector(position_rad, "position_rad") * RAD_TO_DEG
        vel_rpm = self._vector(
            np.zeros(JOINT_DIM, dtype=np.float32) if velocity_rad_s is None else velocity_rad_s,
            "velocity_rad_s",
        ) * RAD_S_TO_RPM
        kp_values = self._command_values(kp, "kp")
        kd_values = self._command_values(kd, "kd")
        if any(value < 0.0 or value > 10.0 for value in kp_values):
            raise ValueError("kp must be within the bc-stark-sdk MIT range [0, 10].")
        if any(value < 0.0 or value > 10.0 for value in kd_values):
            raise ValueError("kd must be within the bc-stark-sdk MIT range [0, 10].")
        await self._ctx.revo3_multi_mit_set_all(
            self.slave_id,
            kp_values,
            kd_values,
            pos_deg.tolist(),
            vel_rpm.tolist(),
            self._command_values(effort_ma, "effort_ma"),
        )

    @property
    def _ctx(self):
        if self.ctx is None:
            raise RuntimeError("Revo3SdkHandIO is not open.")
        return self.ctx

    def _baudrate_enum(self, value: int):
        baudrate_type = self.sdk.Baudrate
        if hasattr(baudrate_type, "from_int"):
            return baudrate_type.from_int(value)
        mapping = {
            115200: baudrate_type.Baud115200,
            57600: baudrate_type.Baud57600,
            19200: baudrate_type.Baud19200,
            460800: baudrate_type.Baud460800,
            1000000: baudrate_type.Baud1Mbps,
            2000000: baudrate_type.Baud2Mbps,
            5000000: baudrate_type.Baud5Mbps,
        }
        if value not in mapping:
            raise ValueError(
                f"Unsupported Modbus baudrate {value}; expected one of {sorted(mapping)}."
            )
        return mapping[value]

    @staticmethod
    def _vector(value, name: str) -> np.ndarray:
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
        if vector.shape != (JOINT_DIM,):
            raise ValueError(f"{name} must have {JOINT_DIM} values.")
        if not np.isfinite(vector).all():
            raise ValueError(f"{name} contains non-finite values.")
        return vector

    @staticmethod
    def _status_vector(value, name: str) -> np.ndarray:
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
        if vector.shape != (JOINT_DIM,):
            raise RuntimeError(
                f"bc-stark-sdk V3MotorStatusData.{name} must have {JOINT_DIM} values, "
                f"got {vector.shape}."
            )
        if not np.isfinite(vector).all():
            raise RuntimeError(f"V3MotorStatusData.{name} contains non-finite values.")
        return vector

    @staticmethod
    def _command_values(value, name: str) -> list[float]:
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
        if not np.isfinite(vector).all():
            raise ValueError(f"{name} contains non-finite values.")
        if vector.shape == (1,):
            return [float(vector[0])] * JOINT_DIM
        if vector.shape != (JOINT_DIM,):
            raise ValueError(f"{name} must be scalar or {JOINT_DIM} values.")
        return [float(v) for v in vector]

    @staticmethod
    def _touch_module_ids(module_ids: Sequence[int]) -> tuple[int, ...]:
        values = tuple(int(value) for value in module_ids)
        if not values:
            raise ValueError("At least one touch module ID is required.")
        if len(set(values)) != len(values):
            raise ValueError("Touch module IDs contain duplicates.")
        invalid = [value for value in values if value not in REVO3_TOUCH_MODULES]
        if invalid:
            raise ValueError(f"Touch module IDs must be in 0..10, got {invalid}.")
        return values
