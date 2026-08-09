"""Run a physical-taxel TactileDAgger ONNX policy on a Revo3 hand.

Overview:
The control loop reads 21 motor angles and the policy-selected fingertip touch
modules, builds training-compatible temporal observations, runs ONNX inference,
and sends delta-action MIT targets. Use ``--dry-run`` before enabling commands.

Quick Start:
    python scripts/run_tactile_policy.py --task nutbolt_tactile \
        --onnx policy.onnx --policy policy.yaml --dry-run

Full Command:
    python scripts/run_tactile_policy.py --task TASK --onnx policy.onnx \
        --policy policy.yaml --profile config/revo3_right.yaml --port /dev/ttyUSB0 \
        --slave-id 126 --touch-data-type 1

Options:
    --task: Task name/alias used to validate observation and action semantics.
    --touch-threshold-on: Contact establishment threshold in SDK sensor units.
    --touch-threshold-off: Lower contact release threshold for hysteresis.
    --sensor-log-every: Print SDK channel values every N control frames.

Notes:
    SDK touch readings are device values rather than physical force units. For
    calibrated data type 1, the default hysteresis is on=150 and off=100.
    Simulation thresholds are not valid for SDK values. Pressure reset is opt-in.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

import numpy as np
import yaml

from revo3_deploy.robot_profile import Revo3Profile
from revo3_deploy.sdk_hand_io import Revo3SdkConfig, Revo3SdkHandIO, REVO3_TOUCH_MODULES
from revo3_deploy.tactile_policy_runner import Revo3TactilePolicyRunner

DEFAULT_TOUCH_THRESHOLD_ON = 150.0
DEFAULT_TOUCH_THRESHOLD_OFF = 100.0


def build_parser() -> argparse.ArgumentParser:
    """Build the tactile deployment command-line parser.

    Returns:
        Parser containing model, hardware, tactile, and MIT control options.
    """

    parser = argparse.ArgumentParser(
        description="Run a Revo3 TactileDAgger ONNX policy through bc-stark-sdk 1.4.5."
    )
    parser.add_argument(
        "--task",
        required=True,
        help=(
            "Tactile task: rotate_ball_tactile, rotate_cylinder_tactile, "
            "nutbolt_tactile, screwdriver_tactile, valvedriver_tactile, "
            "or valvedriver_tactile_40."
        ),
    )
    parser.add_argument("--onnx", required=True, help="Path to tactile student ONNX.")
    parser.add_argument("--policy", required=True, help="Path to tactile policy YAML.")
    parser.add_argument("--profile", default="config/revo3_right.yaml")
    parser.add_argument("--port", default=None, help="Serial port. Omit for SDK auto-detect.")
    parser.add_argument("--baudrate", type=int, default=None)
    parser.add_argument("--slave-id", type=int, default=None)
    parser.add_argument("--rate", type=float, default=None)
    parser.add_argument("--kp", type=float, default=None)
    parser.add_argument("--kd", type=float, default=None)
    parser.add_argument("--effort-ma", type=float, default=None)
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--touch-data-type", type=int, choices=(0, 1), default=1)
    parser.add_argument(
        "--touch-threshold-on",
        type=float,
        default=None,
        help="SDK contact-on value; defaults to 150 for --touch-data-type 1.",
    )
    parser.add_argument(
        "--touch-threshold-off",
        type=float,
        default=None,
        help="SDK contact-off value; defaults to 100 for --touch-data-type 1.",
    )
    parser.add_argument(
        "--touch-thresholds",
        default="",
        help="Per-sensor threshold YAML in diagram sensor-ID order.",
    )
    parser.add_argument("--enable-touch-modules", action="store_true")
    parser.add_argument("--reset-touch-pressure", action="store_true")
    parser.add_argument(
        "--allow-unverified-sensor-map",
        action="store_true",
        help="Allow motor commands with the unverified identity SDK-channel mapping.",
    )
    parser.add_argument(
        "--sensor-log-every",
        type=int,
        default=0,
        help="Print every 1-based SDK channel every N frames; 0 disables.",
    )
    return parser


async def async_main(args: argparse.Namespace) -> int:
    """Run the asynchronous tactile closed-loop policy.

    Args:
        args: Parsed deployment options.

    Returns:
        Process exit status.
    """

    profile = Revo3Profile.load(args.profile)
    threshold_on, threshold_off = _resolve_contact_thresholds(args)
    runner = Revo3TactilePolicyRunner(
        args.onnx,
        args.policy,
        profile,
        task=args.task,
        contact_threshold_on=threshold_on,
        contact_threshold_off=threshold_off,
        use_gpu=args.use_gpu,
    )
    if (
        not args.dry_run
        and not runner.sensor_mapping_verified
        and not args.allow_unverified_sensor_map
    ):
        raise RuntimeError(
            "The policy sensor map is not hardware-verified. Run with --dry-run, export "
            "with --sensor-map, or explicitly pass --allow-unverified-sensor-map."
        )

    sdk_cfg = profile.sdk
    io = Revo3SdkHandIO(
        Revo3SdkConfig(
            port=args.port,
            baudrate=int(args.baudrate or sdk_cfg.get("baudrate", 5000000)),
            slave_id=int(args.slave_id or sdk_cfg.get("slave_id", 126)),
            auto_detect=args.port is None and bool(sdk_cfg.get("auto_detect", True)),
        )
    )
    mit = profile.mit
    kp = float(args.kp if args.kp is not None else mit.get("kp", 1.0))
    kd = float(args.kd if args.kd is not None else mit.get("kd", 0.1))
    effort_ma = float(
        args.effort_ma if args.effort_ma is not None else mit.get("effort_ma", 0.0)
    )
    rate_hz = float(args.rate or runner.rate_hz)
    if rate_hz <= 0.0:
        raise ValueError("Policy rate must be positive.")
    if args.sensor_log_every < 0:
        raise ValueError("--sensor-log-every must be non-negative.")
    period = 1.0 / rate_hz

    await io.open()
    try:
        enabled_mask = await io.prepare_touch(
            runner.module_ids,
            data_type=args.touch_data_type,
            enable_modules=args.enable_touch_modules,
            reset_pressure=args.reset_touch_pressure,
        )
        print(
            f"Connected Revo3 slave_id={io.slave_id}; sdk={io.sdk_version}; "
            f"task={runner.task_name}; policy rate={rate_hz:.2f} Hz; "
            f"touch modules={runner.module_ids}; sensor_map_verified="
            f"{runner.sensor_mapping_verified}; enabled_mask=0x{enabled_mask:03X}"
        )
        next_tick = time.monotonic()
        frame_index = 0
        while True:
            sdk_pos, touch_modules = await io.read_position_and_touch_rad(runner.module_ids)
            policy_pos = profile.measured_sdk_to_policy(sdk_pos)
            policy_target = runner.step(policy_pos, touch_modules)
            sdk_target = profile.target_policy_to_sdk(policy_target)

            if args.sensor_log_every > 0 and frame_index % args.sensor_log_every == 0:
                _print_sensor_frame(frame_index, touch_modules)
            if args.dry_run:
                print(
                    f"frame={frame_index} target_rad="
                    f"{np.array2string(sdk_target, precision=3, suppress_small=True)}"
                )
            else:
                await io.send_mit_command_rad(
                    sdk_target, kp=kp, kd=kd, effort_ma=effort_ma
                )

            frame_index += 1
            next_tick += period
            sleep_time = next_tick - time.monotonic()
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            else:
                next_tick = time.monotonic()
    except KeyboardInterrupt:
        return 130
    finally:
        io.close()


def _print_sensor_frame(
    frame_index: int, touch_modules: dict[int, np.ndarray]
) -> None:
    """Print one readable module snapshot in SDK-returned channel order.

    Args:
        frame_index: Control frame number.
        touch_modules: SDK arrays keyed by physical module ID.
    """

    print(f"touch_frame={frame_index}")
    for module_id, values in touch_modules.items():
        location = REVO3_TOUCH_MODULES[module_id][0]
        pairs = " ".join(
            f"sdk_channel_{channel_id}={float(value):.6g}"
            for channel_id, value in enumerate(values, start=1)
        )
        print(f"  module_{module_id} {location}: {pairs}")


def _resolve_contact_thresholds(
    args: argparse.Namespace,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Load scalar or per-diagram-sensor contact hysteresis thresholds."""

    scalar_supplied = (
        args.touch_threshold_on is not None or args.touch_threshold_off is not None
    )
    if args.touch_thresholds and scalar_supplied:
        raise ValueError(
            "Use either --touch-thresholds or scalar --touch-threshold-on/off, not both."
        )
    if not args.touch_thresholds:
        if args.touch_threshold_on is None and args.touch_threshold_off is None:
            if int(args.touch_data_type) != 1:
                raise ValueError(
                    "Default touch thresholds 150/100 apply only to touch_data_type=1; "
                    "provide explicit thresholds for other SDK value types."
                )
            return DEFAULT_TOUCH_THRESHOLD_ON, DEFAULT_TOUCH_THRESHOLD_OFF
        if args.touch_threshold_on is None or args.touch_threshold_off is None:
            raise ValueError(
                "Provide both --touch-threshold-on and --touch-threshold-off."
            )
        return float(args.touch_threshold_on), float(args.touch_threshold_off)

    threshold_path = Path(args.touch_thresholds).expanduser().resolve()
    with threshold_path.open("r", encoding="utf-8") as stream:
        threshold_cfg = yaml.safe_load(stream) or {}
    with Path(args.policy).expanduser().resolve().open("r", encoding="utf-8") as stream:
        policy_cfg = yaml.safe_load(stream) or {}
    tactile_cfg = policy_cfg.get("tactile") or {}
    if threshold_cfg.get("layout_version") != tactile_cfg.get("layout_version"):
        raise ValueError("Threshold YAML layout_version differs from policy.yaml.")
    if int(threshold_cfg.get("touch_data_type", -1)) != int(args.touch_data_type):
        raise ValueError("Threshold YAML touch_data_type differs from --touch-data-type.")
    threshold_fingers = threshold_cfg.get("fingers")
    if not isinstance(threshold_fingers, dict):
        raise ValueError("Threshold YAML must contain a fingers mapping.")

    threshold_on = []
    threshold_off = []
    for finger in tactile_cfg.get("fingers") or ():
        name = str(finger["name"])
        count = int(finger["sensor_count"])
        values = threshold_fingers.get(name)
        if not isinstance(values, dict):
            raise ValueError(f"Threshold YAML is missing active finger {name!r}.")
        if int(values.get("module_id", -1)) != int(finger["module_id"]):
            raise ValueError(f"Threshold YAML module_id is wrong for {name}.")
        try:
            on_values = [
                float(value)
                for value in values.get("threshold_on_by_official_sensor_id") or ()
            ]
            off_values = [
                float(value)
                for value in values.get("threshold_off_by_official_sensor_id") or ()
            ]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Threshold YAML for {name} contains a non-numeric value."
            ) from exc
        if len(on_values) != count or len(off_values) != count:
            raise ValueError(
                f"Threshold YAML for {name} must contain {count} on/off values."
            )
        threshold_on.extend(on_values)
        threshold_off.extend(off_values)
    return (
        np.asarray(threshold_on, dtype=np.float32),
        np.asarray(threshold_off, dtype=np.float32),
    )


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and start the asynchronous deployment loop.

    Args:
        argv: Optional argument vector used by tests or wrappers.

    Returns:
        Process exit status.
    """

    return asyncio.run(async_main(build_parser().parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
