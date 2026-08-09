"""Read Revo3 joints and numbered fingertip sensors without loading a policy."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time

from revo3_deploy.robot_profile import Revo3Profile
from revo3_deploy.sdk_hand_io import (
    REVO3_FINGERTIP_MODULE_IDS,
    REVO3_TOUCH_MODULES,
    Revo3SdkConfig,
    Revo3SdkHandIO,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the real-hand state inspection parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Print all 21 measured joint angles and every numbered SDK channel value "
            "from selected Revo3 touch modules."
        )
    )
    parser.add_argument("--profile", default="config/revo3_right.yaml")
    parser.add_argument("--port", default=None)
    parser.add_argument("--baudrate", type=int, default=None)
    parser.add_argument("--slave-id", type=int, default=None)
    parser.add_argument("--rate", type=float, default=5.0)
    parser.add_argument(
        "--module-ids",
        default=",".join(str(value) for value in REVO3_FINGERTIP_MODULE_IDS),
        help="Comma-separated module IDs. Default is all five fingertip modules: 1,3,5,7,9.",
    )
    parser.add_argument("--touch-data-type", type=int, choices=(0, 1), default=1)
    parser.add_argument("--enable-touch-modules", action="store_true")
    parser.add_argument("--reset-touch-pressure", action="store_true")
    parser.add_argument("--frames", type=int, default=0, help="Frame count; 0 runs until Ctrl-C.")
    parser.add_argument("--format", choices=("text", "jsonl"), default="text")
    return parser


async def async_main(args: argparse.Namespace) -> int:
    """Open the hand and print timestamped joint/tactile snapshots."""

    if args.rate <= 0.0:
        raise ValueError("--rate must be positive.")
    if args.frames < 0:
        raise ValueError("--frames must be non-negative.")
    module_ids = _parse_module_ids(args.module_ids)
    profile = Revo3Profile.load(args.profile)
    sdk_cfg = profile.sdk
    io = Revo3SdkHandIO(
        Revo3SdkConfig(
            port=args.port,
            baudrate=int(args.baudrate or sdk_cfg.get("baudrate", 5000000)),
            slave_id=int(args.slave_id or sdk_cfg.get("slave_id", 126)),
            auto_detect=args.port is None and bool(sdk_cfg.get("auto_detect", True)),
        )
    )

    await io.open()
    try:
        enabled_mask = await io.prepare_touch(
            module_ids,
            data_type=args.touch_data_type,
            enable_modules=args.enable_touch_modules,
            reset_pressure=args.reset_touch_pressure,
        )
        if args.format == "text":
            print(
                f"Connected Revo3 slave_id={io.slave_id}; sdk={io.sdk_version}; "
                f"enabled_mask=0x{enabled_mask:03X}; modules={module_ids}"
            )
        period = 1.0 / float(args.rate)
        next_tick = time.monotonic()
        frame_index = 0
        while args.frames == 0 or frame_index < args.frames:
            sdk_pos_rad, touch_modules = await io.read_position_and_touch_rad(module_ids)
            record = _build_record(
                frame_index,
                profile.sdk_joint_order,
                sdk_pos_rad,
                touch_modules,
            )
            if args.format == "jsonl":
                print(json.dumps(record, separators=(",", ":"), ensure_ascii=True))
            else:
                _print_record(record)
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
    return 0


def _parse_module_ids(value: str) -> tuple[int, ...]:
    try:
        module_ids = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise ValueError("--module-ids must be a comma-separated integer list.") from exc
    return Revo3SdkHandIO._touch_module_ids(module_ids)


def _build_record(
    frame_index: int,
    sdk_joint_order: tuple[str, ...],
    sdk_pos_rad,
    touch_modules,
) -> dict:
    timestamp_ns = time.time_ns()
    joints = []
    for joint_id, (name, position_rad) in enumerate(zip(sdk_joint_order, sdk_pos_rad)):
        joints.append(
            {
                "joint_id": joint_id,
                "name": name,
                "rad": float(position_rad),
                "deg": float(position_rad) * 180.0 / math.pi,
            }
        )
    touch = []
    for module_id, values in touch_modules.items():
        touch.append(
            {
                "module_id": module_id,
                "location": REVO3_TOUCH_MODULES[module_id][0],
                "sensors": [
                    {"sdk_channel_id": channel_id, "value": float(sensor_value)}
                    for channel_id, sensor_value in enumerate(values, start=1)
                ],
            }
        )
    return {
        "frame": int(frame_index),
        "timestamp_ns": timestamp_ns,
        "joints_sdk_order": joints,
        "touch_modules": touch,
    }


def _print_record(record: dict) -> None:
    print(f"frame={record['frame']} timestamp_ns={record['timestamp_ns']}")
    for joint in record["joints_sdk_order"]:
        print(
            f"  joint_{joint['joint_id']:02d} {joint['name']}: "
            f"rad={joint['rad']:.7f} deg={joint['deg']:.4f}"
        )
    for module in record["touch_modules"]:
        values = " ".join(
            f"sdk_channel_{sensor['sdk_channel_id']}={sensor['value']:.7g}"
            for sensor in module["sensors"]
        )
        print(f"  module_{module['module_id']} {module['location']}: {values}")


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(async_main(build_parser().parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
