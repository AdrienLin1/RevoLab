# Copyright (c) 2026, BrainCo.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Live / on-disk visualization of TacSL fingertip tactile images.

This is a thin ``gymnasium`` wrapper that, every ``N`` environment steps, reads the
GelSight-style tactile images produced by the TacSL ``VisuoTactileSensor`` objects in the
scene (``*_tactile_sensor``) and lays them out into a single frame that is:

* always written to ``<out_dir>/latest.png`` (overwritten each time, easy to watch with
  any auto-refreshing image viewer), and
* optionally written to ``<out_dir>/frame_XXXXXX.png`` as a sequence, and
* optionally shown in a live OpenCV window (``--tactile_vis_show``).

It works with the ``BrainCo-Dexsuite-Revo3-Right-Lift-Tactile-*`` tasks. The sensor already
computes both depth and RGB every step (the depth observation term triggers the render), so
reading the RGB buffer here is essentially free.

The wrapper is intentionally defensive: any failure inside the rendering path is caught and
logged once, so it can never crash a training run.
"""

from __future__ import annotations

import os
import webbrowser

import gymnasium as gym
import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - cv2 ships with the Isaac Sim python env
    cv2 = None

# Preferred finger order for display (thumb -> little). Falls back to sorted names.
_FINGER_ORDER = ("thumb", "index", "middle", "ring", "little")
_SENSOR_SUFFIX = "_tactile_sensor"


class TactileVizWrapper(gym.Wrapper):
    """Visualize TacSL tactile RGB (and optionally depth) images during rollout/training."""

    def __init__(
        self,
        env: gym.Env,
        out_dir: str,
        every: int = 2,
        show: bool = False,
        env_index: int = 0,
        show_depth: bool = True,
    ):
        super().__init__(env)
        self.out_dir = out_dir
        self.every = max(1, int(every))
        self.show = bool(show)
        self.env_index = int(env_index)
        self.show_depth = bool(show_depth)

        self._step_count = 0
        self._frame_count = 0
        self._warned = False
        self._cv2_gui = False
        self._window = "TacSL tactile (RGB top / depth bottom)"

        os.makedirs(self.out_dir, exist_ok=True)
        self._html_path = self._write_live_html()

        self._sensors = self._discover_sensors()
        if not self._sensors:
            print(
                "[TactileViz][WARN] No '*_tactile_sensor' sensors found in the scene. "
                "Is this a '*-Lift-Tactile-*' task with cameras enabled (--enable_cameras)?"
            )
        else:
            names = ", ".join(name for name, _ in self._sensors)
            print(f"[TactileViz] Visualizing {len(self._sensors)} tactile sensors: {names}")
            print(f"[TactileViz] Frames -> {os.path.abspath(self.out_dir)} (every {self.every} steps)")

        if cv2 is None:
            print("[TactileViz][WARN] OpenCV (cv2) not importable; tactile visualization disabled.")
            self._sensors = []

        if self.show:
            self._setup_live_view()

        num_envs = getattr(self.env.unwrapped, "num_envs", 1)
        if num_envs > 64:
            print(
                f"[TactileViz][WARN] num_envs={num_envs}. Tactile cameras render for every env; "
                "use a small --num_envs (e.g. 1-16) when visualizing."
            )

    # ------------------------------------------------------------------ live view
    def _write_live_html(self) -> str:
        """Write a self-refreshing HTML page that always shows the newest frame."""
        path = os.path.join(self.out_dir, "live.html")
        html = (
            "<!doctype html><meta charset=utf-8><title>TacSL tactile</title>"
            "<style>body{margin:0;background:#111;display:flex;align-items:center;"
            "justify-content:center;height:100vh}img{max-width:100vw;max-height:100vh;"
            "image-rendering:pixelated}</style>"
            "<img id=v src=latest.png>"
            "<script>setInterval(()=>{document.getElementById('v').src="
            "'latest.png?t='+Date.now();},400);</script>"
        )
        with open(path, "w") as fh:
            fh.write(html)
        return path

    def _setup_live_view(self) -> None:
        """Prefer a native OpenCV window; fall back to opening the live HTML in a browser."""
        if cv2 is not None:
            try:
                cv2.namedWindow(self._window, cv2.WINDOW_NORMAL)
                cv2.waitKey(1)
                self._cv2_gui = True
                return
            except Exception:
                self._cv2_gui = False
        # cv2 has no GUI backend (e.g. opencv-python-headless): use the browser page instead.
        print(
            "[TactileViz] OpenCV has no GUI backend (opencv-python-headless); opening a browser "
            f"live view instead:\n[TactileViz]   {self._html_path}\n"
            f"[TactileViz] If no browser opens, run:  xdg-open {self._html_path}\n"
            f"[TactileViz] or watch the image directly:  eog {os.path.join(self.out_dir, 'latest.png')}"
        )
        try:
            webbrowser.open(f"file://{os.path.abspath(self._html_path)}")
        except Exception as exc:
            print(f"[TactileViz][WARN] could not auto-open browser ({exc}); open the file above manually.")

    # ------------------------------------------------------------------ discovery
    def _discover_sensors(self) -> list[tuple[str, object]]:
        scene = getattr(self.env.unwrapped, "scene", None)
        sensors = getattr(scene, "sensors", None)
        if sensors is None:
            return []
        found = {name: sensor for name, sensor in sensors.items() if name.endswith(_SENSOR_SUFFIX)}

        def sort_key(name: str) -> tuple[int, str]:
            finger = name[: -len(_SENSOR_SUFFIX)]
            order = _FINGER_ORDER.index(finger) if finger in _FINGER_ORDER else len(_FINGER_ORDER)
            return (order, name)

        return [(name, found[name]) for name in sorted(found, key=sort_key)]

    # ------------------------------------------------------------------ gym API
    def step(self, action):
        out = self.env.step(action)
        self._step_count += 1
        if self._sensors and (self._step_count % self.every == 0):
            try:
                self._render_once()
            except Exception as exc:  # never break training because of visualization
                if not self._warned:
                    print(f"[TactileViz][WARN] tactile render failed (suppressing further warnings): {exc}")
                    self._warned = True
        return out

    def close(self):
        if self.show and cv2 is not None:
            try:
                cv2.destroyWindow(self._window)
            except Exception:
                pass
        return self.env.close()

    # ------------------------------------------------------------------ rendering
    def _label(self, tile: np.ndarray, text: str) -> np.ndarray:
        cv2.rectangle(tile, (0, 0), (tile.shape[1], 18), (0, 0, 0), thickness=-1)
        cv2.putText(tile, text, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        return tile

    def _rgb_tile(self, sensor, finger: str) -> np.ndarray | None:
        img = getattr(sensor.data, "tactile_rgb_image", None)
        if img is None:
            return None
        arr = img[self.env_index].detach().cpu().numpy()  # (H, W, 3)
        if arr.dtype != np.uint8:
            arr = (arr * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
        arr = np.transpose(arr, (1, 0, 2))  # (W, H, 3), matches TacSL demo orientation
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)  # Taxim renders RGB; cv2 expects BGR
        return self._label(np.ascontiguousarray(arr), finger)

    def _depth_tile(self, sensor, finger: str) -> np.ndarray | None:
        img = getattr(sensor.data, "tactile_depth_image", None)
        if img is None:
            return None
        arr = img[self.env_index].detach().cpu().numpy()  # (H, W, 1)
        arr = np.squeeze(arr, axis=-1).T  # (W, H)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        norm = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        color = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
        return self._label(np.ascontiguousarray(color), f"{finger} (depth)")

    def _render_once(self) -> None:
        rgb_tiles = []
        depth_tiles = []
        for name, sensor in self._sensors:
            finger = name[: -len(_SENSOR_SUFFIX)]
            rgb = self._rgb_tile(sensor, finger)
            if rgb is not None:
                rgb_tiles.append(rgb)
            if self.show_depth:
                depth = self._depth_tile(sensor, finger)
                if depth is not None:
                    depth_tiles.append(depth)

        rows = []
        if rgb_tiles:
            rows.append(np.hstack(rgb_tiles))
        if depth_tiles:
            rows.append(np.hstack(depth_tiles))
        if not rows:
            return
        frame = np.vstack(rows) if len(rows) > 1 else rows[0]

        cv2.imwrite(os.path.join(self.out_dir, "latest.png"), frame)
        cv2.imwrite(os.path.join(self.out_dir, f"frame_{self._frame_count:06d}.png"), frame)
        self._frame_count += 1

        if self.show and self._cv2_gui:
            try:
                cv2.imshow(self._window, frame)
                cv2.waitKey(1)
            except Exception as exc:
                print(f"[TactileViz][WARN] live window unavailable ({exc}); browser/disk view still active.")
                self._cv2_gui = False
