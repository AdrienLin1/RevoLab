# Copyright (c) 2026, BrainCo.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Live / on-disk visualization of TacSL fingertip tactile images and arrays.

This is a thin ``gymnasium`` wrapper that, every ``N`` environment steps, reads the
GelSight-style tactile images produced by the TacSL ``VisuoTactileSensor`` objects in the
scene (``*_tactile_sensor``), optional TacSL force-field arrays, and fingertip
3D contact forces. It lays them out into a single frame that is:

* always written to ``<out_dir>/latest.png`` (overwritten each time, easy to watch with
  any auto-refreshing image viewer), and
* optionally written to ``<out_dir>/frame_XXXXXX.png`` as a sequence, and
* optionally shown in a live OpenCV window (``--tactile_vis_show``).

It works with the ``BrainCo-Dexsuite-Revo3-Right-Lift-Tactile-*`` tasks and the
HORA screw tactile tasks. Image sensors may provide RGB/depth; force-field-only
sensors provide normal/shear taxel arrays.

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
# Gel pad physical size from revo3_right_tactile.usda gel_surface (m): 24 mm x 20 mm.
_GEL_SURFACE_ASPECT = 24.0 / 20.0
_ARRAY_CELL_BASE_PX = 14
_ARRAY_TILES_PER_ROW = 3
_LABEL_BAR_H = 28
_SCATTER_LONG_SIDE_PX = 252
_SCATTER_MARGIN_PX = 18
_SCATTER_POINT_RADIUS_PX = 9


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
        show_arrays: bool = True,
        show_contact_forces: bool = True,
    ):
        super().__init__(env)
        self.out_dir = out_dir
        self.every = max(1, int(every))
        self.show = bool(show)
        self.env_index = int(env_index)
        self.show_depth = bool(show_depth)
        self.show_arrays = bool(show_arrays)
        self.show_contact_forces = bool(show_contact_forces)

        self._step_count = 0
        self._frame_count = 0
        self._warned = False
        self._cv2_gui = False
        self._window = "TacSL tactile"
        self._window_size: tuple[int, int] | None = None

        os.makedirs(self.out_dir, exist_ok=True)
        self._html_path = self._write_live_html()

        self._sensors = self._discover_sensors()
        self._force_sensors = self._discover_force_sensors()
        if not self._sensors and not self._force_sensors:
            print(
                "[TactileViz][WARN] No tactile sensors found in the scene. "
                "Use a tactile task and enable cameras/force-field sensors as needed."
            )
        else:
            names = ", ".join(name for name, _ in self._sensors) or "none"
            force_names = ", ".join(name for name, _ in self._force_sensors) or "none"
            print(f"[TactileViz] Visualizing tactile sensors: {names}")
            print(f"[TactileViz] Visualizing 3D force sensors: {force_names}")
            print(f"[TactileViz] Frames -> {os.path.abspath(self.out_dir)} (every {self.every} steps)")

        if cv2 is None:
            print("[TactileViz][WARN] OpenCV (cv2) not importable; tactile visualization disabled.")
            self._sensors = []
            self._force_sensors = []

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
        scene = getattr(self._root_env(), "scene", None)
        sensors = getattr(scene, "sensors", None)
        if sensors is None:
            return []
        found = {name: sensor for name, sensor in sensors.items() if name.endswith(_SENSOR_SUFFIX)}

        def sort_key(name: str) -> tuple[int, str]:
            finger = name[: -len(_SENSOR_SUFFIX)]
            order = _FINGER_ORDER.index(finger) if finger in _FINGER_ORDER else len(_FINGER_ORDER)
            return (order, name)

        return [(name, found[name]) for name in sorted(found, key=sort_key)]

    def _discover_force_sensors(self) -> list[tuple[str, object]]:
        scene = getattr(self._root_env(), "scene", None)
        sensors = getattr(scene, "sensors", None)
        if sensors is None:
            return []

        found = {}
        for name, sensor in sensors.items():
            data = getattr(sensor, "data", None)
            if data is None:
                continue
            has_force = getattr(data, "net_forces_w", None) is not None or getattr(data, "force_matrix_w", None) is not None
            if has_force and (name.startswith("contact_sensor_") or name.endswith("_tactile_force")):
                found[name] = sensor

        def sort_key(name: str) -> tuple[int, str]:
            if name.startswith("contact_sensor_"):
                try:
                    return (int(name.rsplit("_", 1)[-1]), name)
                except ValueError:
                    return (len(_FINGER_ORDER), name)
            for idx, finger in enumerate(_FINGER_ORDER):
                if name.startswith(finger):
                    return (idx, name)
            return (len(_FINGER_ORDER), name)

        return [(name, found[name]) for name in sorted(found, key=sort_key)]

    # ------------------------------------------------------------------ gym API
    def step(self, action):
        out = self.env.step(action)
        self._step_count += 1
        if (self._sensors or self._force_sensors) and (self._step_count % self.every == 0):
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

    def __getattr__(self, name: str):
        """Forward task-specific attributes through the visualization wrapper."""

        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.env, name)

    # ------------------------------------------------------------------ rendering
    def _root_env(self):
        return getattr(self.env, "unwrapped", getattr(self.env, "_env", self.env))

    def _label(self, tile: np.ndarray, text: str) -> np.ndarray:
        bar_h = min(_LABEL_BAR_H, tile.shape[0])
        cv2.rectangle(tile, (0, 0), (tile.shape[1], bar_h), (0, 0, 0), thickness=-1)
        font_scale = 0.42 if tile.shape[1] >= 220 else 0.36
        cv2.putText(
            tile,
            text,
            (4, bar_h - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
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

    @staticmethod
    def _as_numpy_env(value, env_index: int) -> np.ndarray:
        arr = value[env_index]
        if hasattr(arr, "detach"):
            arr = arr.detach().cpu().numpy()
        return np.asarray(arr)

    def _array_shape(self, sensor, flat_size: int) -> tuple[int, int]:
        array_size = getattr(getattr(sensor, "cfg", None), "tactile_array_size", None)
        if array_size is not None:
            rows, cols = int(array_size[0]), int(array_size[1])
            if rows * cols == flat_size:
                return rows, cols
        side = int(round(np.sqrt(flat_size)))
        if side * side == flat_size:
            return side, side
        return 1, flat_size

    @staticmethod
    def _array_display_aspect(sensor) -> float:
        """Return gel-pad width/height for display (rows span the longer in-plane axis)."""
        array_size = getattr(getattr(sensor, "cfg", None), "tactile_array_size", None)
        if array_size is not None and len(array_size) == 2:
            rows, cols = int(array_size[0]), int(array_size[1])
            if rows > 0 and cols > 0 and rows != cols:
                return float(rows) / float(cols)
        return _GEL_SURFACE_ASPECT

    def _array_tile(self, sensor, finger: str) -> np.ndarray | None:
        normal = getattr(sensor.data, "tactile_normal_force", None)
        shear = getattr(sensor.data, "tactile_shear_force", None)
        if normal is None or shear is None:
            return None

        normal_arr = self._as_numpy_env(normal, self.env_index).reshape(-1)
        shear_arr = self._as_numpy_env(shear, self.env_index).reshape(normal_arr.shape[0], -1)
        if shear_arr.shape[-1] < 2:
            return None
        rows, cols = self._array_shape(sensor, normal_arr.size)
        force = np.concatenate([normal_arr[:, None], shear_arr[:, :2]], axis=-1)
        plane_xy = getattr(sensor, "_tactile_plane_xy", None)
        layout_name = getattr(sensor, "_tactile_layout_name", None)
        if layout_name is None:
            layout_name = getattr(getattr(sensor, "cfg", None), "tactile_layout", "regular_grid")
        if layout_name == "estimated_official" and plane_xy is not None:
            if hasattr(plane_xy, "detach"):
                plane_xy = plane_xy.detach().cpu().numpy()
            plane_xy = np.asarray(plane_xy, dtype=np.float32).reshape(-1, 2)
            if plane_xy.shape[0] == force.shape[0]:
                return self._estimated_array_tile(force, plane_xy, finger)

        strength = np.linalg.norm(force, axis=-1).reshape(rows, cols)
        shear_grid = shear_arr[:, :2].reshape(rows, cols, 2)

        aspect = self._array_display_aspect(sensor)
        cell_h = _ARRAY_CELL_BASE_PX
        cell_w = max(4, int(round(cell_h * aspect)))
        pad = 6
        map_h = cols * cell_h
        map_w = rows * cell_w
        tile_h = _LABEL_BAR_H + map_h + 2 * pad
        tile_w = map_w + 2 * pad
        tile = np.full((tile_h, tile_w, 3), 24, dtype=np.uint8)
        max_strength = max(float(np.nanmax(strength)), 1e-9)
        for row in range(rows):
            for col in range(cols):
                value = float(np.clip(strength[row, col] / max_strength, 0.0, 1.0))
                color = cv2.applyColorMap(np.array([[int(value * 255)]], dtype=np.uint8), cv2.COLORMAP_TURBO)[0, 0]
                x0 = pad + row * cell_w
                y0 = _LABEL_BAR_H + pad + col * cell_h
                cv2.rectangle(tile, (x0, y0), (x0 + cell_w - 1, y0 + cell_h - 1), color.tolist(), thickness=-1)
                cx = x0 + cell_w // 2
                cy = y0 + cell_h // 2
                shear_vec = shear_grid[row, col]
                shear_norm = float(np.linalg.norm(shear_vec))
                if shear_norm > 1e-9 and value > 0.05:
                    arrow_len = min(cell_w, cell_h) * (0.18 + 0.34 * value)
                    delta = shear_vec / shear_norm * arrow_len
                    end = (int(round(cx + delta[0])), int(round(cy - delta[1])))
                    cv2.arrowedLine(tile, (cx, cy), end, (255, 255, 255), 1, cv2.LINE_AA, tipLength=0.35)
        return self._label(tile, f"{finger} max={max_strength:.2g}")

    @staticmethod
    def _aggregate_layout_slots(force: np.ndarray, plane_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Average repeated 16x16 query slots at each physical sensor center."""

        rounded_xy = np.round(np.asarray(plane_xy, dtype=np.float32), decimals=7)
        centers, inverse = np.unique(rounded_xy, axis=0, return_inverse=True)
        aggregated = np.zeros((len(centers), force.shape[1]), dtype=np.float32)
        counts = np.zeros(len(centers), dtype=np.float32)
        np.add.at(aggregated, inverse, force)
        np.add.at(counts, inverse, 1.0)
        aggregated /= counts[:, None]
        return centers, aggregated

    @staticmethod
    def _layout_screen_coordinates(centers: np.ndarray, finger: str) -> np.ndarray:
        """Orient every physical finger layout with its distal direction upward.

        Args:
            centers: Physical centers in longitudinal/lateral finger coordinates.
            finger: Canonical finger name retained for a uniform call contract.

        Returns:
            Display coordinates with lateral on x and distal longitudinal on y.
        """

        del finger
        return np.stack((centers[:, 1], centers[:, 0]), axis=-1)

    def _estimated_array_tile(
        self,
        force: np.ndarray,
        plane_xy: np.ndarray,
        finger: str,
    ) -> np.ndarray:
        """Render the estimated physical sensor centers instead of duplicated query slots."""

        centers, aggregated_force = self._aggregate_layout_slots(force, plane_xy)
        screen_xy = self._layout_screen_coordinates(centers, finger)
        spans = np.ptp(screen_xy, axis=0)
        spans = np.maximum(spans, 1.0e-6)
        if spans[0] >= spans[1]:
            map_w = _SCATTER_LONG_SIDE_PX
            map_h = max(150, int(round(map_w * spans[1] / spans[0])))
        else:
            map_h = _SCATTER_LONG_SIDE_PX
            map_w = max(150, int(round(map_h * spans[0] / spans[1])))

        tile_h = _LABEL_BAR_H + map_h + 2 * _SCATTER_MARGIN_PX
        tile_w = map_w + 2 * _SCATTER_MARGIN_PX
        tile = np.full((tile_h, tile_w, 3), 24, dtype=np.uint8)
        map_origin = np.array(
            [_SCATTER_MARGIN_PX, _LABEL_BAR_H + _SCATTER_MARGIN_PX],
            dtype=np.float32,
        )
        normalized = (screen_xy - screen_xy.min(axis=0)) / spans
        pixel_normalized = normalized.copy()
        pixel_normalized[:, 1] = 1.0 - pixel_normalized[:, 1]
        pixel_xy = map_origin + pixel_normalized * np.array([map_w, map_h], dtype=np.float32)

        strength = np.linalg.norm(aggregated_force, axis=-1)
        max_strength = max(float(np.nanmax(strength)), 1.0e-9)
        for center_px, sensor_force, sensor_strength in zip(pixel_xy, aggregated_force, strength):
            value = float(np.clip(sensor_strength / max_strength, 0.0, 1.0))
            if sensor_strength <= 1.0e-9:
                color = (45, 28, 50)
            else:
                color = tuple(
                    int(channel)
                    for channel in cv2.applyColorMap(
                        np.array([[int(value * 255)]], dtype=np.uint8),
                        cv2.COLORMAP_TURBO,
                    )[0, 0]
                )
            center = tuple(np.rint(center_px).astype(np.int32).tolist())
            cv2.circle(tile, center, _SCATTER_POINT_RADIUS_PX, color, thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(
                tile,
                center,
                _SCATTER_POINT_RADIUS_PX,
                (175, 175, 175),
                thickness=1,
                lineType=cv2.LINE_AA,
            )

            shear = sensor_force[1:3]
            shear_norm = float(np.linalg.norm(shear))
            if shear_norm > 1.0e-9 and value > 0.05:
                screen_shear = self._layout_screen_coordinates(
                    shear.reshape(1, 2),
                    finger,
                )[0]
                screen_shear /= max(float(np.linalg.norm(screen_shear)), 1.0e-9)
                screen_shear[1] *= -1.0
                arrow_len = _SCATTER_POINT_RADIUS_PX * (0.7 + 0.8 * value)
                end = tuple(np.rint(center_px + screen_shear * arrow_len).astype(np.int32).tolist())
                cv2.arrowedLine(tile, center, end, (255, 255, 255), 1, cv2.LINE_AA, tipLength=0.35)

        return self._label(tile, f"{finger} sensors={len(centers)} max={max_strength:.2g}")

    def _force_label(self, name: str, index: int) -> str:
        if name.startswith("contact_sensor_") and index < len(_FINGER_ORDER):
            return _FINGER_ORDER[index]
        for finger in _FINGER_ORDER:
            if name.startswith(finger):
                return finger
        return name

    def _force_vector(self, sensor) -> np.ndarray | None:
        data = sensor.data
        force = getattr(data, "net_forces_w", None)
        if force is None:
            force = getattr(data, "force_matrix_w", None)
        if force is None:
            return None
        arr = self._as_numpy_env(force, self.env_index).reshape(-1, 3)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr.sum(axis=0)

    def _force_panel(self, panel_w: int) -> np.ndarray | None:
        rows = []
        for index, (name, sensor) in enumerate(self._force_sensors):
            force = self._force_vector(sensor)
            if force is None:
                continue
            rows.append((self._force_label(name, index), force))
        if not rows:
            return None

        line_h = 24
        panel_w = max(panel_w, 430)
        panel_h = 28 + line_h * len(rows) + 8
        panel = np.full((panel_h, panel_w, 3), 22, dtype=np.uint8)
        cv2.putText(panel, "3D contact force (world, N)", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (240, 240, 240), 1, cv2.LINE_AA)
        for row_idx, (label, force) in enumerate(rows):
            y = 28 + row_idx * line_h + 16
            mag = float(np.linalg.norm(force))
            text = f"{label:<6} Fx {force[0]:+6.2f}  Fy {force[1]:+6.2f}  Fz {force[2]:+6.2f}  |F| {mag:6.2f}"
            color = (80, 220, 255) if mag > 1e-3 else (130, 130, 130)
            cv2.putText(panel, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
        return panel

    @staticmethod
    def _hstack_tiles(tiles: list[np.ndarray]) -> np.ndarray:
        if not tiles:
            return np.empty((0, 0, 3), dtype=np.uint8)
        max_h = max(tile.shape[0] for tile in tiles)
        padded = []
        for tile in tiles:
            if tile.shape[0] < max_h:
                tile = cv2.copyMakeBorder(tile, 0, max_h - tile.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(18, 18, 18))
            padded.append(tile)
        return np.hstack(padded)

    @staticmethod
    def _vstack_rows(rows: list[np.ndarray]) -> np.ndarray:
        rows = [row for row in rows if row.size > 0]
        max_w = max(row.shape[1] for row in rows)
        padded = []
        for row in rows:
            if row.shape[1] < max_w:
                row = cv2.copyMakeBorder(row, 0, 0, 0, max_w - row.shape[1], cv2.BORDER_CONSTANT, value=(18, 18, 18))
            padded.append(row)
        return np.vstack(padded)

    @classmethod
    def _layout_tile_grid(cls, tiles: list[np.ndarray], tiles_per_row: int) -> np.ndarray | None:
        """Lay out same-height tiles in a centered grid (e.g. 3 + 2 fingers)."""
        if not tiles:
            return None
        max_h = max(tile.shape[0] for tile in tiles)
        normalized = []
        for tile in tiles:
            if tile.shape[0] < max_h:
                tile = cv2.copyMakeBorder(
                    tile, 0, max_h - tile.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(18, 18, 18)
                )
            normalized.append(tile)

        grid_rows = []
        for start in range(0, len(normalized), tiles_per_row):
            chunk = normalized[start : start + tiles_per_row]
            row = cls._hstack_tiles(chunk)
            if len(chunk) < tiles_per_row:
                target_w = cls._hstack_tiles(normalized[:tiles_per_row]).shape[1]
                if row.shape[1] < target_w:
                    pad_total = target_w - row.shape[1]
                    left = pad_total // 2
                    right = pad_total - left
                    row = cv2.copyMakeBorder(
                        row, 0, 0, left, right, cv2.BORDER_CONSTANT, value=(18, 18, 18)
                    )
            grid_rows.append(row)
        return cls._vstack_rows(grid_rows)

    def _resize_live_window(self, frame: np.ndarray) -> None:
        if not (self.show and self._cv2_gui and cv2 is not None):
            return
        frame_h, frame_w = frame.shape[:2]
        if self._window_size == (frame_w, frame_h):
            return
        try:
            cv2.resizeWindow(self._window, frame_w, frame_h)
            self._window_size = (frame_w, frame_h)
        except Exception:
            pass

    def _render_once(self) -> None:
        rgb_tiles = []
        depth_tiles = []
        array_tiles = []
        for name, sensor in self._sensors:
            finger = name[: -len(_SENSOR_SUFFIX)]
            rgb = self._rgb_tile(sensor, finger)
            if rgb is not None:
                rgb_tiles.append(rgb)
            if self.show_depth:
                depth = self._depth_tile(sensor, finger)
                if depth is not None:
                    depth_tiles.append(depth)
            if self.show_arrays:
                array_tile = self._array_tile(sensor, finger)
                if array_tile is not None:
                    array_tiles.append(array_tile)

        rows = []
        if rgb_tiles:
            rows.append(self._hstack_tiles(rgb_tiles))
        if depth_tiles:
            rows.append(self._hstack_tiles(depth_tiles))
        array_row = self._layout_tile_grid(array_tiles, _ARRAY_TILES_PER_ROW)
        if array_row is not None:
            rows.append(array_row)
        if self.show_contact_forces:
            target_w = array_row.shape[1] if array_row is not None else 0
            force_panel = self._force_panel(target_w)
            if force_panel is not None:
                rows.append(force_panel)
        if not rows:
            return
        frame = self._vstack_rows(rows) if len(rows) > 1 else rows[0]

        cv2.imwrite(os.path.join(self.out_dir, "latest.png"), frame)
        cv2.imwrite(os.path.join(self.out_dir, f"frame_{self._frame_count:06d}.png"), frame)
        self._frame_count += 1

        if self.show and self._cv2_gui:
            try:
                self._resize_live_window(frame)
                cv2.imshow(self._window, frame)
                cv2.waitKey(1)
            except Exception as exc:
                print(f"[TactileViz][WARN] live window unavailable ({exc}); browser/disk view still active.")
                self._cv2_gui = False
