import argparse
import math
import os
import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(PROJECT_DIR, ".ultralytics"))

try:
    from ultralytics import YOLO

    ULTRALYTICS_AVAILABLE = True
except ImportError:
    YOLO = None
    ULTRALYTICS_AVAILABLE = False

try:
    import rclpy
    from cv_bridge import CvBridge
    from message_filters import ApproximateTimeSynchronizer, Subscriber
    from my_detections.msg import Detection
    from radar_msgs.msg import RadarScan
    from rclpy.node import Node
    from sensor_msgs.msg import CameraInfo, Image

    ROS_AVAILABLE = True
except ImportError:
    rclpy = None
    CvBridge = None
    ApproximateTimeSynchronizer = None
    Subscriber = None
    Detection = None
    RadarScan = None
    CameraInfo = None
    Image = None

    class Node:
        pass

    ROS_AVAILABLE = False

try:
    import torch
except ImportError:
    torch = None


@dataclass
class YoloDetection:
    bbox: tuple
    class_name: str
    confidence: float
    depth_m: float
    center: tuple
    index: int


@dataclass
class RadarPoint:
    index: int
    range_m: float
    azimuth_rad: float
    rcs: float
    doppler_velocity: float
    x_forward: float
    y_left: float
    u: int
    v: int
    in_azimuth_gate: bool
    in_rcs_gate: bool
    in_image: bool


@dataclass
class FusionMatch:
    radar: RadarPoint
    yolo: YoloDetection
    score: float
    depth_diff_m: float
    track_id: int
    x_forward: float
    y_left: float
    range_m: float
    azimuth_rad: float
    camera_weight: float
    vrel_x: float
    vrel_y: float


@dataclass
class SyntheticRadarReturn:
    range: float
    azimuth: float
    rcs: float
    doppler_velocity: float


@dataclass
class SyntheticRadarScan:
    returns: list


class PersonCarRadarYOLOFusionNode(Node):
    """
    CPU-focused radar + YOLO fusion node for person and car only.

    Quick tuning guide. All ranges are practical starting points, not hard limits.

    Model / CPU:
    - model_path: default yolo12n.onnx. ONNX is fastest on CPU; PT is backup.
    - yolo_imgsz: 320-640. Up = better small/far detection, slower CPU.
      Down = faster, less small-object accuracy.
    - yolo_conf: 0.15-0.60. Up = fewer false detections, can miss weak objects.
      Down = more detections, more false positives.
    - yolo_iou: 0.30-0.70. Up = keeps more overlapping boxes.
      Down = removes duplicate boxes more aggressively.
    - max_yolo_hz: 2-10. Up = fresher boxes, more CPU load.
      Down = cooler/faster system, but boxes are cached longer.
    - max_cached_yolo_age_sec: 0.2-1.0. Up = tolerate slow YOLO, more stale risk.
      Down = safer timing, fewer fusions when YOLO is slow.
    - torch_num_threads: 1-8. Up can speed PT inference until CPU saturates.
      Down frees CPU for ROS, camera, and display.
    - opencv_num_threads: 1-4. Up can speed image operations. Down is more stable.

    Camera/depth sync:
    - sync_queue_size: 10-60. Up = tolerate jitter, more latency/memory.
      Down = lower latency, can drop pairs.
    - sync_slop_sec: 0.03-0.20. Up = more color/depth pairs, worse time match.
      Down = stricter sync, can reduce callbacks.

    Radar gating:
    - max_azimuth_deg: 20-45. Up = wider radar field, more side clutter.
      Down = center-only, less clutter.
    - min_rcs: -35 to 0. Up = stronger radar returns only, fewer ghosts.
      Down = accepts weak returns, useful indoors but adds clutter.
    - radar_buffer_sec: 0.3-2.0. Up = tolerate sparse radar, stale risk.
      Down = fresher radar, can miss scans.
    - radar_match_slop_sec: 0.05-0.30. Up = more fusion chances, worse timing.
      Down = stricter time alignment, fewer matches.
    - radar_lateral_offset_m: -1.0 to 1.0. Tune if radar is left/right of camera.
      Positive moves projected radar points right in the image, negative left.

    Depth / association:
    - depth_window_px: 5-31 odd number. Up = smoother depth, can mix background.
      Down = sharper local depth, noisier.
    - depth_gate_base_m: 0.5-3.0. Up = easier depth match, more false fusions.
      Down = stricter close-range match, can reject real objects.
    - depth_gate_ratio: 0.05-0.30. Up = easier far-range match.
      Down = stricter far-range match.
    - allow_missing_depth: True indoors/holes; False when depth is reliable.
    - camera_position_weight: 0.0-1.0. Up = trust camera depth more.
      Down = trust radar range/azimuth more.
    - velocity_alpha: 0.1-0.8. Up = velocity reacts faster, noisier.
      Down = smoother velocity, slower response.

    Debug display:
    - debug_visuals/show_*: False saves CPU and avoids GUI issues.
    - display_scale: 0.3-1.0. Up = bigger windows, more display cost.
    - radar_debug_range_m/lateral_m: increase if points are off the debug map.
    """

    def __init__(self):
        super().__init__("radar_yolo_fusion_node")

        # Model and CPU performance. For first run, keep ONNX beside this script.
        # yolo_imgsz and max_yolo_hz are usually the first two values to tune.
        self.declare_parameter("model_path", self.default_model_path())
        self.declare_parameter("device", "cpu")
        self.declare_parameter("yolo_imgsz", 480)
        self.declare_parameter("yolo_conf", 0.35)
        self.declare_parameter("yolo_iou", 0.45)
        self.declare_parameter("max_yolo_hz", 5.0)
        self.declare_parameter("max_cached_yolo_age_sec", 0.6)
        self.declare_parameter("torch_num_threads", max(1, min(4, os.cpu_count() or 2)))
        self.declare_parameter("opencv_num_threads", 1)

        # ROS topics and color/depth synchronization. Change these only if your
        # camera/radar topics differ or color/depth are not pairing reliably.
        self.declare_parameter("camera_info_topic", "/camera/camera/color/camera_info")
        self.declare_parameter("color_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("radar_topic", "/radar_scan")
        self.declare_parameter("detections_topic", "/detections")
        self.declare_parameter("sync_queue_size", 30)
        self.declare_parameter("sync_slop_sec", 0.1)

        # Radar projection and fusion gates. min_rcs (Radar Cross Section), radar_match_slop_sec,
        # depth_gate_base_m, and depth_gate_ratio are the main fusion knobs.
        self.declare_parameter("max_azimuth_deg", 34.5)
        self.declare_parameter("min_rcs", -25.0)
        self.declare_parameter("radar_buffer_sec", 1.0)
        self.declare_parameter("radar_match_slop_sec", 0.15)
        self.declare_parameter("radar_lateral_offset_m", 0.0)
        self.declare_parameter("depth_window_px", 11)
        self.declare_parameter("depth_gate_base_m", 1.6)
        self.declare_parameter("depth_gate_ratio", 0.16)
        self.declare_parameter("allow_missing_depth", True)
        self.declare_parameter("camera_position_weight", 0.5)
        self.declare_parameter("velocity_alpha", 0.35)

        # Debug views. Turn debug_visuals off for headless runs or weak CPUs.
        self.declare_parameter("debug_visuals", True)
        self.declare_parameter("show_yolo_debug", True)
        self.declare_parameter("show_radar_debug", True)
        self.declare_parameter("show_fusion_debug", True)
        self.declare_parameter("display_scale", 0.5)
        self.declare_parameter("radar_debug_range_m", 50.0)
        self.declare_parameter("radar_debug_lateral_m", 25.0)
        self.declare_parameter("radar_debug_max_labels", 16)
        self.declare_parameter("radar_debug_sample_rows", 8)
        self.declare_parameter("fusion_window_name", "Fusion Pipeline")

        requested_model_path = str(self.get_parameter("model_path").value)
        self.device = self.get_parameter("device").value
        self.yolo_imgsz = int(self.get_parameter("yolo_imgsz").value)
        self.yolo_conf = float(self.get_parameter("yolo_conf").value)
        self.yolo_iou = float(self.get_parameter("yolo_iou").value)
        self.max_yolo_hz = float(self.get_parameter("max_yolo_hz").value)
        self.min_yolo_period = 1.0 / max(self.max_yolo_hz, 0.1)
        self.max_cached_yolo_age_sec = float(self.get_parameter("max_cached_yolo_age_sec").value)
        self.sync_queue_size = int(self.get_parameter("sync_queue_size").value)
        self.sync_slop_sec = float(self.get_parameter("sync_slop_sec").value)
        self.max_azimuth_rad = math.radians(float(self.get_parameter("max_azimuth_deg").value))
        self.min_rcs = float(self.get_parameter("min_rcs").value)
        self.radar_buffer_sec = float(self.get_parameter("radar_buffer_sec").value)
        self.radar_match_slop_sec = float(self.get_parameter("radar_match_slop_sec").value)
        self.radar_lateral_offset_m = float(self.get_parameter("radar_lateral_offset_m").value)
        self.depth_window_px = int(self.get_parameter("depth_window_px").value)
        self.depth_gate_base_m = float(self.get_parameter("depth_gate_base_m").value)
        self.depth_gate_ratio = float(self.get_parameter("depth_gate_ratio").value)
        self.allow_missing_depth = bool(self.get_parameter("allow_missing_depth").value)
        self.camera_position_weight = min(
            max(float(self.get_parameter("camera_position_weight").value), 0.0),
            1.0,
        )
        self.velocity_alpha = float(self.get_parameter("velocity_alpha").value)
        self.debug_visuals = bool(self.get_parameter("debug_visuals").value)
        self.show_yolo_debug = bool(self.get_parameter("show_yolo_debug").value)
        self.show_radar_debug = bool(self.get_parameter("show_radar_debug").value)
        self.show_fusion_debug = bool(self.get_parameter("show_fusion_debug").value)
        self.display_scale = float(self.get_parameter("display_scale").value)
        self.radar_debug_range_m = float(self.get_parameter("radar_debug_range_m").value)
        self.radar_debug_lateral_m = float(self.get_parameter("radar_debug_lateral_m").value)
        self.radar_debug_max_labels = int(self.get_parameter("radar_debug_max_labels").value)
        self.radar_debug_sample_rows = int(self.get_parameter("radar_debug_sample_rows").value)
        self.fusion_window_name = str(self.get_parameter("fusion_window_name").value)
        self.model_path = self.select_model_path(requested_model_path)

        self.configure_cpu_runtime()

        self.fx = self.fy = self.cx = self.cy = None
        self.bridge = CvBridge()
        if not ULTRALYTICS_AVAILABLE:
            raise RuntimeError(
                "The live ROS node requires the ultralytics package. "
                "Install it or run this file with --offline-demo to use OpenCV DNN with the ONNX model."
            )
        self.model = YOLO(self.model_path, task="detect")
        self.target_names = {"person", "car"}
        self.target_class_ids = self.find_target_class_ids()

        camera_info_topic = self.get_parameter("camera_info_topic").value
        color_topic = self.get_parameter("color_topic").value
        depth_topic = self.get_parameter("depth_topic").value
        radar_topic = self.get_parameter("radar_topic").value
        detections_topic = self.get_parameter("detections_topic").value

        self.camera_info_sub = self.create_subscription(
            CameraInfo, camera_info_topic, self.camera_info_callback, 10
        )
        self.color_sub = Subscriber(self, Image, color_topic)
        self.depth_sub = Subscriber(self, Image, depth_topic)
        self.radar_sub = self.create_subscription(RadarScan, radar_topic, self.radar_callback, 100)
        self.detection_pub = self.create_publisher(Detection, detections_topic, 20)

        self.ts = ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub],
            queue_size=self.sync_queue_size,
            slop=self.sync_slop_sec,
        )
        self.ts.registerCallback(self.synced_callback)

        self.radar_buffer = deque(maxlen=120)
        self.yolo_detections = []
        self.yolo_cache_stale = True
        self.last_yolo_ts = 0.0
        self.last_yolo_runtime_ms = 0.0
        self.last_radar_points = []
        self.recent_matches = deque(maxlen=80)
        self.tracks = {}
        self.next_track_id = 1
        self.frame_count = 0
        self.last_status_log = 0.0
        self.display_available = True

        target_id_text = self.target_class_ids if self.target_class_ids is not None else "name-filter"
        self.get_logger().info(
            "CPU person/car Radar-YOLO fusion started. "
            f"model={self.model_path}, imgsz={self.yolo_imgsz}, "
            f"max_yolo_hz={self.max_yolo_hz:.1f}, classes={target_id_text}, "
            f"camera_position_weight={self.camera_position_weight:.2f}"
        )

    @staticmethod
    def default_model_path():
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolo12n.onnx")

    def select_model_path(self, model_path):
        onnx_candidates, pt_candidates, other_candidates = self.model_candidates(model_path)

        for candidate in onnx_candidates:
            if os.path.isfile(candidate):
                self.get_logger().info(f"Using ONNX model: {candidate}")
                return candidate

        for pt_path in pt_candidates:
            if not os.path.isfile(pt_path):
                continue
            converted_path = self.try_export_onnx(pt_path)
            if converted_path is not None:
                self.get_logger().info(f"Using exported ONNX model: {converted_path}")
                return converted_path

        for pt_path in pt_candidates:
            if os.path.isfile(pt_path):
                self.get_logger().warning(
                    f"ONNX model unavailable and export failed; falling back to PT model: {pt_path}"
                )
                return pt_path

        for candidate in other_candidates:
            if os.path.isfile(candidate):
                self.get_logger().warning(
                    f"Using non-ONNX/non-PT model because no ONNX/PT candidate was found: {candidate}"
                )
                return candidate

        tried = ", ".join(onnx_candidates + pt_candidates + other_candidates)
        raise FileNotFoundError(
            f"YOLO model not found. Tried: {tried}. "
            "Pass an existing file with --ros-args -p model_path:=PATH."
        )

    @staticmethod
    def model_candidates(model_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        paths = []

        if os.path.isabs(model_path):
            paths.append(model_path)
        else:
            paths.extend([os.path.join(script_dir, model_path), os.path.abspath(model_path)])

        base_name = os.path.basename(model_path)
        if base_name in {"yolo12n.pt", "yolo12n.onnx"} or model_path == "":
            paths.extend(
                [
                    os.path.join(script_dir, "yolo12n.onnx"),
                    os.path.join(script_dir, "yolo12n.pt"),
                    os.path.abspath("yolo12n.onnx"),
                    os.path.abspath("yolo12n.pt"),
                ]
            )

        expanded = []
        for path in paths:
            root, ext = os.path.splitext(path)
            if ext.lower() == ".onnx":
                expanded.extend([path, root + ".pt"])
            elif ext.lower() == ".pt":
                expanded.extend([root + ".onnx", path])
            else:
                expanded.append(path)

        unique_paths = []
        for path in expanded:
            normalized = os.path.normpath(path)
            if normalized not in unique_paths:
                unique_paths.append(normalized)

        onnx_candidates = [path for path in unique_paths if path.lower().endswith(".onnx")]
        pt_candidates = [path for path in unique_paths if path.lower().endswith(".pt")]
        other_candidates = [
            path for path in unique_paths if not path.lower().endswith((".onnx", ".pt"))
        ]
        return onnx_candidates, pt_candidates, other_candidates

    def try_export_onnx(self, pt_path):
        if not ULTRALYTICS_AVAILABLE:
            return None

        expected_onnx = os.path.splitext(pt_path)[0] + ".onnx"
        self.get_logger().warning(
            f"ONNX model not found. Trying to export from PT: {pt_path}"
        )
        try:
            exported = YOLO(pt_path, task="detect").export(
                format="onnx",
                imgsz=self.yolo_imgsz,
                device=self.device,
                simplify=True,
            )
        except Exception as exc:
            self.get_logger().warning(f"ONNX export failed: {exc}")
            return None

        exported_path = str(exported) if exported is not None else expected_onnx
        if not os.path.isabs(exported_path):
            exported_path = os.path.join(os.path.dirname(pt_path), exported_path)
        exported_path = os.path.normpath(exported_path)

        if os.path.isfile(exported_path):
            return exported_path
        if os.path.isfile(expected_onnx):
            return expected_onnx

        self.get_logger().warning(
            f"ONNX export finished but no ONNX file was found at {exported_path}"
        )
        return None

    def configure_cpu_runtime(self):
        cv2.setUseOptimized(True)
        cv_threads = int(self.get_parameter("opencv_num_threads").value)
        if cv_threads > 0:
            cv2.setNumThreads(cv_threads)

        if torch is None:
            return

        torch_threads = int(self.get_parameter("torch_num_threads").value)
        if torch_threads > 0:
            torch.set_num_threads(torch_threads)
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass

    def find_target_class_ids(self):
        names = self.model.names
        items = names.items() if isinstance(names, dict) else enumerate(names)
        class_ids = [int(class_id) for class_id, name in items if name in self.target_names]
        return class_ids if class_ids else None

    def camera_info_callback(self, msg):
        if self.fx is None:
            self.fx = float(msg.k[0])
            self.cx = float(msg.k[2])
            self.fy = float(msg.k[4])
            self.cy = float(msg.k[5])
            self.get_logger().info(
                f"Camera intrinsics loaded: fx={self.fx:.1f}, fy={self.fy:.1f}, "
                f"cx={self.cx:.1f}, cy={self.cy:.1f}"
            )

    @staticmethod
    def timestamp(msg):
        return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def radar_callback(self, msg):
        self.radar_buffer.append((self.timestamp(msg), msg))

    def synced_callback(self, color_msg, depth_msg):
        current_ts = self.timestamp(color_msg)
        if self.fx is None:
            self.throttled_log("Waiting for camera_info before projecting radar.")
            return

        color_img = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8")
        depth_img = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")

        run_yolo = current_ts - self.last_yolo_ts >= self.min_yolo_period
        if run_yolo:
            self.yolo_detections = self.run_yolo(color_img, depth_img, current_ts)

        yolo_age = current_ts - self.last_yolo_ts if self.last_yolo_ts > 0.0 else float("inf")
        self.yolo_cache_stale = yolo_age > self.max_cached_yolo_age_sec
        fusion_yolo_detections = [] if self.yolo_cache_stale else self.yolo_detections

        closest_radar_msg, radar_dt = self.closest_radar_scan(current_ts)
        radar_points = []
        matches = []
        if closest_radar_msg is not None and radar_dt <= self.radar_match_slop_sec:
            radar_points = self.extract_radar_points(closest_radar_msg, color_img.shape)
            matches = self.associate(radar_points, fusion_yolo_detections, current_ts)
            for match in matches:
                self.publish_detection(match)
        self.last_radar_points = radar_points

        if self.debug_visuals:
            self.draw_debug_views(color_img, radar_points, matches, current_ts, radar_dt, run_yolo)

        self.frame_count += 1

    def run_yolo(self, color_img, depth_img, current_ts):
        start = time.perf_counter()
        predict_kwargs = {
            "imgsz": self.yolo_imgsz,
            "conf": self.yolo_conf,
            "iou": self.yolo_iou,
            "device": self.device,
            "verbose": False,
        }
        if self.target_class_ids is not None:
            predict_kwargs["classes"] = self.target_class_ids

        result = self.model(color_img, **predict_kwargs)[0]
        detections = []
        for index, box in enumerate(result.boxes):
            class_id = int(box.cls[0])
            class_name = self.model.names[class_id]
            if class_name not in self.target_names:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            c_x = int((x1 + x2) * 0.5)
            c_y = int((y1 + y2) * 0.5)
            depth_m = self.robust_depth(depth_img, c_x, c_y, self.depth_window_px)
            detections.append(
                YoloDetection(
                    bbox=(x1, y1, x2, y2),
                    class_name=class_name,
                    confidence=float(box.conf[0]),
                    depth_m=depth_m,
                    center=(c_x, c_y),
                    index=index,
                )
            )

        self.last_yolo_runtime_ms = (time.perf_counter() - start) * 1000.0
        self.last_yolo_ts = current_ts
        return detections

    @staticmethod
    def robust_depth(depth_img, c_x, c_y, window):
        h, w = depth_img.shape[:2]
        half = max(1, int(window) // 2)
        x_min = max(0, c_x - half)
        x_max = min(w, c_x + half + 1)
        y_min = max(0, c_y - half)
        y_max = min(h, c_y + half + 1)

        roi = depth_img[y_min:y_max, x_min:x_max]
        valid = roi[np.isfinite(roi)]
        valid = valid[valid > 0]
        if valid.size == 0:
            return 0.0

        median_value = float(np.median(valid))
        return median_value / 1000.0 if median_value > 100.0 else median_value

    def closest_radar_scan(self, current_ts):
        while self.radar_buffer and current_ts - self.radar_buffer[0][0] > self.radar_buffer_sec:
            self.radar_buffer.popleft()

        closest_msg = None
        min_dt = float("inf")
        for ts, msg in self.radar_buffer:
            dt = abs(current_ts - ts)
            if dt < min_dt:
                min_dt = dt
                closest_msg = msg

        return closest_msg, min_dt

    def extract_radar_points(self, radar_msg, image_shape):
        h, w = image_shape[:2]
        points = []
        for index, radar_return in enumerate(radar_msg.returns):
            range_m = float(radar_return.range)
            azimuth_rad = float(radar_return.azimuth)
            rcs = float(radar_return.rcs)
            doppler_velocity = float(getattr(radar_return, "doppler_velocity", 0.0))

            x_forward = range_m * math.cos(azimuth_rad)
            y_left = range_m * math.sin(azimuth_rad)
            z_forward = max(x_forward, 0.001)
            u = int(self.fx * (y_left + self.radar_lateral_offset_m) / z_forward + self.cx)
            v = int(self.cy)

            in_azimuth_gate = abs(azimuth_rad) <= self.max_azimuth_rad
            in_rcs_gate = rcs >= self.min_rcs
            in_image = 0 <= u < w and 0 <= v < h and x_forward > 0.1
            points.append(
                RadarPoint(
                    index=index,
                    range_m=range_m,
                    azimuth_rad=azimuth_rad,
                    rcs=rcs,
                    doppler_velocity=doppler_velocity,
                    x_forward=x_forward,
                    y_left=y_left,
                    u=u,
                    v=v,
                    in_azimuth_gate=in_azimuth_gate,
                    in_rcs_gate=in_rcs_gate,
                    in_image=in_image,
                )
            )
        return points

    def associate(self, radar_points, yolo_detections, current_ts):
        candidates = []
        for radar in radar_points:
            if not (radar.in_azimuth_gate and radar.in_rcs_gate and radar.in_image):
                continue

            for yolo in yolo_detections:
                x1, y1, x2, y2 = yolo.bbox
                box_width = max(1, x2 - x1)
                margin = max(10, int(0.08 * box_width))
                if not (x1 - margin <= radar.u <= x2 + margin):
                    continue

                depth_valid = yolo.depth_m > 0.2
                depth_diff = abs(yolo.depth_m - radar.range_m) if depth_valid else float("inf")

                # Adaptive depth gate:
                # close range uses at least depth_gate_base_m;
                # far range grows by depth_gate_ratio * radar range.
                # Increase these values when real targets are rejected by depth.
                # Decrease them when radar points attach to the wrong YOLO box.
                depth_gate = max(self.depth_gate_base_m, self.depth_gate_ratio * radar.range_m)
                if depth_valid and depth_diff > depth_gate:
                    continue
                if not depth_valid and not self.allow_missing_depth:
                    continue

                # Final association score. Higher score wins. Horizontal alignment
                # matters most, then depth agreement, YOLO confidence, and RCS.
                box_center_u = (x1 + x2) * 0.5
                horizontal_error = abs(radar.u - box_center_u) / max(box_width * 0.5 + margin, 1)
                horizontal_score = max(0.0, 1.0 - horizontal_error)
                depth_score = 0.45 if not depth_valid else max(0.0, 1.0 - depth_diff / depth_gate)
                rcs_score = min(max((radar.rcs - self.min_rcs) / 35.0, 0.0), 1.0)
                score = (
                    0.48 * horizontal_score
                    + 0.32 * depth_score
                    + 0.15 * yolo.confidence
                    + 0.05 * rcs_score
                )
                candidates.append((score, depth_diff, radar, yolo))

        candidates.sort(key=lambda item: item[0], reverse=True)
        used_radars = set()
        used_yolos = set()
        matches = []
        for score, depth_diff, radar, yolo in candidates:
            if radar.index in used_radars or yolo.index in used_yolos:
                continue
            fused_x, fused_y, fused_range, fused_azimuth, camera_weight = self.fuse_position(radar, yolo)
            track_id = self.assign_track(
                radar,
                yolo,
                current_ts,
                fused_x,
                fused_y,
                fused_range,
                fused_azimuth,
            )
            track = self.tracks[track_id]
            match = FusionMatch(
                radar=radar,
                yolo=yolo,
                score=score,
                depth_diff_m=depth_diff,
                track_id=track_id,
                x_forward=fused_x,
                y_left=fused_y,
                range_m=fused_range,
                azimuth_rad=fused_azimuth,
                camera_weight=camera_weight,
                vrel_x=float(track.get("vx", 0.0)),
                vrel_y=float(track.get("vy", 0.0)),
            )
            matches.append(match)
            used_radars.add(radar.index)
            used_yolos.add(yolo.index)
            self.recent_matches.append(
                {
                    "ts": current_ts,
                    "u": radar.u,
                    "v": yolo.center[1],
                    "range_m": fused_range,
                    "class_name": yolo.class_name,
                    "track_id": track_id,
                    "score": score,
                    "camera_weight": camera_weight,
                }
            )

        self.drop_stale_tracks(current_ts)
        return matches

    def camera_position_from_yolo(self, yolo):
        if yolo.depth_m <= 0.2 or self.fx is None:
            return None

        x_forward = yolo.depth_m
        y_left = ((yolo.center[0] - self.cx) * yolo.depth_m / self.fx) - self.radar_lateral_offset_m
        return x_forward, y_left

    def fuse_position(self, radar, yolo):
        camera_position = self.camera_position_from_yolo(yolo)
        camera_weight = self.camera_position_weight if camera_position is not None else 0.0
        radar_weight = 1.0 - camera_weight

        if camera_position is None:
            fused_x = radar.x_forward
            fused_y = radar.y_left
        else:
            camera_x, camera_y = camera_position
            fused_x = radar_weight * radar.x_forward + camera_weight * camera_x
            fused_y = radar_weight * radar.y_left + camera_weight * camera_y

        fused_range = math.hypot(fused_x, fused_y)
        fused_azimuth = math.atan2(fused_y, fused_x) if fused_range > 0.001 else radar.azimuth_rad
        return fused_x, fused_y, fused_range, fused_azimuth, camera_weight

    def assign_track(self, radar, yolo, current_ts, x_forward, y_left, range_m, azimuth_rad):
        best_id = None
        best_cost = float("inf")
        for track_id, track in self.tracks.items():
            if track["class_name"] != yolo.class_name:
                continue
            age = current_ts - track["ts"]
            if age > 1.0:
                continue

            dx = x_forward - track["x_forward"]
            dy = y_left - track["y_left"]
            spatial_cost = math.hypot(dx, dy)
            pixel_cost = abs(radar.u - track["u"]) / 220.0
            cost = spatial_cost + pixel_cost
            if cost < best_cost and spatial_cost < 3.0:
                best_cost = cost
                best_id = track_id

        if best_id is None:
            best_id = self.next_track_id
            self.next_track_id += 1

        old = self.tracks.get(best_id)
        position_alpha = 0.65
        if old is None:
            vx = radar.doppler_velocity * math.cos(azimuth_rad)
            vy = radar.doppler_velocity * math.sin(azimuth_rad)
            self.tracks[best_id] = {
                "class_name": yolo.class_name,
                "x_forward": x_forward,
                "y_left": y_left,
                "range_m": range_m,
                "azimuth_rad": azimuth_rad,
                "u": radar.u,
                "vx": vx,
                "vy": vy,
                "ts": current_ts,
            }
        else:
            dt = current_ts - old["ts"]
            if dt > 0.03:
                raw_vx = (x_forward - old["x_forward"]) / dt
                raw_vy = (y_left - old["y_left"]) / dt
                if math.hypot(raw_vx, raw_vy) < 50.0:
                    old["vx"] = self.velocity_alpha * raw_vx + (1.0 - self.velocity_alpha) * old.get("vx", 0.0)
                    old["vy"] = self.velocity_alpha * raw_vy + (1.0 - self.velocity_alpha) * old.get("vy", 0.0)

            old["x_forward"] = position_alpha * x_forward + (1.0 - position_alpha) * old["x_forward"]
            old["y_left"] = position_alpha * y_left + (1.0 - position_alpha) * old["y_left"]
            old["range_m"] = position_alpha * range_m + (1.0 - position_alpha) * old["range_m"]
            old["azimuth_rad"] = position_alpha * azimuth_rad + (1.0 - position_alpha) * old["azimuth_rad"]
            old["u"] = int(position_alpha * radar.u + (1.0 - position_alpha) * old["u"])
            old["ts"] = current_ts
        return best_id

    def drop_stale_tracks(self, current_ts):
        stale_ids = [track_id for track_id, track in self.tracks.items() if current_ts - track["ts"] > 1.5]
        for track_id in stale_ids:
            del self.tracks[track_id]

    def publish_detection(self, match):
        msg = Detection()
        msg.header.stamp = self.get_clock().now().to_msg()
        self.safe_set(msg, "track_id", match.track_id)
        self.safe_set(msg, "class_name", match.yolo.class_name)
        self.safe_set(msg, "confidence", match.yolo.confidence)
        self.safe_set(msg, "x", match.x_forward)
        self.safe_set(msg, "y", match.y_left)
        self.safe_set(msg, "range", match.range_m)
        self.safe_set(msg, "azimuth", match.azimuth_rad)
        self.safe_set(msg, "rcs", match.radar.rcs)
        self.safe_set(msg, "vrel_x", match.vrel_x)
        self.safe_set(msg, "vrel_y", match.vrel_y)
        self.detection_pub.publish(msg)

    @staticmethod
    def safe_set(msg, field_name, value):
        if hasattr(msg, field_name):
            setattr(msg, field_name, value)

    def draw_debug_views(self, color_img, radar_points, matches, current_ts, radar_dt, ran_yolo):
        if self.show_radar_debug:
            radar_img = self.draw_radar_debug(radar_points, matches, radar_dt)
            self.show_image("Radar sample debug", radar_img, scale=1.0)

        vision_panels = []
        if self.show_yolo_debug:
            vision_panels.append(self.draw_yolo_debug(color_img.copy(), current_ts, ran_yolo))
        if self.show_fusion_debug:
            vision_panels.append(self.draw_fusion_debug(color_img.copy(), radar_points, matches, current_ts, radar_dt))
        if vision_panels:
            vision_debug = self.combine_panels(vision_panels)
            self.show_image(self.fusion_window_name, vision_debug)

        if self.display_available:
            try:
                cv2.waitKey(1)
            except cv2.error as exc:
                self.disable_display(exc)

    @staticmethod
    def combine_panels(panels):
        if len(panels) == 1:
            return panels[0]

        target_h = min(panel.shape[0] for panel in panels)
        resized = []
        for panel in panels:
            if panel.shape[0] != target_h:
                scale = target_h / panel.shape[0]
                panel = cv2.resize(panel, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            resized.append(panel)

        divider = np.full((target_h, 6, 3), (35, 42, 48), dtype=np.uint8)
        output = resized[0]
        for panel in resized[1:]:
            output = np.hstack((output, divider, panel))
        return output

    def draw_yolo_debug(self, img, current_ts, ran_yolo):
        for det in self.yolo_detections:
            color = (0, 220, 0) if det.class_name == "person" else (0, 180, 255)
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"{det.class_name} {det.confidence:.2f} depth:{det.depth_m:.1f}m"
            self.put_label(img, label, (x1, max(18, y1 - 8)), color)

        cache_age = current_ts - self.last_yolo_ts
        mode = "new inference" if ran_yolo else f"cached {cache_age:.2f}s"
        freshness = "stale for fusion" if self.yolo_cache_stale else "fresh for fusion"
        self.draw_hud(
            img,
            [
                "YOLO debug: person + car only",
                f"boxes: {len(self.yolo_detections)} | {mode}",
                f"{freshness} | inference: {self.last_yolo_runtime_ms:.1f} ms | imgsz: {self.yolo_imgsz}",
            ],
        )
        return img

    def draw_fusion_debug(self, img, radar_points, matches, current_ts, radar_dt):
        matched_radar_ids = {match.radar.index for match in matches}
        matched_yolo_ids = {match.yolo.index for match in matches}

        for det in self.yolo_detections:
            color = (0, 160, 0) if det.index in matched_yolo_ids else (70, 190, 70)
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        for radar in radar_points:
            if not radar.in_image:
                continue
            if radar.index in matched_radar_ids:
                color = (0, 0, 255)
                radius = 7
            elif radar.in_azimuth_gate and radar.in_rcs_gate:
                color = (255, 80, 0)
                radius = 4
            else:
                color = (130, 130, 130)
                radius = 3
            cv2.circle(img, (radar.u, radar.v), radius, color, -1)

        live_recent = []
        for item in self.recent_matches:
            if current_ts - item["ts"] <= 0.6:
                live_recent.append(item)
                cv2.circle(img, (item["u"], item["v"]), 10, (0, 0, 255), 2)
                camera_pct = int(round(item["camera_weight"] * 100.0))
                label = f"#{item['track_id']} {item['class_name']} R:{item['range_m']:.1f} C:{camera_pct}% S:{item['score']:.2f}"
                self.put_label(img, label, (item["u"] + 12, item["v"] + 6), (0, 0, 255))
        self.recent_matches = deque(live_recent, maxlen=80)

        radar_text = "no radar"
        if radar_dt != float("inf"):
            radar_text = f"radar dt: {radar_dt * 1000.0:.0f} ms"
        self.draw_hud(
            img,
            [
                "Fusion debug",
                f"YOLO boxes: {len(self.yolo_detections)} | radar returns: {len(radar_points)} | matches: {len(matches)}",
                radar_text,
            ],
        )
        return img

    def draw_radar_debug(self, radar_points, matches, radar_dt):
        width, height = 900, 540
        map_width = 650
        panel_x = map_width + 12
        img = np.full((height, width, 3), (24, 32, 38), dtype=np.uint8)
        origin = (map_width // 2, height - 42)
        range_m = max(self.radar_debug_range_m, 1.0)
        lateral_m = max(self.radar_debug_lateral_m, 1.0)

        def world_to_px(y_left, x_forward):
            px = int(origin[0] + (y_left / lateral_m) * (map_width * 0.43))
            py = int(origin[1] - (x_forward / range_m) * (height * 0.84))
            return px, py

        matched_ids = {match.radar.index for match in matches}
        gated_points = [p for p in radar_points if p.in_azimuth_gate and p.in_rcs_gate]
        in_image_points = [p for p in gated_points if p.in_image]
        matched_points = [p for p in radar_points if p.index in matched_ids]

        sample_points = self.unique_radar_points(
            matched_points
            + sorted(gated_points, key=lambda p: p.rcs, reverse=True)
            + sorted(gated_points, key=lambda p: p.range_m)
        )
        label_ids = {p.index for p in sample_points[: max(0, self.radar_debug_max_labels)]}

        for r in range(10, int(range_m) + 1, 10):
            _, py = world_to_px(0.0, float(r))
            cv2.line(img, (36, py), (map_width - 26, py), (48, 62, 72), 1)
            cv2.putText(img, f"{r}m", (44, py - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 165, 175), 1)

        for lateral in range(-int(lateral_m), int(lateral_m) + 1, 10):
            px, _ = world_to_px(float(lateral), 0.0)
            cv2.line(img, (px, 40), (px, height - 36), (48, 62, 72), 1)
            cv2.putText(img, f"{lateral}", (px - 12, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 165, 175), 1)

        left_fov = world_to_px(math.sin(-self.max_azimuth_rad) * range_m, math.cos(-self.max_azimuth_rad) * range_m)
        right_fov = world_to_px(math.sin(self.max_azimuth_rad) * range_m, math.cos(self.max_azimuth_rad) * range_m)
        cv2.line(img, origin, left_fov, (70, 110, 150), 1)
        cv2.line(img, origin, right_fov, (70, 110, 150), 1)
        cv2.circle(img, origin, 5, (230, 230, 230), -1)

        for point in radar_points:
            px, py = world_to_px(point.y_left, point.x_forward)
            if not (0 <= px < map_width and 0 <= py < height):
                continue
            if point.index in matched_ids:
                color = (0, 0, 255)
                radius = 7
            elif point.in_azimuth_gate and point.in_rcs_gate:
                color = (255, 120, 40)
                radius = 5
            else:
                color = (100, 100, 100)
                radius = 4
            cv2.circle(img, (px, py), radius, color, -1)
            if point.index in label_ids:
                label = f"{point.index} R:{point.range_m:.1f} rcs:{point.rcs:.0f} v:{point.doppler_velocity:.1f}"
                cv2.putText(img, label, (px + 7, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

        radar_text = "no synchronized radar"
        if radar_dt != float("inf"):
            radar_text = f"closest radar dt: {radar_dt * 1000.0:.0f} ms"
        self.draw_hud(
            img,
            [
                "Radar debug top view",
                f"returns: {len(radar_points)} | matched: {len(matches)} | RCS gate: {self.min_rcs:.1f}",
                radar_text,
            ],
        )
        self.draw_radar_side_panel(
            img,
            panel_x,
            radar_points,
            gated_points,
            in_image_points,
            matched_points,
            sample_points,
            radar_text,
        )
        return img

    @staticmethod
    def unique_radar_points(points):
        seen = set()
        unique = []
        for point in points:
            if point.index in seen:
                continue
            seen.add(point.index)
            unique.append(point)
        return unique

    def draw_radar_side_panel(
        self,
        img,
        x,
        radar_points,
        gated_points,
        in_image_points,
        matched_points,
        sample_points,
        radar_text,
    ):
        cv2.rectangle(img, (x - 8, 10), (img.shape[1] - 10, img.shape[0] - 10), (18, 23, 28), -1)
        lines = [
            "Radar sample only",
            f"total returns: {len(radar_points)}",
            f"gate pass: {len(gated_points)}",
            f"in image: {len(in_image_points)}",
            f"matched: {len(matched_points)}",
            radar_text,
            "",
            "colors:",
            "red = fused match",
            "orange = valid radar",
            "gray = rejected",
            "",
            "sample rows:",
        ]

        y = 32
        for line in lines:
            color = (230, 238, 242) if line else (160, 170, 178)
            cv2.putText(img, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1, cv2.LINE_AA)
            y += 18

        for point in sample_points[: max(0, self.radar_debug_sample_rows)]:
            az_deg = math.degrees(point.azimuth_rad)
            line = (
                f"#{point.index:02d} R {point.range_m:4.1f}m "
                f"az {az_deg:5.1f} rcs {point.rcs:5.1f}"
            )
            color = (0, 0, 255) if any(match.index == point.index for match in matched_points) else (255, 160, 70)
            cv2.putText(img, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
            y += 17

    def show_image(self, window_name, img, scale=None):
        if not self.display_available:
            return
        if scale is None:
            scale = self.display_scale
        if scale > 0.0 and abs(scale - 1.0) > 0.01:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        try:
            cv2.imshow(window_name, img)
        except cv2.error as exc:
            self.disable_display(exc)

    def disable_display(self, exc):
        if self.display_available:
            self.get_logger().warning(
                f"OpenCV display disabled; continuing without debug windows: {exc}"
            )
        self.display_available = False
        self.debug_visuals = False

    @staticmethod
    def put_label(img, text, origin, color):
        x, y = origin
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.48
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        y = max(th + 6, y)
        cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y + baseline), (20, 24, 28), -1)
        cv2.putText(img, text, (x + 3, y - 3), font, scale, color, thickness, cv2.LINE_AA)

    @staticmethod
    def draw_hud(img, lines):
        pad = 8
        line_h = 20
        width = max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)[0][0] for line in lines)
        height = pad * 2 + line_h * len(lines)
        cv2.rectangle(img, (8, 8), (width + 2 * pad + 14, height + 8), (18, 23, 28), -1)
        for i, line in enumerate(lines):
            y = 8 + pad + 15 + i * line_h
            cv2.putText(img, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (230, 238, 242), 1, cv2.LINE_AA)

    def throttled_log(self, text, period_sec=2.0):
        now = time.monotonic()
        if now - self.last_status_log > period_sec:
            self.get_logger().info(text)
            self.last_status_log = now


class OfflineLogger:
    def info(self, text):
        print(f"[INFO] {text}")

    def warning(self, text):
        print(f"[WARN] {text}")


def build_arg_parser():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description="ROS radar + YOLO fusion node, with an optional offline fake-radar demo."
    )
    parser.add_argument(
        "--offline-demo",
        action="store_true",
        help="Run against a video file with fake YOLO depth and fake radar returns.",
    )
    parser.add_argument(
        "--input",
        default=os.path.join(script_dir, "car.mp4"),
        help="Input video for --offline-demo.",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(script_dir, "car_fake_radar_fusion_debug.mp4"),
        help="Output annotated video for --offline-demo.",
    )
    parser.add_argument(
        "--model-path",
        default=PersonCarRadarYOLOFusionNode.default_model_path(),
        help="YOLO .onnx or .pt model path for --offline-demo.",
    )
    parser.add_argument("--device", default="cpu", help="YOLO device for --offline-demo.")
    parser.add_argument("--imgsz", type=int, default=416, help="YOLO image size for --offline-demo.")
    parser.add_argument("--conf", type=float, default=0.35, help="YOLO confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="YOLO IoU threshold.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Limit processed frames in --offline-demo. 0 means full video.",
    )
    parser.add_argument(
        "--yolo-every",
        type=int,
        default=1,
        help="Run YOLO every N frames in --offline-demo; cached boxes are reused between runs.",
    )
    parser.add_argument(
        "--synthetic-clutter",
        type=int,
        default=8,
        help="Number of extra fake radar clutter returns per frame.",
    )
    parser.add_argument(
        "--synthetic-seed",
        type=int,
        default=7,
        help="Random seed for repeatable fake radar noise.",
    )
    parser.add_argument(
        "--camera-fov-deg",
        type=float,
        default=69.4,
        help="Approximate horizontal camera FOV used to fake camera intrinsics.",
    )
    parser.add_argument(
        "--realtime-playback",
        action="store_true",
        help="Throttle --offline-demo to the input video FPS so it behaves like a live camera.",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop the input video in --offline-demo until Esc/Ctrl+C or --max-frames is reached.",
    )
    parser.add_argument(
        "--no-output",
        action="store_true",
        help="Do not write an annotated video file in --offline-demo; useful for live display only.",
    )
    parser.add_argument("--show", action="store_true", help="Show OpenCV windows during --offline-demo.")
    return parser


def select_existing_model_path(model_path):
    onnx_candidates, pt_candidates, other_candidates = PersonCarRadarYOLOFusionNode.model_candidates(
        model_path
    )
    for candidate in onnx_candidates + pt_candidates + other_candidates:
        if os.path.isfile(candidate):
            return candidate
    tried = ", ".join(onnx_candidates + pt_candidates + other_candidates)
    raise FileNotFoundError(f"YOLO model not found. Tried: {tried}")


def offline_camera_intrinsics(width, height, horizontal_fov_deg):
    horizontal_fov_rad = math.radians(max(1.0, min(horizontal_fov_deg, 175.0)))
    fx = (width * 0.5) / math.tan(horizontal_fov_rad * 0.5)
    fy = fx
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5
    return fx, fy, cx, cy


def configure_offline_fusion(args, frame_width, frame_height):
    fusion = PersonCarRadarYOLOFusionNode.__new__(PersonCarRadarYOLOFusionNode)
    logger = OfflineLogger()
    fusion.get_logger = lambda: logger

    fusion.model_path = select_existing_model_path(args.model_path)
    fusion.device = args.device
    fusion.yolo_imgsz = int(args.imgsz)
    fusion.yolo_conf = float(args.conf)
    fusion.yolo_iou = float(args.iou)
    fusion.max_yolo_hz = 999.0
    fusion.min_yolo_period = 0.0
    fusion.max_cached_yolo_age_sec = 999.0
    fusion.max_azimuth_rad = math.radians(34.5)
    fusion.min_rcs = -25.0
    fusion.radar_match_slop_sec = 0.15
    fusion.radar_lateral_offset_m = 0.0
    fusion.depth_window_px = 11
    fusion.depth_gate_base_m = 1.6
    fusion.depth_gate_ratio = 0.16
    fusion.allow_missing_depth = True
    fusion.camera_position_weight = 0.5
    fusion.velocity_alpha = 0.35
    fusion.debug_visuals = True
    fusion.show_yolo_debug = True
    fusion.show_radar_debug = True
    fusion.show_fusion_debug = True
    fusion.display_scale = 0.5
    fusion.radar_debug_range_m = 50.0
    fusion.radar_debug_lateral_m = 25.0
    fusion.radar_debug_max_labels = 16
    fusion.radar_debug_sample_rows = 8
    fusion.fusion_window_name = "Offline Fake Radar Fusion"
    fusion.fx, fusion.fy, fusion.cx, fusion.cy = offline_camera_intrinsics(
        frame_width,
        frame_height,
        args.camera_fov_deg,
    )

    cv2.setUseOptimized(True)
    cv2.setNumThreads(1)
    if torch is not None:
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 2)))

    fusion.target_names = {"person", "car"}
    if ULTRALYTICS_AVAILABLE:
        fusion.model = YOLO(fusion.model_path, task="detect")
        fusion.target_class_ids = fusion.find_target_class_ids()
        fusion.offline_backend = "ultralytics"
    else:
        if not fusion.model_path.lower().endswith(".onnx"):
            raise RuntimeError(
                "Ultralytics is not installed, so --offline-demo needs an ONNX model. "
                f"Got: {fusion.model_path}"
            )
        fusion.model = cv2.dnn.readNetFromONNX(fusion.model_path)
        fusion.target_class_ids = [0, 2]
        fusion.offline_backend = "opencv_dnn"
    fusion.yolo_detections = []
    fusion.yolo_cache_stale = False
    fusion.last_yolo_ts = 0.0
    fusion.last_yolo_runtime_ms = 0.0
    fusion.last_radar_points = []
    fusion.recent_matches = deque(maxlen=80)
    fusion.tracks = {}
    fusion.next_track_id = 1
    fusion.frame_count = 0
    fusion.last_status_log = 0.0
    fusion.display_available = bool(args.show)

    print(
        "[INFO] Offline demo configured: "
        f"model={fusion.model_path}, imgsz={fusion.yolo_imgsz}, "
        f"backend={fusion.offline_backend}, fx={fusion.fx:.1f}, fy={fusion.fy:.1f}"
    )
    return fusion


def estimate_fake_depth_m(fusion, class_name, bbox):
    x1, y1, x2, y2 = bbox
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)

    if class_name == "person":
        depth_m = fusion.fy * 1.70 / box_h
    else:
        depth_from_width = fusion.fx * 1.85 / box_w
        depth_from_height = fusion.fy * 1.50 / box_h
        depth_m = 0.72 * depth_from_width + 0.28 * depth_from_height

    return float(np.clip(depth_m, 2.5, 55.0))


def run_offline_yolo(fusion, color_img, current_ts):
    if fusion.offline_backend == "opencv_dnn":
        return run_offline_yolo_opencv_dnn(fusion, color_img, current_ts)

    start = time.perf_counter()
    predict_kwargs = {
        "imgsz": fusion.yolo_imgsz,
        "conf": fusion.yolo_conf,
        "iou": fusion.yolo_iou,
        "device": fusion.device,
        "verbose": False,
    }
    if fusion.target_class_ids is not None:
        predict_kwargs["classes"] = fusion.target_class_ids

    result = fusion.model(color_img, **predict_kwargs)[0]
    h, w = color_img.shape[:2]
    detections = []
    for index, box in enumerate(result.boxes):
        class_id = int(box.cls[0])
        class_name = fusion.model.names[class_id]
        if class_name not in fusion.target_names:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x1, x2 = max(0, x1), min(w - 1, x2)
        y1, y2 = max(0, y1), min(h - 1, y2)
        c_x = int((x1 + x2) * 0.5)
        c_y = int((y1 + y2) * 0.5)
        bbox = (x1, y1, x2, y2)
        depth_m = estimate_fake_depth_m(fusion, class_name, bbox)
        detections.append(
            YoloDetection(
                bbox=bbox,
                class_name=class_name,
                confidence=float(box.conf[0]),
                depth_m=depth_m,
                center=(c_x, c_y),
                index=index,
            )
        )

    fusion.last_yolo_runtime_ms = (time.perf_counter() - start) * 1000.0
    fusion.last_yolo_ts = current_ts
    return detections


def run_offline_yolo_opencv_dnn(fusion, color_img, current_ts):
    start = time.perf_counter()
    h, w = color_img.shape[:2]
    size = int(fusion.yolo_imgsz)
    blob = cv2.dnn.blobFromImage(
        color_img,
        scalefactor=1.0 / 255.0,
        size=(size, size),
        mean=(0, 0, 0),
        swapRB=True,
        crop=False,
    )
    fusion.model.setInput(blob)
    output = fusion.model.forward(fusion.model.getUnconnectedOutLayersNames())[0]
    predictions = output[0].T

    class_scores = predictions[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    confidences = class_scores[np.arange(class_scores.shape[0]), class_ids]
    target_mask = np.isin(class_ids, fusion.target_class_ids) & (confidences >= fusion.yolo_conf)

    boxes = []
    kept_confidences = []
    kept_class_ids = []
    x_scale = w / float(size)
    y_scale = h / float(size)
    for row, class_id, confidence in zip(predictions[target_mask], class_ids[target_mask], confidences[target_mask]):
        cx, cy, box_w, box_h = row[:4]
        x1 = int((cx - box_w * 0.5) * x_scale)
        y1 = int((cy - box_h * 0.5) * y_scale)
        width = int(box_w * x_scale)
        height = int(box_h * y_scale)
        boxes.append([x1, y1, width, height])
        kept_confidences.append(float(confidence))
        kept_class_ids.append(int(class_id))

    nms_indices = cv2.dnn.NMSBoxes(boxes, kept_confidences, fusion.yolo_conf, fusion.yolo_iou)
    if len(nms_indices) == 0:
        fusion.last_yolo_runtime_ms = (time.perf_counter() - start) * 1000.0
        fusion.last_yolo_ts = current_ts
        return []

    detections = []
    for det_index, nms_index in enumerate(np.array(nms_indices).reshape(-1)):
        x1, y1, box_w, box_h = boxes[int(nms_index)]
        x2 = x1 + box_w
        y2 = y1 + box_h
        x1, x2 = max(0, x1), min(w - 1, x2)
        y1, y2 = max(0, y1), min(h - 1, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        class_id = kept_class_ids[int(nms_index)]
        class_name = "person" if class_id == 0 else "car"
        bbox = (x1, y1, x2, y2)
        c_x = int((x1 + x2) * 0.5)
        c_y = int((y1 + y2) * 0.5)
        detections.append(
            YoloDetection(
                bbox=bbox,
                class_name=class_name,
                confidence=kept_confidences[int(nms_index)],
                depth_m=estimate_fake_depth_m(fusion, class_name, bbox),
                center=(c_x, c_y),
                index=det_index,
            )
        )

    fusion.last_yolo_runtime_ms = (time.perf_counter() - start) * 1000.0
    fusion.last_yolo_ts = current_ts
    return detections


def make_synthetic_radar_scan(fusion, yolo_detections, current_ts, state, clutter_count):
    rng = state["rng"]
    returns = []

    for det in yolo_detections:
        camera_position = fusion.camera_position_from_yolo(det)
        if camera_position is None:
            continue

        x_forward, y_left = camera_position
        x_forward += float(rng.normal(0.0, max(0.05, 0.015 * x_forward)))
        y_left += float(rng.normal(0.0, max(0.04, 0.010 * x_forward)))
        range_m = max(0.2, math.hypot(x_forward, y_left))
        azimuth_rad = math.atan2(y_left, x_forward)

        key = (det.class_name, det.index)
        previous_range = state["prev_ranges"].get(key)
        previous_ts = state["prev_ts"].get(key)
        if previous_range is None or previous_ts is None or current_ts <= previous_ts:
            doppler_velocity = 0.0
        else:
            doppler_velocity = (range_m - previous_range) / max(1e-3, current_ts - previous_ts)
        state["prev_ranges"][key] = range_m
        state["prev_ts"][key] = current_ts

        base_rcs = 7.0 if det.class_name == "car" else -4.0
        rcs = float(base_rcs + rng.normal(0.0, 2.0))
        returns.append(
            SyntheticRadarReturn(
                range=range_m,
                azimuth=azimuth_rad,
                rcs=rcs,
                doppler_velocity=float(doppler_velocity),
            )
        )

    for _ in range(max(0, int(clutter_count))):
        range_m = float(rng.uniform(3.0, fusion.radar_debug_range_m))
        azimuth_rad = float(rng.uniform(-1.25 * fusion.max_azimuth_rad, 1.25 * fusion.max_azimuth_rad))
        rcs = float(rng.uniform(-38.0, -12.0))
        doppler_velocity = float(rng.normal(0.0, 1.2))
        returns.append(
            SyntheticRadarReturn(
                range=range_m,
                azimuth=azimuth_rad,
                rcs=rcs,
                doppler_velocity=doppler_velocity,
            )
        )

    return SyntheticRadarScan(returns=returns)


def ensure_even_video_frame(img):
    h, w = img.shape[:2]
    pad_bottom = h % 2
    pad_right = w % 2
    if pad_bottom or pad_right:
        img = cv2.copyMakeBorder(
            img,
            0,
            pad_bottom,
            0,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
    return img


def compose_offline_debug_frame(fusion, color_img, radar_points, matches, current_ts, ran_yolo):
    yolo_panel = fusion.draw_yolo_debug(color_img.copy(), current_ts, ran_yolo)
    fusion_panel = fusion.draw_fusion_debug(color_img.copy(), radar_points, matches, current_ts, 0.0)
    radar_panel = fusion.draw_radar_debug(radar_points, matches, 0.0)
    output = fusion.combine_panels([yolo_panel, fusion_panel, radar_panel])
    fusion.put_label(
        output,
        "OFFLINE TEST: fake YOLO depth + fake radar",
        (16, output.shape[0] - 18),
        (80, 220, 255),
    )
    return ensure_even_video_frame(output)


def run_offline_demo(args):
    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open input video: {input_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frame_width <= 0 or frame_height <= 0:
        raise RuntimeError(f"Cannot read video dimensions from: {input_path}")

    fusion = configure_offline_fusion(args, frame_width, frame_height)
    yolo_every = max(1, int(args.yolo_every))
    max_frames = max(0, int(args.max_frames))
    synthetic_state = {
        "rng": np.random.default_rng(int(args.synthetic_seed)),
        "prev_ranges": {},
        "prev_ts": {},
    }

    writer = None
    cached_detections = []
    frame_index = 0
    total_detections = 0
    total_matches = 0
    start = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if args.loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = cap.read()
                if not ok:
                    break
            if max_frames and frame_index >= max_frames:
                break

            current_ts = frame_index / fps
            if args.realtime_playback:
                target_time = start + current_ts
                wait_sec = target_time - time.perf_counter()
                if wait_sec > 0.0:
                    time.sleep(wait_sec)

            ran_yolo = frame_index % yolo_every == 0 or not cached_detections
            if ran_yolo:
                cached_detections = run_offline_yolo(fusion, frame, current_ts)

            fusion.yolo_detections = cached_detections
            fusion.yolo_cache_stale = False
            radar_scan = make_synthetic_radar_scan(
                fusion,
                cached_detections,
                current_ts,
                synthetic_state,
                args.synthetic_clutter,
            )
            radar_points = fusion.extract_radar_points(radar_scan, frame.shape)
            matches = fusion.associate(radar_points, cached_detections, current_ts)
            output_frame = compose_offline_debug_frame(
                fusion,
                frame,
                radar_points,
                matches,
                current_ts,
                ran_yolo,
            )

            if not args.no_output and writer is None:
                output_dir = os.path.dirname(output_path)
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(
                    output_path,
                    fourcc,
                    fps,
                    (output_frame.shape[1], output_frame.shape[0]),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Cannot open output video writer: {output_path}")

            if writer is not None:
                writer.write(output_frame)
            total_detections += len(cached_detections)
            total_matches += len(matches)

            if args.show:
                cv2.imshow(fusion.fusion_window_name, output_frame)
                if cv2.waitKey(1) == 27:
                    break

            frame_index += 1
            if frame_index % 30 == 0:
                elapsed = max(1e-6, time.perf_counter() - start)
                print(
                    f"[INFO] processed {frame_index} frames "
                    f"({frame_index / elapsed:.1f} fps), latest matches={len(matches)}"
                )
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

    if frame_index == 0:
        raise RuntimeError(f"No frames were processed from: {input_path}")

    print(
        "[INFO] Offline demo complete: "
        f"frames={frame_index}, detections/frame={total_detections / frame_index:.2f}, "
        f"matches/frame={total_matches / frame_index:.2f}"
    )
    if writer is None:
        print("[INFO] Output video disabled (--no-output).")
    else:
        print(f"[INFO] Wrote: {output_path}")


def main(args=None):
    parsed_args, ros_args = build_arg_parser().parse_known_args(args)
    if parsed_args.offline_demo:
        run_offline_demo(parsed_args)
        return

    if not ROS_AVAILABLE:
        raise RuntimeError(
            "ROS Python packages are not available. "
            "Run with --offline-demo to test this file using fake radar/depth data."
        )

    node = None
    try:
        rclpy.init(args=ros_args)
        node = PersonCarRadarYOLOFusionNode()
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
