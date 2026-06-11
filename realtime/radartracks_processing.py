import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from collections import deque
import cv2
import numpy as np
import math
from ultralytics import YOLO
from radar_msgs.msg import RadarScan
from my_detections.msg import Detection
from message_filters import Subscriber, ApproximateTimeSynchronizer

class RadarYOLOFusionNode(Node):
    def __init__(self):
        super().__init__('radar_yolo_fusion_node')

        self.fx = self.fy = self.cx = self.cy = None
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/camera/camera/color/camera_info', self.camera_info_callback, 10)
        
        self.MAX_AZIMUTH_RAD = math.radians(34.5) 
        self.bridge = CvBridge()
        self.model = YOLO("yolo12n.pt")
        self.target_classes = ["person", "car", "bus", "bicycle", "motorcycle", "truck"]

        self.radar_buffer = deque(maxlen=60)
        self.bboxes = []
        self.recent_fusions = []
        self.tracks = {}
        self.next_track_id = 1
        self.velocity_alpha = 0.35


        self.color_sub = Subscriber(self, Image, '/camera/camera/color/image_raw')
        self.depth_sub = Subscriber(self, Image, '/camera/camera/aligned_depth_to_color/image_raw')
        self.create_subscription(RadarScan, '/radar_scan', self.radar_tracks_callback, 100)
        
        self.detection_pub = self.create_publisher(Detection, "/detections", 20)

        self.ts = ApproximateTimeSynchronizer([self.color_sub, self.depth_sub], queue_size=30, slop=0.1)
        self.ts.registerCallback(self.synced_callback)

        self.prev_color_ts = None
        self.get_logger().info("Continental-Optimized Radar-YOLO node started")



    def camera_info_callback(self, msg):
        if self.fx is None:
            self.fx, self.cx = msg.k[0], msg.k[2]
            self.fy, self.cy = msg.k[4], msg.k[5]

    def timestamp(self, msg):
        return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def radar_tracks_callback(self, msg):
        self.radar_buffer.append((self.timestamp(msg), msg))

    def get_robust_depth(self, depth_img, c_x, c_y, window=10):
        """Extracts the median depth to ignore D435i infrared noise/holes."""
        h, w = depth_img.shape
        x_min, x_max = max(0, c_x - window // 2), min(w, c_x + window // 2)
        y_min, y_max = max(0, c_y - window // 2), min(h, c_y + window // 2)

        roi = depth_img[y_min:y_max, x_min:x_max]
        valid_pixels = roi[roi > 0] 
        
        if len(valid_pixels) == 0: return 0.0
        return float(np.median(valid_pixels)) / 1000.0

    def synced_callback(self, color_msg, depth_msg):
        current_ts = self.timestamp(color_msg)
        if self.prev_color_ts is None:
            self.prev_color_ts = current_ts
            return

        color_img = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
        depth_img = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

        results = self.model(color_img, verbose=False)[0]
        self.bboxes = []
        
        for box in results.boxes:
            class_name = self.model.names[int(box.cls[0])]
            if class_name not in self.target_classes:
                continue
                
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            c_x, c_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
            
            depth_m = self.get_robust_depth(depth_img, c_x, c_y)

            self.bboxes.append({
                'bbox': (x1, y1, x2, y2),
                'class_name': class_name,
                'confidence': float(box.conf[0]),
                'rs_depth': depth_m
            })

            cv2.rectangle(color_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(color_img, f"{class_name} {depth_m:.1f}m", (x1, y1 - 8), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        while self.radar_buffer and current_ts - self.radar_buffer[0][0] > 1.0:
            self.radar_buffer.popleft()

        closest_msg = None
        min_time_diff = 1.0 
        for ts, msg in self.radar_buffer:
            diff = abs(current_ts - ts)
            if diff < min_time_diff:
                min_time_diff = diff
                closest_msg = msg

        if closest_msg is not None and min_time_diff < 0.15:
            color_img = self.fuse_and_project(closest_msg, color_img, current_ts)

        valid_fusions = []
        for fusion in self.recent_fusions:
            # Keep the red dot alive for 0.5 seconds (500ms)
            if current_ts - fusion['ts'] < 0.5: 
                cv2.circle(color_img, (fusion['u'], fusion['c_y']), 10, (0, 0, 255), -1)
                speed = math.hypot(fusion['vx'], fusion['vy'])
                cv2.putText(color_img, f"#{fusion['track_id']} R:{fusion['range']:.1f}m V:{speed:.1f}m/s", (fusion['u'] + 15, fusion['c_y'] + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                valid_fusions.append(fusion)
        
        self.recent_fusions = valid_fusions # Clear out expired memory
        
        self.prev_color_ts = current_ts
        cv2.imshow("Fusion Pipeline", color_img)
        cv2.waitKey(1)

    def fuse_and_project(self, r_scan_msg, color_img, frame_ts):
        if self.fx is None: return color_img
        
        RADAR_LATERAL_OFFSET = 0.0  
        radar_ts = self.timestamp(r_scan_msg)

        for r_msg in r_scan_msg.returns:
            az_rad = r_msg.azimuth
            
            # 1. Lower the RCS threshold to -25.0 to survive indoor fading!
            if abs(az_rad) > self.MAX_AZIMUTH_RAD or r_msg.rcs < -25.0:
                continue 

            x_r = r_msg.range * math.cos(az_rad)
            y_r = r_msg.range * math.sin(az_rad)
            
            pt_cam = np.array([y_r + RADAR_LATERAL_OFFSET, 0, x_r])

            if pt_cam[2] <= 0.1: continue

            u = int(self.fx * pt_cam[0] / pt_cam[2] + self.cx)
            v = int(self.fy * pt_cam[1] / pt_cam[2] + self.cy)

            if not (0 <= u < color_img.shape[1]): continue
            
            # Draw Raw Radar (Blue) on the horizon line
            cv2.circle(color_img, (u, v), 4, (255, 0, 0), -1) 

            best_match = None
            min_depth_diff = float('inf')

            for obj in self.bboxes:
                x1, y1, x2, y2 = obj['bbox']
                
                if x1 <= u <= x2:
                    rs_depth = obj['rs_depth']
                    radar_depth = r_msg.range
                    
                    diff = abs(rs_depth - radar_depth)
                    # Relaxed depth tolerance to 3.0m for indoor multipath ghosting
                    if rs_depth == 0.0 or rs_depth > 5.0 or diff < 3.0:
                        if diff < min_depth_diff:
                            min_depth_diff = diff
                            best_match = obj

            if best_match:
                x1, y1, x2, y2 = best_match['bbox']
                c_y = int((y1 + y2) / 2) 
                doppler_velocity = float(getattr(r_msg, "doppler_velocity", 0.0))
                track_id, est_vx, est_vy = self.update_track(
                    best_match['class_name'],
                    x_r,
                    y_r,
                    r_msg.range,
                    az_rad,
                    u,
                    radar_ts,
                    doppler_velocity
                )
                
                # --- NEW: Save to memory instead of drawing instantly ---
                self.recent_fusions.append({
                    'ts': frame_ts,
                    'u': u,
                    'c_y': c_y,
                    'range': r_msg.range,
                    'track_id': track_id,
                    'vx': est_vx,
                    'vy': est_vy
                })

                # Publish final data
                det_msg = Detection()
                det_msg.header.stamp = self.get_clock().now().to_msg()
                self.safe_set(det_msg, "track_id", track_id)
                self.safe_set(det_msg, "class_name", best_match['class_name'])
                self.safe_set(det_msg, "confidence", best_match['confidence'])
                self.safe_set(det_msg, "x", x_r)
                self.safe_set(det_msg, "y", y_r)
                self.safe_set(det_msg, "range", r_msg.range)
                self.safe_set(det_msg, "azimuth", az_rad)
                self.safe_set(det_msg, "rcs", r_msg.rcs)
                self.safe_set(det_msg, "vrel_x", est_vx)
                self.safe_set(det_msg, "vrel_y", est_vy)
                self.detection_pub.publish(det_msg)

        self.drop_stale_tracks(frame_ts)
        return color_img

    def update_track(self, class_name, x_r, y_r, range_m, az_rad, u, measurement_ts, doppler_velocity):
        best_id = None
        best_cost = float('inf')
        for track_id, track in self.tracks.items():
            if track['class_name'] != class_name:
                continue
            age = measurement_ts - track['ts']
            if age < 0.0 or age > 1.0:
                continue

            spatial_cost = math.hypot(x_r - track['x'], y_r - track['y'])
            pixel_cost = abs(u - track['u']) / 220.0
            cost = spatial_cost + pixel_cost
            if spatial_cost < 3.0 and cost < best_cost:
                best_cost = cost
                best_id = track_id

        if best_id is None:
            best_id = self.next_track_id
            self.next_track_id += 1
            vx = doppler_velocity * math.cos(az_rad)
            vy = doppler_velocity * math.sin(az_rad)
            self.tracks[best_id] = {
                'class_name': class_name,
                'x': x_r,
                'y': y_r,
                'range': range_m,
                'azimuth': az_rad,
                'u': u,
                'vx': vx,
                'vy': vy,
                'ts': measurement_ts,
            }
            return best_id, vx, vy

        track = self.tracks[best_id]
        dt = measurement_ts - track['ts']
        if dt > 0.03:
            raw_vx = (x_r - track['x']) / dt
            raw_vy = (y_r - track['y']) / dt
            if math.hypot(raw_vx, raw_vy) < 50.0:
                alpha = self.velocity_alpha
                track['vx'] = alpha * raw_vx + (1.0 - alpha) * track['vx']
                track['vy'] = alpha * raw_vy + (1.0 - alpha) * track['vy']

        track['x'] = x_r
        track['y'] = y_r
        track['range'] = range_m
        track['azimuth'] = az_rad
        track['u'] = u
        track['ts'] = measurement_ts
        return best_id, track['vx'], track['vy']

    def drop_stale_tracks(self, current_ts):
        stale_ids = [track_id for track_id, track in self.tracks.items() if current_ts - track['ts'] > 1.5]
        for track_id in stale_ids:
            del self.tracks[track_id]

    @staticmethod
    def safe_set(msg, field_name, value):
        if hasattr(msg, field_name):
            setattr(msg, field_name, value)
    
def main(args=None):
    rclpy.init(args=args)
    node = RadarYOLOFusionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
