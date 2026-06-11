#node to do the detection and projection together on rosbag with filtered tracks data , radar buffer update
#This node fuses the radar and camera detections and publishes the detections as "/detections"
import rclpy                                                                # ROS2 Python client library 
from rclpy.node import Node                                                 # Base class for creating ROS2 nodes
from sensor_msgs.msg import Image                                           # ROS2 message type for camera images
from cv_bridge import CvBridge                                              # Bridge to convert ROS images to OpenCV images
from collections import deque                                               # Efficient double-ended queue for buffering messages
import cv2                                                                  # OpenCV for image processing and display
import numpy as np                                                          # NumPy for numerical operations
import math                                                                 # Math library for basic math functions
from ultralytics import YOLO                                                # YOLO object detection library
import matplotlib.pyplot as plt                                             # Matplotlib for plotting graphs
from my_msgs.msg import CustomRadarTrack                                    # User-defined custom message type for radar tracks data (must be defined in my_msgs package)
from my_detections.msg import Detection                                     #used to publish fused YOLO + radar detection results.Custom message type "Detection" from my_detections package.
from message_filters import Subscriber,ApproximateTimeSynchronizer          #synchronizes multiple topics

class RadarYOLOFusionNode(Node):
    def __init__(self):
        super().__init__('radar_yolo_fusion_node')

        # Camera intrinsics
        self.fx = 916.357
        self.fy = 916.645
        self.cx = 630.561
        self.cy = 367.721

        # CV bridge
        self.bridge = CvBridge()

        # YOLO model
        self.model = YOLO("yolo12n.pt")
        self.target_classes = ["person", "car", "bus", "bicycle", "motorcycle", "truck", "traffic light", "stop sign", "train", "parking meter"]

        # Buffers
        self.color_buffer = deque(maxlen=60)
        self.depth_buffer = deque(maxlen=60)
        self.radar_buffer = deque(maxlen=200)

        self.bboxes = []

        # Subscriptions
        self.color_sub= Subscriber(self,Image, '/camera/camera/color/image_raw')
        self.depth_sub= Subscriber(self,Image, '/camera/camera/aligned_depth_to_color/image_raw')
        self.create_subscription(CustomRadarTrack, '/radar_track', self.radar_tracks_callback, 100)
        self.detection_pub = self.create_publisher(Detection, "/detections", 20)
        self.get_logger().info("Radar-YOLO node started")

        # synchronize between color and depth images
        self.ts = ApproximateTimeSynchronizer([self.color_sub, self.depth_sub],queue_size=60,
            slop=0.1,  # Tolerance in seconds
            allow_headerless=False
        )
        self.ts.registerCallback(self.synced_callback)

        # Radar-only live plot
        self.fig_radar, self.ax_radar = plt.subplots(figsize=(8, 6))
        self.sc_radar = self.ax_radar.scatter([], [], s=40, c=[], cmap='jet')
        self.cbar_radar = self.fig_radar.colorbar(self.sc_radar, ax=self.ax_radar)
        self.cbar_radar.set_label("RCS (dBm²)")
        self.ax_radar.set_xlim(-30,30)
        self.ax_radar.set_ylim(0, 50)
        self.ax_radar.set_xlabel('Left / Right (m)')
        self.ax_radar.set_ylabel('Forward (m)')
        self.ax_radar.set_title('Radar-Only Live Points')
        self.ax_radar.grid(True)
        plt.ion()
        plt.show()

        self.color_img_ts = None
        self.prev_color_ts = None

        # Periodic timer to update the plot
        self.create_timer(0.1, self.update_radar_only_plot)

        self.tracked_objects = {}  # to store the tracked objects within the bounding box

        self.projected_points = {} # to store the points to be projected filtered based on timestamp

        self.MIN_LIFETIME = 10  # Minimum lifetime to consider a radar track as stable

        self.trails_radar = {}  # key: track_id, value: deque of (y_r, x_r)
        self.TRAIL_LENGTH = 10  # number of points in the trail

    def timestamp(self, msg):
        return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    # Call back to acquire radar data into a buffer
    def radar_tracks_callback(self, radar_tracks_msg):
        ts = self.timestamp(radar_tracks_msg)
        self.radar_buffer.append((ts, radar_tracks_msg))
    # Call back to do the detection and initiate the projection
    def synced_callback(self, color_msg,depth_msg):
        current_ts = self.timestamp(color_msg)
        self.color_img_ts = current_ts

        if self.prev_color_ts is None:
            self.prev_color_ts = current_ts
            return

        # Convert images
        color_img = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='rgb8')
        color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
        depth_img = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

        # Run YOLO detection
        results = self.model(color_img)[0]
        self.bboxes = []
        for box in results.boxes:
            class_id = int(box.cls[0])
            class_name = self.model.names[class_id]
            # Only consider target classes
            if class_name not in self.target_classes:
                continue
            # Bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            self.bboxes.append({
                'bbox': (x1, y1, x2, y2), # to check radar points inside bbox
                'class_name': class_name,
                'confidence': float(box.conf[0])
            })
            # Get depth at the bounding box center
            c_x = int((x1 + x2) / 2)
            c_y = int((y1 + y2) / 2)

            if 0 <= c_y < depth_img.shape[0] and 0 <= c_x < depth_img.shape[1]:
                raw_depth = depth_img[c_y, c_x]
                depth_m = float(raw_depth) / 1000.0
            else:
                depth_m = 0.0
          
            # Draw bounding box and label
            label = f"{class_name}({depth_m:.2f}m)"
            cv2.rectangle(color_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(color_img, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        # Filter radar messages between timestamps
        radar_msgs_to_project = []
        for ts,msg in self.radar_buffer:
            if self.prev_color_ts <= ts <= current_ts:
                radar_msgs_to_project.append(msg)

        for radar_msg in radar_msgs_to_project:
            # Project radar
            color_img = self.project_radar(radar_msg, color_img,current_ts)

        # Update timestamp
        self.prev_color_ts = current_ts
        # Display
        cv2.imshow("YOLO + Radar Projection", color_img)
        cv2.waitKey(1)
    # Callback to project radar detetcions into camera image
    def project_radar(self, radar_tracks_msg, color_img,color_img_ts):
        radar_timestamp = self.timestamp(radar_tracks_msg)      

        if radar_tracks_msg.rcs <= -90 : #or radar_tracks_msg.lifetime < self.MIN_LIFETIME: # to filter highly negative rcs and points that does not stay long in the field
            return color_img

        r = radar_tracks_msg.range
        az = math.radians(radar_tracks_msg.azimuth)
        rcs  = radar_tracks_msg.rcs
        vrel_x = radar_tracks_msg.vrel_x
        vrel_y = radar_tracks_msg.vrel_y
        track_id = radar_tracks_msg.track_id
        lifetime = radar_tracks_msg.lifetime
        # Convert radar points from polar to Cartesian coordinates
        x_r = r * math.cos(az)
        y_r = r * math.sin(az)       
        # Convert radar point to camera frame
        pt_cam = np.array([y_r, 0, x_r])

        if pt_cam[2] <= 0:
            return color_img
        # Project to image plane using camera intrinsics
        u = int(self.fx * pt_cam[0] / pt_cam[2] + self.cx)
        v = int(self.fy * pt_cam[1] / pt_cam[2] + self.cy)

        if not (0 <= u < color_img.shape[1] and 0 <= v < color_img.shape[0]):
            return color_img
        #Check if radar point falls inside YOLO bbox
        for detection in self.bboxes:
            x1, y1, x2, y2 = detection['bbox']
            if x1 <= u <= x2 and y1 <= v <= y2:
                class_name = detection['class_name']
                confidence = detection['confidence']
                self.projected_points[track_id] = {
                'u': u,
                'v': v,
                'rcs': rcs,
                'timestamp': radar_timestamp,
                'class_name': class_name,
                'confidence': confidence
                }   
                self.tracked_objects[track_id] = {
                    'x_r' : x_r,
                    'y_r' : y_r,
                    'rcs' : rcs,
                    'range' : r,
                    'azimuth_rad' : az, 
                    'vrel_x' : vrel_x,
                    'vrel_y' : vrel_y,
                    'track_id' : track_id,
                    'lifetime' : lifetime,
                    'radar_ts' : radar_timestamp,
                    'class_name': class_name,
                    'confidence': confidence
                }
                #break

        expired_ids = []   # to store the points that are out of the 1 sec window
        #Draw projected points
        for track_id, point in self.projected_points.items():
            if color_img_ts - point['timestamp'] > 1.0:
                expired_ids.append(track_id)
                continue

            u = point['u']
            v = point['v']
            rcs = point['rcs']

            # Normalize RCS for coloring
            rcs = min(rcs, 20.0) / 20.0
            color = plt.get_cmap("jet")(rcs)
            bgr = tuple(int(c * 255) for c in color[:3])[::-1]
            cv2.circle(color_img, (u, v), 4, bgr, -1)
            obj = self.tracked_objects.get(track_id, None)
            if obj : 

                label1 = f"{track_id},{obj['rcs']:.1f},{obj['x_r']:.1f}"
                cv2.putText(color_img,label1, (u + 5, v - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, bgr, 1)
                # Highlight fast-moving objects
                # Calculate speed from vrel_x and vrel_y
                speed = math.sqrt(obj['vrel_x']**2 + obj['vrel_y']**2)
                if speed > 1:  # threshold
                    box_size = 10
                    top_left = (max(u - box_size, 0), max(v - box_size, 0))
                    bottom_right = (min(u + box_size, color_img.shape[1] - 1), min(v + box_size, color_img.shape[0] - 1))
                    cv2.rectangle(color_img, top_left, bottom_right, (0, 0, 255), 2)
                     # Maintain trails
                    if track_id not in self.trails_radar:
                        self.trails_radar[track_id] = deque(maxlen=self.TRAIL_LENGTH)
                    y_r = obj['y_r']
                    x_r = obj['x_r']
                    self.trails_radar[track_id].append((y_r, x_r))

        # for tid in expired_ids:
            # self.projected_points.pop(tid,None)
            #self.tracked_objects.pop(tid,None)
            

        return color_img
    #Update radar-only scatter plot
    def update_radar_only_plot(self):

        if not self.tracked_objects or self.color_img_ts is None:
            return
        
        # Extract radar track data
        xs = [obj['x_r'] for obj in self.tracked_objects.values()]
        ys = [obj['y_r'] for obj in self.tracked_objects.values()]
        rcs = [obj['rcs'] for obj in self.tracked_objects.values()]
        track_id = [obj['track_id'] for obj in self.tracked_objects.values()]
        lifetimes = [obj['lifetime'] for obj in self.tracked_objects.values()]
        vrel_x = [obj['vrel_x'] for obj in self.tracked_objects.values()]
        vrel_y = [obj['vrel_y'] for obj in self.tracked_objects.values()]

        # Update scatter points
        points = np.column_stack((ys, xs))
        self.sc_radar.set_offsets(points)
        self.sc_radar.set_array(np.array(rcs))

        self.sc_radar.set_clim(vmin=min(rcs), vmax=max(rcs))
        # Clear previous annotations
        self.ax_radar.texts.clear()
        self.ax_radar.patches.clear()
        #self.ax_radar.lines.clear()
        # Add text + boxes for tracked objects
        for obj in self.tracked_objects.values():
            x = obj['x_r']
            y = obj['y_r']
            r = obj['rcs']
            t_id = obj['track_id']
            vrel_x = obj['vrel_x']
            vrel_y = obj['vrel_y']
            class_name = obj.get('class_name', 'unknown')
            confidence = obj.get('confidence', 0.0)
            range_m = obj['range']
            az_rad = obj['azimuth_rad']
            lifetime = obj['lifetime']
            label = f"{class_name} {t_id} ({confidence:.2f})"
            # Publish Detection message
            detection_msg = Detection()
            detection_msg.header.stamp = self.get_clock().now().to_msg()
            detection_msg.header.frame_id = 'detection_link'
            detection_msg.track_id = t_id
            detection_msg.class_name = class_name
            detection_msg.confidence = confidence
            detection_msg.x = x
            detection_msg.y = y
            detection_msg.range = range_m
            detection_msg.azimuth = az_rad
            detection_msg.rcs = r 
            detection_msg.vrel_x = vrel_x
            detection_msg.vrel_y = vrel_y
            detection_msg.lifetime = lifetime
            self.detection_pub.publish(detection_msg)

            # Draw label on radar plot
            self.ax_radar.text(y + 0.2, x + 0.2,label, fontsize=8, color='black', zorder=5)
            # Highlight moving objects in radar-only view
            speed = math.hypot(vrel_x, vrel_y)  # same as sqrt(vx^2 + vy^2)
            if speed > 1:  # threshold
                box_size = 2
                rect = plt.Rectangle((y - box_size/2, x - box_size/2),
                                    box_size, box_size,
                                    linewidth=1.5, edgecolor='red', facecolor='none', zorder=4)
                self.ax_radar.add_patch(rect)   

            for tid, points in self.trails_radar.items():
                if len(points) > 1:
                    ys = [p[0] for p in points]
                    xs = [p[1] for p in points]
                    #self.ax_radar.plot(ys, xs, color='red', linewidth=1.5, alpha=0.7)
        # Refresh figure
        self.fig_radar.canvas.draw()
        self.fig_radar.canvas.flush_events()  
        # Reset for next cycle
        self.projected_points = {}
        self.tracked_objects = {}
        
def main(args=None):
    rclpy.init(args=args)
    node = RadarYOLOFusionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

# ROS2 node entry point 
if __name__ == '__main__':
    main()