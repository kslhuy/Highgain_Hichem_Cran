
# radar node to capture tracks data as "radar_track" by using CustomRadarTrack message from my_msgs
import rclpy                                            # ROS2 Python client library
from rclpy.node import Node                             # Base class for creating ROS2 nodes
from std_msgs.msg import String                         # Standard ROS2 string message type
import can                                              # Python-CAN library for reading/writing CAN bus messages
import matplotlib.pyplot as plt                         # Matplotlib for plotting
from time import time                                   # Standard Python function to use time
import datetime                                         # Standard Python module for date/time usage
from radar_msgs.msg import RadarReturn, RadarScan       # Custom ROS2 messages for radar data (ensure radar_msgs installed)
import math                                             # Math library for basic math functions
from my_msgs.msg import CustomRadarTrack                # User-defined custom message type for radar tracks data (must be defined in my_msgs package)


class RadarDriverNode(Node):
    def __init__(self):
        super().__init__('radar_driver_node')
        # CAN bus configuration
        self.interface = 'can0'
        self.bustype = 'socketcan'
        self.bitrate = 500000 # 500 kbps
        #self.conf = [0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]   #cluster
        self.conf = [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01]    #tracks , Output type valid bit set
        self.conf_id = 0x200  # Sensor Configuration
        self.cluster_id = 0x70B  # Cluster Status
        self.radar_cluster_id = 0x70C  # Radar Data
        self.track_id = 0x60B  # Track Status
        self.radar_track_1_id = 0x60C  # Track_1 Information
        self.radar_track_2_id = 0x60D  # Track_2 Information
        # Counters
        self.cluster_count = 0
        self.track_count = 0
        self.mode = 1 # 1 for clusters, 0 for tracks
        self.t0 = None
        # Initialize CAN bus
        self.bus = can.interface.Bus(channel=self.interface, interface=self.bustype, bitrate=self.bitrate)
        self.change_mode(0) # to set the radar output type to 0 to get tracks data
        self.track_pub = self.create_publisher(CustomRadarTrack,'radar_track',10)# Create a publisher for radar message

        # Create a subscriber for radar data
        self.create_timer(0.0005, self.receive_data_callback)  


        self.track_buffer = {} # Temporary storage for track parts

        self.cluster_start_time = None
        self.track_start_time = None
        self.remaining_clusters = 0
    # Send data over CAN
    def send_data(self, send_id, data):
        msg = can.Message(arbitration_id=send_id, data=data, is_extended_id=False)
        try:
            self.bus.send(msg)
            #self.get_logger().info(f"Message sent on {self.bus.channel_info}")
        except can.CanError:
            self.get_logger().error("Message NOT sent")
    # Change radar mode (clusters or tracks)
    def change_mode(self, mode=None):
        if mode is None:
            self.mode = 1 - self.mode
        else:
            self.mode = mode
        self.conf[0] = self.mode * 16
        self.send_data(self.conf_id, self.conf)
    # Receive CAN message
    def receive_data(self):
        try:
            return self.bus.recv(timeout=0.05)  
        except can.CanError as e:
            self.get_logger().error(f"CAN error: {e}")
        return None
    # Process incoming CAN messages and extract radar clusters or tracks
    def data_collection(self, msg):
        #print(msg)
        if self.t0 is None:
            self.t0 = msg.timestamp
        if self.mode == 1:  # Cluster data
            if msg.arbitration_id == self.cluster_id:
                self.cluster_count = msg.data[0]

                self.remaining_clusters = self.cluster_count
                self.cluster_start_time = time() 

                self.get_logger().info(f"Cluster count: {self.cluster_count}")
            elif msg.arbitration_id == self.radar_cluster_id:
                cluster_index = msg.data[0]
                cluster_rcs = (msg.data[1] * 0.5) - 50  # dBm^2
                distance = (msg.data[2]) * 0.2  # m
                azimuth_angle = ((msg.data[3]) * 2) - 90  # degrees
                timestamp = msg.timestamp - self.t0
                vrel = ((((msg.data[4] & 0b00000111) << 8) | msg.data[5]) * 0.05 )- 35  # m/s                
                self.remaining_clusters -= 1
                yield [self.cluster_count, cluster_index, cluster_rcs, timestamp, distance, azimuth_angle, vrel]
        else:  # Track data
            if msg.arbitration_id == self.track_id:
                self.track_count = msg.data[0]
                self.get_logger().info(f"Track count: {self.track_count}")
            elif msg.arbitration_id == self.radar_track_1_id:
                # Extract track data part 1
                self.track_ID = (((msg.data[0] & 0b11111111) << 8) | msg.data[1]) * 1 # Track ID
                self.track_LongDispl = (((msg.data[2] << 3) | ((msg.data[3]>>5)& 0b00000111)))* 0.1 # Track Longitudinal Display in m
                self.track_LatDispl = (((msg.data[4] << 2) | (msg.data[5] >> 6 )& 0b00000011) * 0.1 )- 51.1 # Track Latitudinal Display in m
                self.track_index = (msg.data[3] & 0b00011111)  * 1 # Track Index
                self.track_VrelLong = (((((msg.data[5] & 0b00111111) << 6) | (msg.data[6] >> 2) & 0b00111111)) * 0.02) - 35 # Track Longitudinal Vrel in m/s
                self.track_VrelLat = ((msg.data[7]) * 0.25) - 32 # Track Latitudinal Vrel in m/s

                # Store part 1 in buffer
                self.track_buffer.setdefault(self.track_index,{})['part1'] = {
                    'track_ID': self.track_ID,
                    'x': self.track_LongDispl, 
                    'y': self.track_LatDispl, 
                    'vrel_x': self.track_VrelLong, 
                    'vrel_y': self.track_VrelLat 
                }               

            elif msg.arbitration_id == self.radar_track_2_id:
                # Extract track data part 2
                self.track_index2 = (msg.data[3] & 0b00011111)  * 1 # Track Index2
                self.track_RCSvalue = (msg.data[0] * 0.5) - 50 # Track RCS Value in dBm^2
                self.track_Lifetime = (((msg.data[1] & 0b11111111) << 8) | msg.data[2]) * 0.1 # Track Lifetime in s
                
                # Store part 2 in buffer
                self.track_buffer.setdefault(self.track_index2,{})['part2'] = {
                    'rcs': self.track_RCSvalue,
                    'lifetime': self.track_Lifetime
                }

                # Combine parts if both received
                if 'part1' in self.track_buffer[self.track_index2]:
                    track = self.track_buffer.pop(self.track_index2)
                    p1 = track['part1']
                    p2 = track['part2']

                    range_m = (math.sqrt(p1['x']**2 + p1['y']**2))
                    azimuth_deg = math.degrees(math.atan2(p1['y'],p1['x']))

                else : 
                    return
                # Yield combined track message
                yield {
                    'track_ID': p1['track_ID'],
                    'track_count': self.track_count,
                    'track_index': self.track_index2,
                    'x': p1['x'], #longitudinal displacement
                    'y': p1['y'], #lateral displacement
                    'range': range_m,
                    'azimuth': azimuth_deg,
                    'vrel_x': p1['vrel_x'], #longitudinal vrel
                    'vrel_y': p1['vrel_y'], #lateral vrel
                    'rcs': p2['rcs'],
                    'lifetime': p2['lifetime']
                }                        
    # Callback to process CAN messages
    def receive_data_callback(self):
        # Receive data from CAN bus
        radar_data = self.receive_data()
        if radar_data is None:
            self.get_logger().info("No CAN message received this cycle")
            return
        for result in self.data_collection(radar_data):
            # Log track info
            self.get_logger().info(
                f"[Track ID {result['track_ID']}] Pos=({result['x']:.2f}m, {result['y']:.2f}m), "
                f"Track Index: {result['track_index']}, "
                f"Range={result['range']:.2f}m, Azimuth={result['azimuth']:.2f}°, "
                f"Vrel=({result['vrel_x']:.2f}, {result['vrel_y']:.2f}) m/s, "
                f"RCS={result['rcs']} dBm², Lifetime={result['lifetime']:.2f} s"                    
                )  
            # Publish ROS2 message
            radar_track_msg = CustomRadarTrack()
            radar_track_msg.header.stamp = self.get_clock().now().to_msg()
            radar_track_msg.header.frame_id = "radar_tracks_frame" 
            radar_track_msg.track_id = result['track_ID']
            radar_track_msg.track_index = result['track_index']
            radar_track_msg.track_count = result['track_count']
            radar_track_msg.x = result['x']
            radar_track_msg.y =result['y']
            radar_track_msg.range = result['range']
            radar_track_msg.azimuth = result['azimuth']
            radar_track_msg.vrel_x =result['vrel_x']
            radar_track_msg.vrel_y =result['vrel_y']
            radar_track_msg.rcs = result['rcs']
            radar_track_msg.lifetime = result['lifetime']

            self.track_pub.publish(radar_track_msg)

# ROS2 node entry point               
def main(args=None):
    rclpy.init(args=args)

    radar_driver_node = RadarDriverNode()

    try:
        rclpy.spin(radar_driver_node)
    except KeyboardInterrupt:
        pass
    finally:   
        # Safe shutdown of CAN bus    
        if hasattr(radar_driver_node, 'bus') and radar_driver_node.bus is not None:
            try:
                radar_driver_node.bus.shutdown()                
            except Exception as e:
                radar_driver_node.get_logger().error(f"Error shutting down CAN bus: {e}")
        
        radar_driver_node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

