#!/usr/bin/env python3

import rosbag  # ROS1
import rclpy   # ROS2
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import importlib
import threading
import time

class Ros1ToRos2Replay(Node):
    def __init__(self, bag_path):
        super().__init__('ros1_to_ros2_replay')
        self.bag = rosbag.Bag(bag_path)
        self.publishers = {}

        # Relaxed QoS to avoid ROS1/ROS2 mismatch
        self.qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

    def get_publisher(self, topic, msg_type):
        if topic not in self.publishers:
            pkg_name = msg_type._type.split('/')[0]
            msg_name = msg_type._type.split('/')[1]
            ros2_module = importlib.import_module(f'{pkg_name}.msg')
            ros2_type = getattr(ros2_module, msg_name)
            self.publishers[topic] = self.create_publisher(ros2_type, topic, self.qos)
        return self.publishers[topic]

    def replay(self):
        for topic, msg, t in self.bag.read_messages():
            pub = self.get_publisher(topic, msg._type)
            ros2_msg = self.convert_msg(msg)
            if ros2_msg:
                pub.publish(ros2_msg)
            time.sleep(0.01)  # small delay to control speed

    def convert_msg(self, ros1_msg):
        # ROS1 <-> ROS2 messages do not auto-convert
        # You must write field-by-field mapping here if needed
        # Placeholder: try naive assignment
        return ros1_msg

def main(bag_path):
    rclpy.init()
    node = Ros1ToRos2Replay(bag_path)

    replay_thread = threading.Thread(target=node.replay)
    replay_thread.start()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python3 ros1_to_ros2_convert.py <ros1_bag.bag>")
        exit(1)
    main(sys.argv[1])
