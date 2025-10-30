import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/vom/ros2_ws/PikeRobotics_automated_detection/install/automated_counter'
