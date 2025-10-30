import os
import sys
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import cv2 as cv
import numpy as np
from scipy.optimize import differential_evolution

import csv 

#  User defined script inside the workspace
from depth_stream import p_tile_algo 
from depth_stream import image_morph
from depth_stream import iqr_dist
from std_msgs.msg import Float32 as distance_msg

global fps_depth
global depth_prev_frame_time, depth_new_frame_time

# Setting the ROI 200:400, 175:425
global width_final , width_start
global height_final, height_start
width_final=575
width_start=75
height_final=400
height_start=150
global width_depth, height_depth
width_depth=width_final-width_start
height_depth=height_final-height_start

global fourcc # video codec (4-character code)
fourcc = cv.VideoWriter_fourcc(*'MJPG')

global AOI
AOI = 0.95 # Area of interest of the obstacle to be detected in the cropped stream

CSV_FILE_PATH = "Obstacle Saliency Experiments/de_latency_profilet1.csv"

total_latency = 0
total_frames = 0
latency_avg=0
latency_ms_realtime_cumulative = 0
def csv_logger(popsize, maxiter, latency_ms, trial_type):
    file_exists = os.path.exists(CSV_FILE_PATH)
    
    
    with open(CSV_FILE_PATH, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['popsize', 'maxiter', 'latency_ms', 'trial_type'])    
        # Write header only if file didn't exist before
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            'popsize': popsize,
            'maxiter': maxiter,
            'latency_ms': latency_ms,
            'trial_type': trial_type
        })
        print(f"Latency: {latency_ms:.2f} ms for pop={popsize}, iter={maxiter}, trial_type={trial_type}")
        
class DESaliencyNode(Node):
    def __init__(self):
        super().__init__('de_saliency_node')
        header_txt = "\n Obstacle Saliency\n"
        fmt_header_txt = header_txt.center(100, '*')
        print("\033[36:4m" + fmt_header_txt)
        self.bridge = CvBridge()
        # self.depth_sub = self.create_subscription(Image, '/camera/depth/image_raw', self.depth_callback, 20)
        self.depth_sub =  self.create_subscription(CompressedImage,
                    '/wombot_gen3proto/seal_cameras_realsense/color/image_raw/compressed', self.depth_callback, 10)
        # self.rgb_sub = self.create_subscription(Image, '/camera/color/image_raw', self.rgb_callback, 10)
        # self.pub = self.create_publisher(Image, '/saliency_image', 10) 
        self.get_logger().info('DE Saliency Node started')   
        self.paused = False        

    def preprocess(self, depth_frame):
        return cv.resize(depth_frame[height_start:height_final, width_start:width_final],
                            (250, 
                            150),
                            interpolation=cv.INTER_NEAREST)
    

    def fitness(self, individual, depth_map):
        """
        Fitness function for Differential Evolution.
        'individual' contains [x, y, w, h] parameters.
        """
        # Ensure individual values are integers and within bounds
        x, y, w, h = map(int, individual)

        # Clip values to ensure they are within the bounds of the image
        x = np.clip(x, 0, depth_map.shape[1] - 1)
        y = np.clip(y, 0, depth_map.shape[0] - 1)
        w = np.clip(w, 5, depth_map.shape[1] - x)
        h = np.clip(h, 5, depth_map.shape[0] - y)

        # Get the region defined by [x, y, w, h]
        region = depth_map[y:y + h, x:x + w]
        if region.size == 0 or np.isnan(region).any():
            return 1e6  # Invalid region, high penalty

        # Compute the fitness score
        mean_d = np.mean(region)
        if mean_d > 100:
            # distance_penalty = (mean_d - 0.75) * 5  # or some other steep penalty
            return 1e8
        else:
            distance_penalty = 0.0
        std_d = np.std(region)
        proximity = 150 / (mean_d + 1e-6)  # Closer means more salient
        contrast = std_d
        _lambda_ = 0.1
        penalty = abs(w * h - 100) * _lambda_  # Discourage too small/big regions

        # Total score is the sum of proximity, contrast, and penalty
        total_score = proximity + contrast - penalty + distance_penalty

        return total_score  # DE minimizes the function, so we return it as it is.
    
    def rgb_callback(self, color):
        rgb_frame = self.bridge.imgmsg_to_cv2(color, desired_encoding='bgr8')
        cv.imshow("RawRGBStream", rgb_frame)
        if cv.waitKey(1) == ord("q"):
            sys.exit(0)
    
    def depth_augmented(self, depth):
		 #---------Segmentation
        thresh_value = p_tile_algo.p_tile_thresh(depth, AOI) # Area of interest = 90-100% in cropped depth
        ret, cropped_depth_seg = cv.threshold(depth, thresh_value, np.max(depth), cv.THRESH_TOZERO_INV) # Actual segemnted image
        seg_morph = image_morph.image_morph_algo(cropped_depth_seg) # Used to obtaining the obstacle distance
        # print("testttttttt")
		#-- Displaying necessary frame(s)
        seg_morph_display = cv.normalize(seg_morph, None, 0, 65535, cv.NORM_MINMAX) # Just for display 
        cv.namedWindow('DepthAugmented', cv.WINDOW_NORMAL)
        cv.resizeWindow('DepthAugmented', 640, 480)
        cv.imshow('DepthAugmented', seg_morph)
        if cv.waitKey(1)==ord('q'):
            sys.exit(0)        

        return seg_morph_display
    
    def DE_LatencyProfiling(self, bounds, depth):
        #-- Latency Profiling
        trials=100
        # --- Trial 1: Fix maxiter, vary popsize ---
        fixed_maxiter = 3
        popsize_values = [1, 3, 6, 8, 10]

        for popsize in popsize_values:
            latencies = []
            for _ in range(trials):
                start_time = time.time()
                result = differential_evolution(self.fitness, bounds, args=(depth,), maxiter=fixed_maxiter, popsize=popsize)
                latency_ms = (time.time() - start_time) * 1000
                latencies.append(latency_ms)
            # Removing Outliers
            latencies_np = np.array(latencies)
            Q1 = np.percentile(latencies_np, 5)
            Q3 = np.percentile(latencies_np, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 2.0 * IQR
            upper_bound = Q3 + 0.1 * IQR
            filtered_latencies = latencies_np[(latencies_np >= lower_bound) & (latencies_np <= upper_bound)]
            # Compute average latency after outlier removal
            avg_latency = np.mean(filtered_latencies)
            csv_logger( popsize,
                        fixed_maxiter,
                        avg_latency,
                        'popsize_varied'
                        )

        # --- Trial 2: Fix popsize, vary maxiter ---
        fixed_popsize = 3
        maxiter_values = [1, 3, 6, 8, 10]

        for maxiter in maxiter_values:
            latencies = []
            for _ in range(trials):
                start_time = time.time()
                result = differential_evolution(self.fitness, bounds, args=(depth,), maxiter=maxiter, popsize=fixed_popsize)
                latency_ms = (time.time() - start_time) * 1000
                latencies.append(latency_ms)
            # Removing Outliers
            latencies_np = np.array(latencies)
            Q1 = np.percentile(latencies_np, 5)
            Q3 = np.percentile(latencies_np, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 2.0 * IQR
            upper_bound = Q3 + 0.1 * IQR
            filtered_latencies = latencies_np[(latencies_np >= lower_bound) & (latencies_np <= upper_bound)]
            # Compute average latency after outlier removal
            avg_latency = np.mean(filtered_latencies)
            csv_logger(fixed_popsize, maxiter, avg_latency, 'maxiter_varied')
        
        sys.exit()
            
    def DE_ObstacleSaliency(self, depth):
        global latency_avg, latency_ms_realtime_cumulative
        # Define bounds for Differential Evolution [x, y, w, h]
        bounds = [(10, 85), (10, 55), (5, 25), (5, 25)]

        # ---- DE Obstacle Saliency
        start_time_realtime = time.time()                
        result = differential_evolution(self.fitness, bounds, args=(depth,), maxiter=10, popsize=30)
        latency_realtime_ms = (time.time() - start_time_realtime) * 1000
        latency_ms_realtime_cumulative += latency_realtime_ms
        # Latency Profiling      
        # self.DE_LatencyProfiling(bounds, depth)
        # result = differential_evolution(self.fitness, bounds, args=(depth,), maxiter=3, popsize=3)
        # Get the best bounding box coordinates
        x, y, w, h = map(int, result.x)

        # Normalize and convert to BGR for visualization
        vis = cv.normalize(depth, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
        vis = cv.cvtColor(vis, cv.COLOR_GRAY2BGR)

        # Draw the bounding box on the image
        cv.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 1)

        # Convert the processed image back to a ROS image message
        saliency_msg = self.bridge.cv2_to_imgmsg(vis, encoding='bgr8')

        # Publish the saliency-detected image
        # self.pub.publish(saliency_msg)
        cv.namedWindow('Saliency', cv.WINDOW_NORMAL)
        cv.resizeWindow('Saliency', 640, 480)
        if not self.paused:
            cv.imshow('Saliency', vis)
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            cv.destroyAllWindows()
            sys.exit(0)
        elif key == ord('p'):
            if self.paused:
                self.paused = False
            else:
                self.paused = True

    def depth_callback(self, msg):
        global total_frames, latency_avg, latency_ms_realtime_cumulative
        try:
            # Convert the ROS image message to an OpenCV image
            # depth_frame =(self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough'))
            np_arr = np.frombuffer(msg.data, np.uint8)
            depth_frame = cv.imdecode(np_arr, cv.IMREAD_UNCHANGED)
            #----Loading from the directory
            # depth_frame = cv.imread("/home/husarion/ros2_ws/Obstacle Saliency Experiments/EXP-4/RawDepthMap.png", cv.IMREAD_UNCHANGED)
            if len(depth_frame.shape) == 3:
            # Some images load as (H, W, 3) even if grayscale, convert to 1-channel
                depth_frame = cv.cvtColor(depth_frame, cv.COLOR_BGR2GRAY)
            else:
                depth_frame = depth_frame.copy()

            # # Preprocess the depth frame (resize and crop)
            depth_map = self.preprocess(depth_frame)
            
            #---Display Results
            # self.depth_augmented(depth_map)
            self.DE_ObstacleSaliency(depth_map)  
            total_frames+=1
            # print(f"Frame_count {total_frames:d}")
            # latency_avg = latency_ms_realtime_cumulative/total_frames
            # print(f"Average latency per frame: {latency_avg:.2f} ms")
            cv.imshow("RawDepthMap", cv.normalize(depth_frame, None, 0, 65535, cv.NORM_MINMAX))
            if cv.waitKey(1) == ord("q"):
                sys.exit(0)
        except Exception as e:
            self.get_logger().error(f'Error in image_callback: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    node = DESaliencyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv.destroyAllWindows()

if __name__ == '__main__':
    main()