import os
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import time
import sys

from automated_counter import detector


        
# ==========================
# CONFIGURATION
# ==========================
use_camera = False          # True = live ROS2, False = load from folder

# H, W = rgb_frame.shape[:2]
H, W = (480,640)
roi_w, roi_h = 80, 75
roi_x = (W - roi_w) // 2
roi_y = (H - roi_h) // 2
roi = (roi_x, roi_y, roi_w, roi_h)



# ==========================
# DATASET CONFIGURATION
# ==========================
use_path_1 = True  # Choose dataset CPCHem or SUNCOR

# Dataset paths
working_dir= "/home/vom/ros2_ws/PikeRobotics_automated_counting"

# Dynamically set dataset path and name
if use_path_1:
    __dataname__ = "CPC"
    path = f"{working_dir}/{__dataname__}"
    # <-- set this to whichever CPC subfolder you’re using
    current_dataset = "CPC_9"
    # current_dataset = "CPC_10"
    # current_dataset = "CPC_19"
    # current_dataset = "CPC_43"
    # current_dataset = "CPC_61"   
else:
    __dataname__ = "SUNCOR"
    path = f"{working_dir}/{__dataname__}"
    current_dataset = "SUNCOR_24"  # <-- set this to whichever SUNCOR subfolder you’re using

#------ Current dataset 
dataset_path = current_dataset

# Define video paths dynamically 
#-- CPC Files 
rgb_video_path = f"{path}/{dataset_path}/{dataset_path}__wombot_gen3proto_seal_cameras_realsense_color_image_raw_compressed.mp4" 
# ir_video_path = f"{path}/{dataset_path}/CPC_9__wombot_gen3proto_seal_cameras_realsense_infra1_image_rect_raw_compressed.mp4" 
# gray_video_path = f"{path}/{dataset_path}/CPC_9__wombot_gen3proto_seal_cameras_flexx_gray_image_raw_compressed.mp4" 
# depth_video_path = f"{path}/{dataset_path}/depth.mp4" 
# #--SUNCOR Files #
# rgb_video_path = f"{path}/{dataset_path}/4_RealsenseColor.mp4" # 
# ir_video_path = f"{path}/{dataset_path}/5_realsense_infra1_image_rect_raw_compressed.mp4" # 
# gray_video_path = f"{path}/{dataset_path}/5_RealsenseGray.mp4" # 
# depth_video_path = f"{path}/{dataset_path}/6_RealsenseDepth.mp4" # 
# print("Using dataset:", path) # 
# print("gray video path:", gray_video_path)


# ==========================
# MODE 1: VIDEO FILES
# ==========================
if not use_camera:    
    #----Selecting the stream...
    use_rgb = True
    use_ir = False
    use_gray = False
    use_depth = False  # Enable if depth video exists

    # ------------Video Recorder Configuration
    record_output = False
    output_dir = "Feature count recordings"
    os.makedirs(output_dir, exist_ok=True)
    # Define codec and VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_rgb = None
    out_gray = None

    # Open videos
    rgb_cap   = cv2.VideoCapture(rgb_video_path)   if use_rgb else None
    ir_cap    = cv2.VideoCapture(ir_video_path)    if use_ir else None
    gray_cap  = cv2.VideoCapture(gray_video_path)  if use_gray else None
    depth_cap = cv2.VideoCapture(depth_video_path) if use_depth else None

    # Get FPS
    # rgb_fps   = rgb_cap.get(cv2.CAP_PROP_FPS)   if rgb_cap else 30
    # ir_fps    = ir_cap.get(cv2.CAP_PROP_FPS)    if ir_cap else 30
    # gray_fps  = gray_cap.get(cv2.CAP_PROP_FPS)  if gray_cap else 30
    # depth_fps = depth_cap.get(cv2.CAP_PROP_FPS) if depth_cap else 30

    # Per-frame delays
    # rgb_delay   = int(1000 / rgb_fps) if rgb_cap else None
    # ir_delay    = int(1000 / ir_fps) if ir_cap else None
    # gray_delay  = int(1000 / gray_fps) if gray_cap else None
    # depth_delay = int(1000 / depth_fps) if depth_cap else None

    # Check if videos opened
    if use_rgb and not rgb_cap.isOpened(): raise IOError(f"RGB video cannot be opened: {rgb_video_path}")
    if use_ir and not ir_cap.isOpened(): raise IOError(f"IR video cannot be opened: {ir_video_path}")
    if use_gray and not gray_cap.isOpened(): raise IOError(f"Gray video cannot be opened: {gray_video_path}")
    if use_depth and not depth_cap.isOpened(): raise IOError(f"Depth video cannot be opened: {depth_video_path}")

    print("Playing videos from folder...")

    while True:
        ret_rgb,   rgb_frame   = (rgb_cap.read()   if use_rgb else (False, None))
        ret_ir,    ir_frame    = (ir_cap.read()    if use_ir else (False, None))
        ret_gray,  gray_frame  = (gray_cap.read()  if use_gray else (False, None))
        ret_depth, depth_frame = (depth_cap.read() if use_depth else (False, None))

        if ((use_rgb and not ret_rgb) or
            (use_ir and not ret_ir) or
            (use_gray and not ret_gray) or
            (use_depth and not ret_depth)):
            print("End of one of the videos reached.")
            break

        # Callbacks 
        ############ RGB
        if use_rgb:   
            # ####################
            # DETECTOR
            #######################
            # count, bboxes = detector.detect(rgb_frame, method="template", method_type="cv", roi =roi)
            count, bboxes = detector.detect(rgb_frame, method="template", method_type="cv", roi =roi)
            cv2.putText(rgb_frame, f"Bolts: {count}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            for (x, y, w, h) in bboxes:
                cv2.rectangle(rgb_frame, (x, y), (x+w, y+h), (0,0,255), 2)
            
            # Initialize video writer for RGB if not yet created
            if record_output and out_rgb is None:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                video_path_rgb = os.path.join(output_dir, f"timestamp_rgb_{timestamp}.avi")
                h, w, _ = rgb_frame.shape
                out_rgb = cv2.VideoWriter(video_path_rgb, fourcc, 10.0, (w, h))

            # Write frame
            if record_output and out_rgb is not None:
                out_rgb.write(rgb_frame)

            # Display
            cv2.imshow("RGB Video", rgb_frame)
        
        ############ IR
        if use_ir:    
            # ####################
            # DETECTOR
            #######################
            count_ir, bboxes_ir = detector.detect(ir_frame, method="template", method_type="cv",  roi =roi)
            print(roi)
            cv2.putText(ir_frame, f"Bolts: {count_ir}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            for (x, y, w, h) in bboxes_ir:
                cv2.rectangle(ir_frame, (x, y), (x+w, y+h), (0,0,255), 2)
            cv2.imshow("IR Video", ir_frame)
            
        ############ Grayscale
        if use_gray:  
            # ####################
            # DETECTOR
            #######################
            count_gray, bboxes_gray = detector.detect(gray_frame, method="template", method_type="cv",  roi =roi)
            cv2.putText(gray_frame, f"Bolts: {count_gray}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            for (x, y, w, h) in bboxes_gray:
                cv2.rectangle(gray_frame, (x, y), (x+w, y+h), (0,0,255), 2)
                
            # Initialize video writer for IR if not yet created
            if record_output and out_gray is None:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                video_path_gray = os.path.join(output_dir, f"timestamp_gray_{timestamp}.avi")
                h, w = gray_frame.shape[:2]  # ignore channels
                out_gray = cv2.VideoWriter(video_path_gray, fourcc, 10.0, (w, h), isColor=False)

            # Write frame
            if record_output and out_gray is not None:
                # Ensure gray_frame is actually 2D before writing
                if len(gray_frame.shape) == 3:
                    gray_frame_to_write = cv2.cvtColor(gray_frame, cv2.COLOR_BGR2GRAY)
                else:
                    gray_frame_to_write = gray_frame
                out_gray.write(gray_frame_to_write)
            #display
            cv2.imshow("Gray Video", gray_frame)
            
        ##########Depth
        if use_depth: 
            # ####################
            # DETECTOR
            #######################
            count_depth, bboxes_depth = detector.detect(depth_frame, method="template", method_type="cv",  roi =roi)
            cv2.putText(depth_frame, f"Bolts: {count_depth}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            
            for (x, y, w, h) in bboxes_depth:
                cv2.rectangle(depth_frame, (x, y), (x+w, y+h), (0,0,255), 2)
            
            cv2.imshow("Depth Video", depth_frame)

        # max_delay = max(rgb_delay, ir_delay, gray_delay, depth_delay)
        if cv2.waitKey(150) & 0xFF == ord("q"):  # ESC to quit
            self.get_logger().info("Q pressed. Exiting...")
            sys.exit()

    # Cleanup
    if use_rgb:   rgb_cap.release()
    if use_ir:    ir_cap.release()
    if use_gray:  gray_cap.release()
    if use_depth: depth_cap.release()
    cv2.destroyAllWindows()

# ==========================
# MODE 2: LIVE ROS2 CAMERA
# ==========================
else:
    class IntelRealSenseSubscriber(Node):
        def __init__(self):
            super().__init__('intel_realsense_node')
            header_txt = "\n ROS 2 topic Subscription started\n"
            fmt_header_txt = header_txt.center(100, '*')
            print("\033[36:4m" + fmt_header_txt)
            self.bridge = CvBridge()
            
            # Flags: compressed or not
            self.rgb_compressed   = True  # Only compressed available
            self.ir_compressed    = True
            self.depth_compressed = False # No available topic for depth compressed

            # Image containers
            self.rgb_img = None
            self.ir_img = None
            self.gray_img = None
            self.depth_img = None

            # --------------------------
            # Subscribe to RGB
            # --------------------------
            if self.rgb_compressed:
                self.rgb_sub = self.create_subscription(
                    CompressedImage,
                    '/wombot_gen3proto/seal_cameras_realsense/color/image_raw/compressed',
                    self.rgb_callback,
                    10
                )
            else:
                self.rgb_sub = self.create_subscription(
                    Image,
                    '/wombot_gen3proto/seal_cameras_realsense/color/image_raw',
                    self.rgb_callback,
                    10
                )

            # --------------------------
            # Subscribe to IR
            # --------------------------
            # if self.ir_compressed:
            #     self.ir_sub = self.create_subscription(
            #         CompressedImage,
            #         '/wombot_gen3proto/seal_cameras_realsense/infra1/image_rect_raw/compressed',
            #         self.ir_callback,
            #         10
            #     )
            # else:
            #     self.ir_sub = self.create_subscription(
            #         Image,
            #         '/wombot_gen3proto/seal_cameras_realsense/infra1/image_rect_raw',
            #         self.ir_callback,
            #         10
            #     )

            # --------------------------
            # Subscribe to Depth
            # --------------------------
            # if self.depth_compressed:
            #     self.depth_sub = self.create_subscription(
            #         CompressedImage,
            #         '/wombot_gen3proto/seal_cameras_realsense/depth/image_rect_raw/compressed',
            #         self.depth_callback,
            #         10
            #     )
            # else:
            #     self.depth_sub = self.create_subscription(
            #         Image,
            #         '/wombot_gen3proto/seal_cameras_realsense/depth/image_rect_raw',
            #         self.depth_callback,
            #         10
            #     )
            
            self.save_videos = False  # toggle this flag to enable/disable saving

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

            if self.save_videos:
                # RGB
                rgb_topic_name = '/wombot_gen3proto/seal_cameras_realsense/color/image_raw'
                self.rgb_writer = cv2.VideoWriter(
                    f'{path}/{dataset_path}/CPC_9_{rgb_topic_name[1:].replace("/", "_")}.mp4',
                    fourcc, 15.0, (640,480)
                )

                # IR
                ir_topic_name = '/wombot_gen3proto/seal_cameras_realsense/infra1/image_rect_raw'
                self.ir_writer = cv2.VideoWriter(
                    f'{path}/{dataset_path}/CPC_9_{ir_topic_name[1:].replace("/", "_")}.mp4',
                    fourcc, 15.0, (640,480)
                )

                # Gray (from RGB)
                gray_topic_name = 'gray_from_rgb'
                self.gray_writer = cv2.VideoWriter(
                    f'{path}/{dataset_path}/CPC_9_{gray_topic_name}.mp4',
                    fourcc, 15.0, (640,480)
                )

                # Depth
                depth_topic_name = '/wombot_gen3proto/seal_cameras_realsense/depth/image_rect_raw'
                self.depth_writer = cv2.VideoWriter(
                    f'{path}/{dataset_path}/CPC_9_{depth_topic_name[1:].replace("/", "_")}.mp4',
                    fourcc, 15.0, (640,480)
                )
            else:
                self.rgb_writer = None
                self.ir_writer = None
                self.gray_writer = None
                self.depth_writer = None


            # Previous stamps for FPS calculation
            self.rgb_prev_stamp = 0.0
            self.ir_prev_stamp = 0.0
            self.depth_prev_stamp = 0.0

            # FPS buffers (optional smoothing)
            self.rgb_fps_buffer = []
            self.ir_fps_buffer = []
            self.depth_fps_buffer = []


        def compute_fps(self, msg, prev_stamp_attr, fps_buffer_attr=None, smoothing=3):
            """
            Compute FPS from ROS message timestamp and optionally smooth over a buffer.

            msg: ROS Image or CompressedImage with header.stamp
            prev_stamp_attr: string name of attribute storing previous timestamp
            fps_buffer_attr: string name of list attribute storing last N FPS values
            smoothing: number of frames to average over
            """
            curr_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            prev_stamp = getattr(self, prev_stamp_attr, curr_stamp)
            dt = curr_stamp - prev_stamp
            fps = 1.0 / dt if dt > 0 else 0.0
            setattr(self, prev_stamp_attr, curr_stamp)

            # Optional smoothing
            if fps_buffer_attr is not None:
                buffer_list = getattr(self, fps_buffer_attr)
                buffer_list.append(fps)
                if len(buffer_list) > smoothing:
                    buffer_list.pop(0)
                fps = sum(buffer_list) / len(buffer_list)

            return fps




        def _generic_callback(self, msg, compressed_flag, name, prev_stamp_attr, fps_buffer_attr=None, write_depth=False):
            try:
                # Decode/compressed logic
                if compressed_flag:
                    np_arr = np.frombuffer(msg.data, np.uint8)
                    img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
                    if img is None:
                        self.get_logger().warning(f"[{name}] Decoded image is None")
                        return
                else:
                    img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                    if img is None:
                        self.get_logger().warning(f"[{name}] Converted image is None")
                        return

                # Update FPS
                if fps_buffer_attr:
                    fps = self.compute_fps(msg, prev_stamp_attr, fps_buffer_attr=fps_buffer_attr)
                    # Store the latest FPS in a separate attribute
                    setattr(self, f"{name.lower()}_fps", fps)

                # Store image for later
                if name == "RGB":
                    self.rgb_img = img
                elif name == "IR":
                    self.ir_img = img
                elif name == "Depth":
                    self.depth_img = img

                # Process image for display (grayscale to BGR)
                if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
                    img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
                    img_bgr = cv2.cvtColor(img_norm.astype('uint8'), cv2.COLOR_GRAY2BGR)
                else:
                    img_bgr = img.copy()

                # Save BGR for later display in callback
                setattr(self, f"{name.lower()}_bgr", img_bgr)

            except Exception as e:
                self.get_logger().error(f"[{name}] Error in callback: {e}")


            
        # ----------------------
        # RGB Callback
        # ----------------------
        def rgb_callback(self, msg):
            self._generic_callback(msg, self.rgb_compressed, "RGB", "rgb_prev_time", fps_buffer_attr="rgb_fps_buffer")

            # Retrieve FPS stored in generic callback
            fps = getattr(self, "rgb_fps", 0.0)
            img_bgr = getattr(self, "rgb_bgr", None)

            if img_bgr is not None:
                # ####################
                # DETECTOR
                #######################
                count, bboxes = detector.detect(self.rgb_img, method="template", method_type="cv")
                cv2.putText(img_bgr, f"Bolts: {count}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                for (x, y, w, h) in bboxes:
                    cv2.rectangle(img_bgr, (x, y), (x+w, y+h), (0,0,255), 2)

                cv2.putText(img_bgr, f"FPS: {fps:.2f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.imshow("RGB Camera", img_bgr)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.get_logger().info("Q pressed. Exiting...")
                    sys.exit()


        # ----------------------
        # IR Callback
        # ----------------------
        def ir_callback(self, msg):
            self._generic_callback(msg, self.ir_compressed, "IR", "ir_prev_time", fps_buffer_attr="ir_fps_buffer")
            fps = getattr(self, "ir_fps", 0.0)
            img_bgr = getattr(self, "ir_bgr", None)
            if img_bgr is not None:
                # ####################
                # DETECTOR
                #######################
                count_ir, bboxes_ir = detector.detect(self.ir_img, img_type="ir", method="template", method_type="cv")
                cv2.putText(img_bgr, f"Bolts: {count_ir}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                for (x, y, w, h) in bboxes_ir:
                    cv2.rectangle(img_bgr, (x, y), (x+w, y+h), (0,0,255), 2)
                
                cv2.putText(img_bgr, f"FPS: {fps:.2f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.imshow("IR Camera", img_bgr)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.get_logger().info("Q pressed. Exiting...")
                    sys.exit()

        # ----------------------
        # Depth Callback
        # ----------------------
        def depth_callback(self, msg):
            self._generic_callback(msg, self.depth_compressed, "Depth", "depth_prev_time", fps_buffer_attr="depth_fps_buffer",
                                   write_depth=False  # write to video if needed
            )

            # Retrieve computed FPS and the converted BGR image
            fps = getattr(self, "depth_fps", 0.0)
            img_bgr = getattr(self, "depth_bgr", None)

            if img_bgr is not None:
                # Overlay FPS text
                # ####################
                # DETECTOR
                #######################
                count_depth, bboxes_depth = detector.detect(self.depth_img, method="template", method_type="cv")
                cv2.putText(img_bgr, f"FPS: {fps:.2f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.putText(img_bgr, f"Bolts: {count_depth}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

                for (x, y, w, h) in bboxes_depth:
                    cv2.rectangle(img_bgr, (x, y), (x+w, y+h), (0,0,255), 2)

                # Display the depth frame
                cv2.imshow("Depth Camera", img_bgr)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.get_logger().info("Q pressed. Exiting...")
                    sys.exit()

            # ----------------------
            
            
        def destroy_node(self):
            # self.rgb_writer.release()
            # self.ir_writer.release()
            # self.gray_writer.release()
            # self.depth_writer.release()
            cv2.destroyAllWindows()
            super().destroy_node()

    # ----------------------
    # MAIN ROS2
    # ----------------------
    rclpy.init()
    node = IntelRealSenseSubscriber()
    try:
        rclpy.spin(node)  # Continuously process callbacks
    except KeyboardInterrupt:
        print("\n\033[91m KeyboardInterrupt detected. Exiting...\033[0m")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()
