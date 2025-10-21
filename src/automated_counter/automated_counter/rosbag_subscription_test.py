import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import sys

class GenericCameraNode(Node):
    def __init__(self, topic_name: str, compressed: bool = False, window_name: str = "Camera"):
        super().__init__('generic_camera_node')
        self.bridge = CvBridge()
        self.window_name = window_name
        self.compressed = compressed

        # Subscribe to the topic
        if compressed:
            self.sub = self.create_subscription(
                CompressedImage,
                topic_name,
                self.callback_compressed,
                10
            )
            self.get_logger().info(f"Subscribed to compressed topic: {topic_name}")
        else:
            self.sub = self.create_subscription(
                Image,
                topic_name,
                self.callback,
                10
            )
            self.get_logger().info(f"Subscribed to uncompressed topic: {topic_name}")

        # FPS calculation
        self.prev_time = time.time()
        self.fps = 0.0

    def callback(self, msg: Image):
        """Callback for uncompressed ROS Image."""
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.process_and_show(img)
        except Exception as e:
            self.get_logger().error(f"Error converting image: {e}")

    def callback_compressed(self, msg: CompressedImage):
        """Callback for CompressedImage."""
        try:
            # Decode compressed image to OpenCV format
            np_arr = np.frombuffer(msg.data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
            if img is None:
                self.get_logger().warning("Decoded compressed image is None")
                return
            self.process_and_show(img)
        except Exception as e:
            self.get_logger().error(f"Error decoding compressed image: {e}")

    def process_and_show(self, img: np.ndarray):
        """Normalize, compute FPS, and display."""
        if img is None:
            return

        # Compute FPS
        curr_time = time.time()
        dt = curr_time - self.prev_time
        if dt > 0:
            self.fps = 1.0 / dt
        self.prev_time = curr_time

        # Normalize to 0-255 if needed
        if len(img.shape) == 2 or img.shape[2] == 1:
            img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            img_bgr = cv2.cvtColor(img_norm.astype('uint8'), cv2.COLOR_GRAY2BGR)
        else:
            img_bgr = img.copy()

        # Put FPS on image
        cv2.putText(img_bgr, f"FPS: {self.fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show image
        cv2.imshow(self.window_name, img_bgr)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.get_logger().info("Q pressed. Exiting...")
            sys.exit()


def main(args=None):
    rclpy.init(args=args)

    # Example usage: change topic_name and compressed flag as needed
    topic_name='/wombot_gen3proto/seal_cameras_realsense/infra1/image_rect_raw/compressed'
    # topic_name = '/wombot_gen3proto/seal_cameras_realsense/color/image_raw/compressed'
    # topic_name = '/wombot_gen3proto/seal_cameras_realsense/depth/image_rect_raw'
    node = GenericCameraNode(topic_name=topic_name, compressed=True, window_name="Camera")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("KeyboardInterrupt, shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
