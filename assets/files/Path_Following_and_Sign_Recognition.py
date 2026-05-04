#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge


class LineFollowerPID:
    def __init__(self):
        rospy.init_node("line_follower_pid")

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.callback, queue_size=1)
        self.scan_sub = rospy.Subscriber("/scan", LaserScan, self.scan_callback)
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

        self.kp = 0.006
        self.ki = 0.0000
        self.kd = 0.002

        self.prev_error = 0.0
        self.integral = 0.0

        self.base_speed = 0.18

        self.red_threshold = 4000
        self.green_threshold = 1200

        self.last_state = None

        self.obstacle_detected = False
        self.obstacle_distance = 0.20
        self.obstacle_last_seen = rospy.Time(0)
        self.obstacle_clear_delay = 0.25

        self.last_error = 0.0
        self.last_seen_time = rospy.Time.now()
        self.line_seen_once = False
        self.lost_timeout = 1.0

        rospy.on_shutdown(self.cleanup)
        rospy.loginfo("Line follower started")

    def cleanup(self):
        cv2.destroyAllWindows()

    def publish_lost_line(self, twist):
        if not self.line_seen_once:
            twist.linear.x = 0.0
            twist.angular.z = 0.0
        else:
            time_since_seen = (rospy.Time.now() - self.last_seen_time).to_sec()

            if time_since_seen < self.lost_timeout:
                twist.linear.x = 0.10
                twist.angular.z = -self.kp * self.last_error
            else:
                twist.linear.x = 0.0
                twist.angular.z = 0.0

        self.cmd_pub.publish(twist)

    def callback(self, data):
        twist = Twist()

        if self.obstacle_detected:
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.cmd_pub.publish(twist)

            if self.last_state != "OBSTACLE":
                rospy.loginfo("OBSTACLE STOP")
                self.last_state = "OBSTACLE"

            cv2.imshow("frame", np.zeros((240, 320, 3), dtype=np.uint8))
            cv2.waitKey(1)
            return

        try:
            frame_rgb = self.bridge.imgmsg_to_cv2(data, desired_encoding="rgb8")
        except Exception as e:
            rospy.logwarn("cv_bridge error: %s", e)
            return

        frame_rgb = cv2.resize(frame_rgb, (320, 240))
        h, w = frame_rgb.shape[:2]

        hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)

        # Red detection
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])

        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        red_pixels = cv2.countNonZero(red_mask)

        if red_pixels > self.red_threshold:
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.cmd_pub.publish(twist)

            cv2.putText(frame_rgb, "STOP", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            if self.last_state != "STOP":
                rospy.loginfo("STOP")
                self.last_state = "STOP"

            cv2.imshow("frame", frame_rgb)
            cv2.waitKey(1)
            return

        # Green detection in lower ROI
        roi_y = int(h * 0.40)
        roi_hsv = hsv[roi_y:h, :]

        lower_green = np.array([35, 80, 80])
        upper_green = np.array([85, 255, 255])

        green_mask = cv2.inRange(roi_hsv, lower_green, upper_green)

        kernel = np.ones((5, 5), np.uint8)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)

        green_pixels = cv2.countNonZero(green_mask)
        M = cv2.moments(green_mask)

        line_detected = (green_pixels > self.green_threshold) and (M["m00"] > 0)

        if line_detected:
            cx = int(M["m10"] / M["m00"])
            error = cx - (w / 2.0)

            self.last_error = error
            self.last_seen_time = rospy.Time.now()
            self.line_seen_once = True

            self.integral += error
            derivative = error - self.prev_error
            angular_z = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
            self.prev_error = error

            twist.linear.x = self.base_speed
            twist.angular.z = -angular_z
            self.cmd_pub.publish(twist)

            cy = roi_y + int(M["m01"] / M["m00"])

            cv2.circle(frame_rgb, (cx, cy), 5, (0, 255, 0), -1)
            cv2.line(frame_rgb, (w // 2, roi_y), (w // 2, h), (255, 0, 0), 2)
            cv2.line(frame_rgb, (cx, roi_y), (cx, h), (0, 255, 0), 2)

            cv2.putText(frame_rgb, "MOVE", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            if self.last_state != "MOVE":
                rospy.loginfo("MOVE")
                self.last_state = "MOVE"
        else:
            self.publish_lost_line(twist)

            if self.last_state != "NORMAL":
                rospy.loginfo("NORMAL")
                self.last_state = "NORMAL"

        cv2.imshow("frame", frame_rgb)
        cv2.waitKey(1)

    def scan_callback(self, msg):
        center = len(msg.ranges) // 2
        front_ranges = msg.ranges[center - 20:center + 20]

        valid_ranges = [r for r in front_ranges if np.isfinite(r) and r > 0.0]

        if len(valid_ranges) == 0:
            if self.obstacle_detected and (rospy.Time.now() - self.obstacle_last_seen).to_sec() > self.obstacle_clear_delay:
                self.obstacle_detected = False
            return

        min_dist = min(valid_ranges)

        if min_dist < self.obstacle_distance:
            self.obstacle_detected = True
            self.obstacle_last_seen = rospy.Time.now()
        else:
            if self.obstacle_detected and (rospy.Time.now() - self.obstacle_last_seen).to_sec() > self.obstacle_clear_delay:
                self.obstacle_detected = False


if __name__ == "__main__":
    node = LineFollowerPID()
    rospy.spin()