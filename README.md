# UGProject
Endoscopy Image Analysis Web-platform

EzEndo is a web-based platform designed to empower healthcare professionals and researchers by providing a seamless way to analyze endoscopy images. Users can upload endoscopy images onto our platform to be processed and analyzed by our advanced AI algorithms to identify if their image is normal or abnormal, and provide insights into various similar endoscopy cases. With EzEndo, we aim to enhance diagnostic accuracy, speed up the analysis process, and contribute to the advancement of gastrointestinal health care.


import threading
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from rclpy.exceptions import ROSInterruptException
import signal
import math

class TraceSquare(Node):
    def __init__(self):
        super().__init__('drive_circular')
        self.publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.rate = self.create_rate(10)
        
    def walk_forward(self, distance, speed):
        desired_velocity = Twist()
        desired_velocity.linear.x = float(speed)
        duration = distance / speed
        
        for _ in range(int(duration)):
            self.publisher.publish(desired_velocity)
            self.rate.sleep()
        
        # stop walking
        desired_velocity.linear.x = 0
        self.publisher.publish(desired_velocity)
        self.rate.sleep()
        
    def turn(self, angle, turning_speed):
        desired_velocity = Twist()
        desired_velocity.angular.z = float(turning_speed)
        duration = angle / turning_speed
        for _ in range(int(duration)):
            self.publisher.publish(desired_velocity)
            self.rate.sleep()
        
        # stop turning
        desired_velocity.angular.z = 0
        self.publisher.publish(desired_velocity)
        self.rate.sleep()
        
    def drive_square(self):
        distance = 1.0
        speed = 0.2
        angle = math.pi / 2
        turning_speed = math.pi / 8
        
        # loop 4 times to make a square 
        for _ in range(4):
            self.walk_forward(distance, speed)
            self.turn(angle, turning_speed)
            
def main():
    def signal_handler(sig, frame):
        trace_square.stop()
        rclpy.shutdown()
        
    rclpy.init(args=None)
    trace_square = TraceSquare()
    
    signal.signal(signal.SIGINT, signal_handler)
    thread = threading.Thread(target=rclpy.spin, args=(trace_square,), daemon=True)
    thread.start()
    
    try:
        trace_square.drive_square()
    except ROSInterruptException:
        pass
    
    if __name__ == '__main__':
        main()
