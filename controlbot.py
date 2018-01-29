#! /usr/bin/env python

# rospy for the subscriber
import rospy
# ROS Image message
from sensor_msgs.msg import Image
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError
# OpenCV2 for saving an image
import cv2
from network import network
from geometry_msgs.msg import Twist
import numpy as np

# Instantiate CvBridge
bridge = CvBridge()

model = network()
model.load('/home/z/Dropbox/bachelorarbeit/catkin_ws/src/rosnet/src/network.model')
inputWidth = 128
inputHeight = 72

pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)


def ImageCallback(msg):
    print("Received an image!")
    try:
        # Convert your ROS Image message to OpenCV2
        cv2Img = bridge.imgmsg_to_cv2(msg, "bgr8")
    except:
        pass

    prediction = model.predict(cv2Img.reshape(-1, inputWidth, inputHeight, 3))[0]
    prediction[0] = prediction[0] /10
    print(prediction)


    controlMsg = Twist()
    controlMsg.linear.x = prediction[0]
    controlMsg.angular.z = prediction[1]

    global pub
    pub.publish(controlMsg)

    while True:
        cv2.imshow('frame', cv2Img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        break

def main():
    rospy.init_node('controlbot')
    # Define your image topic
    image_topic = "/camera/image_raw"
    # Set up your subscriber and define its callback
    rospy.Subscriber(image_topic, Image, ImageCallback)


    # Spin until ctrl + c
    rospy.spin()

if __name__ == "__main__":
    main()
