#! /usr/bin/env python

# rospy for the subscriber
import rospy
# ROS Image message
from sensor_msgs.msg import Image

from geometry_msgs.msg import Twist
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError
# OpenCV2 for saving an image
import cv2
from random import shuffle
import time
import os
import sys
import numpy as np
bridge = CvBridge()
frameCounter = 0
trainSet = []
joystickInput = [0,0,0]


def ImageCallback(msg):
    print("Received an image!")
    try:
        # Convert your ROS Image message to OpenCV2
        cv2Img = bridge.imgmsg_to_cv2(msg, "bgr8")
    except:
        pass
    trainSet.append([cv2Img, joystickInput])
    print(joystickInput)

    while (True):
        cv2.imshow('test', cv2Img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        break
    #show frame
    global frameCounter
    frameCounter += 1

def InputCallback(msg):
    global joystickInput

    if (msg.angular.z > 0.1):
        joystickInput = [1, 0, 0]
    elif (msg.angular.z < -0.1):
        joystickInput = [0, 0, 1]
    else:
        joystickInput = [0,1,0]


def main():
    print("Starting recoring soon")
    time.sleep(5)

    rospy.init_node('image_listener')
    # Set up your subscriber and define its callback
    rospy.Subscriber("/cmd_vel", Twist, InputCallback)
    rospy.Subscriber("/camera/image_raw", Image, ImageCallback)
    # Spin until ctrl + c
    while not rospy.is_shutdown():
        if frameCounter > 1000:
            print("=================== Frames Saved ======================")
            shuffle(trainSet)
            try:
                os.remove("trainSet.npy")
            except OSError:
                pass

            np.save("/home/z/Dropbox/bachelorarbeit/catkin_ws/src/rosnet/src/trainSet.npy", trainSet)
            print("saved")
            rospy.signal_shutdown("asdf")

if __name__ == "__main__":
    main()