#! /usr/bin/env python
import numpy as np
import rospy
# ROS Image message
from cv_bridge import CvBridge, CvBridgeError
import cv2
from sensor_msgs.msg import Joy, Image
from geometry_msgs.msg import Twist
import os
from subprocess import Popen
import time

bridge = CvBridge()
trainSet = []
joystickInput = [0,0,0]
controlBotRunning = 0
xSum = 0
recordFlag = 0
pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

def pubStop():
    controlMsg = Twist()
    controlMsg.linear.x = 0
    controlMsg.angular.z = 0
    global pub
    pub.publish(controlMsg)

def deleteTrainingData():
    print("DELETE")
def takeFrame():
    print("PHOTO")
def trainNet():
    print("Train!")
def saveTrainingData():
    np.save("/home/z/Dropbox/bachelorarbeit/catkin_ws/src/rosnet/src/trainSetJoy.npy", trainSet)
    print("saved")

def TwistCallback(msg):
    global joystickInput

    if (msg.angular.z > 0.1):
        joystickInput = [1, 0, 0]
    elif (msg.angular.z < -0.1):
        joystickInput = [0, 0, 1]
    else:
        joystickInput = [0,1,0]
def JoyCallback(msg):
    r1 = msg.buttons[11]
    delta = msg.buttons[12]
    o = msg.buttons[13]
    x = msg.buttons[14]
    select = msg.buttons[0]
    start  = msg.buttons[3]
    print("R1: \t {}".format(r1))
    print("delta:\t {}".format(delta))
    print("O: \t {}".format(o))
    print("X: \t {}".format(x))

    if x:
        global xSum
        xSum += 1
        print xSum
    if delta:
        trainNet()
    if r1:
        takeFrame()
    if select and start:
        deleteTrainingData()
    if o:
        saveTrainingData()

    global controlBotRunning
    if xSum > 3 and not controlBotRunning:
        controlBotRunning = 1
        global xterm
        xterm = Popen(["xterm", "-e", "rosrun rosnet controlbotGazebo.py"])
        xSum = 0
        time.sleep(2)

    if xSum > 3 and controlBotRunning:
        xterm.terminate()
        pubStop()
        controlBotRunning = 0
        xSum = 0
        time.sleep(2)


def ImageCallback(msg):
    print("Received an image!")
    try:
        # Convert your ROS Image message to OpenCV2
        cv2Img = bridge.imgmsg_to_cv2(msg, "bgr8")
        global recordFlag
        global trainSet
        if recordFlag:
            trainSet.append([cv2Img, joystickInput])
    except:
        pass
    print(joystickInput)

    while (True):
        cv2.imshow('test', cv2Img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        break



def main():
    rospy.init_node('rosjoy_listener')
    # Set up your subscriber and define its callback
    rospy.Subscriber("/joy", Joy, JoyCallback,queue_size = 1)
    rospy.Subscriber("/cmd_vel", Twist, TwistCallback)
    rospy.Subscriber("/camera/image_raw", Image, ImageCallback)

    rospy.spin()
    #while not rospy.is_shutdown():

if __name__ == "__main__":
    main()