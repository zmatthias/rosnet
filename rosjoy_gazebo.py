#! /usr/bin/env python
import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
import cv2
from sensor_msgs.msg import Joy, Image
from geometry_msgs.msg import Twist
import os
from subprocess import Popen
import time
import datetime
from generateSound import tts
import glob

bridge = CvBridge()
trainSet = []
joystickInput = [0,0,0]
controlBotRunning = 0
startPreviouslyPressed = 0
xPreviouslyPressed = 0
recordFlag = 0
saveFlag = 0
trainFlag = 0
pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

def JoyCallback(msg):
    r1 = msg.buttons[11]
    delta = msg.buttons[12]
    o = msg.buttons[13]
    x = msg.buttons[14]
    square = msg.buttons[15]
    select = msg.buttons[0]
    start  = msg.buttons[3]

   # print("R1: \t {}".format(r1))
   # print("delta:\t {}".format(delta))
   # print("O: \t {}".format(o))
   # print("X: \t {}".format(x))
    global controlBotRunning
    global startPreviouslyPressed
    global xPreviouslyPressed

    if start and not controlBotRunning and not startPreviouslyPressed:
        controlBotRunning = 1
        global xterm
        xterm = Popen(["xterm", "-e", "rosrun rosnet controlbotGazebo.py"])
        startPreviouslyPressed = 1
        tts("autonomous mode")
        time.sleep(2)

    if start and controlBotRunning and not startPreviouslyPressed:
        xterm.terminate()
        pubStop()
        controlBotRunning = 0
        startPreviouslyPressed = 1
        tts("manual mode")

    if not start:
        startPreviouslyPressed = 0

    global trainFlag
    global xtermTrain
    if o and not trainFlag:
        xtermTrain = Popen(["xterm", "-e", "python /home/z/Dropbox/bachelorarbeit/catkin_ws/src/rosnet/src/trainmodel.py"])
        trainFlag = 1

    if o and trainFlag:
        poll = xtermTrain.poll()
        if poll != None:
            print("terminated")
            trainFlag = 0

    if delta:
        global recordFlag
        recordFlag = 1
    if not delta:
        if recordFlag:
            saveTrainingData()
        recordFlag = 0

    if x and not xPreviouslyPressed:
       # liste = glob.glob("/home/z/Dropbox/bachelorarbeit/catkin_ws/src/rosnet/src/data/*.npy")
        file_list = sorted(glob.glob("/home/z/Dropbox/bachelorarbeit/catkin_ws/src/rosnet/src/data/*.npy"))
        print(file_list)
        print("deleting last set")
        try:
            os.remove(file_list[-1])
        except:
            pass

        xPreviouslyPressed = 1

    if not x:
        xPreviouslyPressed = 0

    if select and square:
        os.system("mv /home/z/Dropbox/bachelorarbeit/catkin_ws/src/rosnet/src/data/*.npy /home/z/Dropbox/bachelorarbeit/catkin_ws/src/rosnet/src/data/archive")
        tts("training sets, archived")


def ImageCallback(msg):
    try:
        # Convert your ROS Image message to OpenCV2
        cv2Img = bridge.imgmsg_to_cv2(msg, "bgr8")
        global recordFlag
        global trainSet
        if recordFlag:
            trainSet.append([cv2Img, joystickInput])
            print("PHOTO")
    except:
        pass

    while (True):
        cv2Img = cv2.resize(cv2Img, (512, 288), interpolation=cv2.INTER_NEAREST)
        cv2.imshow('test', cv2Img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        break
def TwistCallback(msg):
    global joystickInput

    if (msg.angular.z > 0.15):
        joystickInput = [1, 0, 0, 0]
    elif (msg.angular.z < -0.15):
        joystickInput = [0, 0, 1, 0]
    elif (msg.linear.x > 0.01):
        joystickInput = [0, 1, 0, 0]
    else:
        joystickInput = [0, 0, 0, 1]

    print(joystickInput)

def pubStop():
    controlMsg = Twist()
    controlMsg.linear.x = 0
    controlMsg.angular.z = 0
    global pub
    pub.publish(controlMsg)

def trainNet():
    print("Train!")

def saveTrainingData():
    global trainSet
    if len(trainSet) >= 1:
        dateString = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        np.save("/home/z/Dropbox/bachelorarbeit/catkin_ws/src/rosnet/src/data/" + dateString + "_trainSet.npy", trainSet)
        print("{} Frames saved".format(len(trainSet)))
        trainSet = []

def main():
    rospy.init_node('rosjoy_listener')
    rospy.Subscriber("/joy", Joy, JoyCallback,queue_size = 1)
    rospy.Subscriber("/cmd_vel", Twist, TwistCallback)
    rospy.Subscriber("/camera/image_raw", Image, ImageCallback)
    rospy.spin()

if __name__ == "__main__":
    main()