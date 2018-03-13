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
model.load('/home/z/Dropbox/bachelorarbeit/catkin_ws/src/rosnet/src/model/network.model')
inputWidth = 128
inputHeight = 72

pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)


def printPrediction(prediction):
    forwardCertainty = prediction[1] * 100
    if (prediction[0] - prediction[2]) > 0:

        turningString = "left"
    else:
        turningString = "right"

    turningCertainty = abs((prediction[0] - prediction[2])) * 100
    stoppingCertainty = prediction[3]*100

    print("Forward: {}% \t \t Turn {}: {}% \t Stop {}% ".format(format(forwardCertainty, '.0f'), turningString,
                                                    format(turningCertainty, '.0f'), format(stoppingCertainty, '.0f')))


def ImageCallback(msg):
    print("Received an image!")
    try:
        # Convert your ROS Image message to OpenCV2
        cv2Img = bridge.imgmsg_to_cv2(msg, "bgr8")
    except:
        pass

    prediction = model.predict(cv2Img.reshape(-1, inputWidth, inputHeight, 3))[0]
    printPrediction(prediction)

    controlMsg = Twist()

    if prediction[3] < 0.5: #if stop neuron not on
        controlMsg.linear.x = 0.05 * prediction[1]
        controlMsg.angular.z = 1 * (prediction[0] - prediction[2])
    else:
        controlMsg.linear.x = 0
        controlMsg.angular.z = 0

    global pub
   # pub.publish(controlMsg)
    cv2Img = cv2.resize(cv2Img, (512, 288), interpolation=cv2.INTER_NEAREST)

    while True:
        cv2.imshow('frame', cv2Img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        break


def StopBot():
    print "shutdown!"
    controlMsg = Twist()
    controlMsg.linear.x = 0
    controlMsg.angular.z = 0
    pub.publish(controlMsg)


def main():
    rospy.init_node('controlbot')
    # Define your image topic
    image_topic = "/camera/image"
    rospy.on_shutdown(StopBot)

    # Set up your subscriber and define its callback
    rospy.Subscriber(image_topic, Image, ImageCallback, queue_size=1, buff_size=2**24)

    # Spin until ctrl + c
    rospy.spin()


if __name__ == "__main__":
    main()