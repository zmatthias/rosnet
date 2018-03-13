import numpy as np
from network_nonstop import network
import glob
from random import shuffle
from generate_sound import tts
import cv2

inputWidth = 128
inputHeight = 72

file_list = glob.glob("/home/z/Dropbox/bachelorarbeit/catkin_ws/src/rosnet/src/data/*.npy")
print(file_list)

totalTrainSet = []
for file_path in file_list:
    totalTrainSet.extend(np.load(file_path))

print("Trainsets have a total length of {}".format(len(totalTrainSet)))
tts("Starting training")
tts("Training sets have a total length of {} Frames".format(len(totalTrainSet)))

lefts = []
rights = []
forwards = []
balancedData = []
mirroredData = []
augmentedData = []

def BalanceData():
    global totalTrainSet
    global lefts
    global rights
    global forwards
    global balancedData
    shuffle(totalTrainSet)

    for data in totalTrainSet:
        img = data[0]
        joystickInput = data[1]

        if (joystickInput == [1,0,0]):
            lefts.append([img,joystickInput])

        if (joystickInput == [0,0,1]):
            rights.append([img,joystickInput])

        if (joystickInput == [0,1,0]):
            forwards.append([img,joystickInput])

    print(len(lefts))
    print(len(rights))
    print(len(forwards))

    maxLen=max(len(lefts),len(rights),len(forwards))

    while(len(lefts)<maxLen):
        lefts.extend(forwards)

    while(len(rights)<maxLen):
        rights.extend(forwards)

    while(len(forwards)<maxLen):
        forwards.extend(forwards)

    forwards = forwards[:maxLen]
    lefts = lefts[:maxLen]
    rights = rights[:maxLen]

    print(len(lefts))
    print(len(rights))
    print(len(forwards))

    balancedData = forwards + lefts + rights
    shuffle(balancedData)
    print("balanced data length")
    print(len(balancedData))

def AddNoise(image,noise_factor):
    m = (50, 50, 50)
    s = (50, 50, 50)
    image_template = np.copy(image)
    noise = cv2.randn(image_template, m, s)
    noisy_image = cv2.addWeighted(image,(1-noise_factor),noise,noise_factor,0)

    return noisy_image

def adjust_gamma(image, gamma=1.0):

   invGamma = 1.0 / gamma
   table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")

   return cv2.LUT(image, table)

#def AugmentDataNoise():
 #   global mirroredData
 #   for data in mirroredData:
 #       image = data[0]
 #       image = AddNoise(image,0.7)
 #       augmentedData.append([image, data[1]])

def AugmentDataGamma():
    global balancedData
    for data in balancedData:
        image_brighter = data[0]
        image_darker = np.copy(data[0])
        image_brighter = adjust_gamma(image_brighter,10)
        image_darker = adjust_gamma(image_darker,0.1)
        augmentedData.append([image_brighter, data[1]])
        augmentedData.append([image_darker, data[1]])
    augmentedData.extend(balancedData)
    shuffle(augmentedData)

#def MirrorData():
#    for data in balancedData:
#        image = data[0]
#        image = cv2.flip(image, 1)
#        image = cv2.line(image, (0,0), (100,100), (0,0,255),4)
#        if data[1] == [1,0,0]:
#            data[1] = [0,0,1]
#        if data[1] == [0,0,1]:
#            data[1] = [1,0,0]
#
#        mirroredData.append([image,data[1]])
#    mirroredData.extend(balancedData)
#    #shuffle(mirroredData)

def TrainNetwork():
    global augmentedData
    shuffle(augmentedData)
    lenTrainSet=int((len(augmentedData)/2)-1)
    trainSet = augmentedData[0:lenTrainSet]
    valSet = augmentedData[lenTrainSet:-1]

    # each image is reshaped from a matrix into a single array
    trainImageSet = np.array([i[0] for i in trainSet]).reshape(-1, inputWidth, inputHeight, 3)
    trainSolutionSet = np.array([i[1] for i in trainSet])

    valImageSet = np.array([i[0] for i in valSet]).reshape(-1, inputWidth, inputHeight, 3)
    valSolutionSet = np.array([i[1] for i in valSet])

    modelName = '/home/z/Dropbox/bachelorarbeit/catkin_ws/src/rosnet/src/model/network.model'
    model = network()
#    model.load(modelName)

    #savedEvalSet = np.load("evalSet.npy")
    #evalImageSet = np.array([i[0] for i in savedEvalSet]).reshape(-1, inputWidth, inputHeight, 1)
    #evalSolutionSet = np.array([i[1] for i in savedEvalSet])

    for epochs in range (1,3000):
        if 100+epochs**2 < 1000:
            batchSize = 50+epochs**2
        else:
            batchSize = 1000 ##8GB

        print(batchSize)
        model.fit({'input': trainImageSet}, {'targets': trainSolutionSet}, n_epoch=1, validation_set=({'input': valImageSet}, {'targets': valSolutionSet}),
                  snapshot_step=500, show_metric=True, run_id=modelName, batch_size=batchSize, shuffle=True)
        model.save(modelName)


    #    evalScore = model.evaluate(evalImageSet, evalSolutionSet)[0]
    #    print(evalScore)
    #
    #    valScore = model.evaluate(valImageSet, valSolutionSet)[0]
    #    print(valScore)

    #    if(evalScore > 0.1) and (valScore> 0.1):
    #        print("{},{}".format(format(evalScore, '.4f'),format(valScore, '.4f')))
    #        f = open('epochslog_2x10_long.csv','a')
    #        f.write("{},{},{}\n".format(format(epochs),format(evalScore, '.4f'),format(valScore, '.4f')))
    #        f.close()
    model.save(modelName)
    tts("Training complete")
    valScore = model.evaluate(valImageSet, valSolutionSet)[0]
    tts("Validation accuracy is, {0:.1f}%".format(valScore*100))

BalanceData()
#MirrorData()
AugmentDataGamma()
#AugmentDataNoise()
TrainNetwork()