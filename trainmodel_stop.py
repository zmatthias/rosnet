import numpy as np
from network import network
import glob
from random import shuffle
from generate_sound import tts

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
stops = []

shuffle(totalTrainSet)

for data in totalTrainSet:
    img = data[0]
    joystickInput = data[1]

    if (joystickInput == [1,0,0,0]):
        lefts.append([img,joystickInput])

    if (joystickInput == [0,0,1,0]):
        rights.append([img,joystickInput])

    if (joystickInput == [0,1,0,0]):
        forwards.append([img,joystickInput])

    if (joystickInput == [0,0,0,1]):
        stops.append([img,joystickInput])

print(len(lefts))
print(len(rights))
print(len(forwards))
print(len(stops))

maxLen=max(len(lefts),len(rights),len(forwards),len(stops))

while(len(lefts)<maxLen):
    lefts.extend(forwards)

while(len(rights)<maxLen):
    rights.extend(forwards)

while(len(forwards)<maxLen):
    forwards.extend(forwards)

while(len(stops)<maxLen):
    stops.extend(forwards)


forwards = forwards[:maxLen]
lefts = lefts[:maxLen]
rights = rights[:maxLen]
stops = stops[:maxLen]

#forwards = forwards[:len(lefts)][:len(rights)]
#lefts = lefts[:len(forwards)]
#rights = rights[:len(forwards)]


print(len(lefts))
print(len(rights))
print(len(forwards))

finalData = forwards + lefts + rights
shuffle(finalData)
print(len(finalData))

lenTrainSet=int((len(finalData)/2)-1)
trainSet = finalData[0:lenTrainSet]
valSet = finalData[lenTrainSet:-1]

# each image is reshaped from a matrix into a single array
trainImageSet = np.array([i[0] for i in trainSet]).reshape(-1, inputWidth, inputHeight, 3)
trainSolutionSet = np.array([i[1] for i in trainSet])

valImageSet = np.array([i[0] for i in valSet]).reshape(-1, inputWidth, inputHeight, 3)
valSolutionSet = np.array([i[1] for i in valSet])

modelName = '/home/z/Dropbox/bachelorarbeit/catkin_ws/src/rosnet/src/model/network.model'
model = network()

#savedEvalSet = np.load("evalSet.npy")
#evalImageSet = np.array([i[0] for i in savedEvalSet]).reshape(-1, inputWidth, inputHeight, 1)
#evalSolutionSet = np.array([i[1] for i in savedEvalSet])


for epochs in range (1,20):
    model.fit({'input': trainImageSet}, {'targets': trainSolutionSet}, n_epoch=1, validation_set=({'input': valImageSet}, {'targets': valSolutionSet}),
              snapshot_step=500, show_metric=True, run_id=modelName)

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
