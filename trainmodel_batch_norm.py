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


def TrainNetwork():
    shuffle(totalTrainSet)
    lenTrainSet=int((len(totalTrainSet)/2)-1)
    trainSet = totalTrainSet[0:lenTrainSet]
    valSet = totalTrainSet[lenTrainSet:-1]

    # each image is reshaped from a matrix into a single array
    trainImageSet = np.array([i[0] for i in trainSet]).reshape(-1, inputWidth, inputHeight, 3)
    trainSolutionSet = np.array([i[1] for i in trainSet])

    valImageSet = np.array([i[0] for i in valSet]).reshape(-1, inputWidth, inputHeight, 3)
    valSolutionSet = np.array([i[1] for i in valSet])

    modelName = '/home/z/Dropbox/bachelorarbeit/catkin_ws/src/rosnet/src/model/network.model'
    model = network()
    model.load(modelName)

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

TrainNetwork()