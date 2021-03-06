import numpy as np
from network import network
import glob

inputWidth = 128
inputHeight = 72

file_list = glob.glob("data/*.npy")
print(file_list)

savedTrainSet = []
for file_path in file_list:
    savedTrainSet.extend(np.load(file_path))

print(len(savedTrainSet))
#savedTrainSet = np.load("data/2018_02_13_18_50_34_trainSet.npy")


lenTrainSet=int((len(savedTrainSet)/2)-1)
print(lenTrainSet)
trainSet = savedTrainSet[0:lenTrainSet]
valSet = savedTrainSet[lenTrainSet:-1]

# each image is reshaped from a matrix into a single array
trainImageSet = np.array([i[0] for i in trainSet]).reshape(-1, inputWidth, inputHeight, 3)
trainSolutionSet = np.array([i[1] for i in trainSet])

valImageSet = np.array([i[0] for i in valSet]).reshape(-1, inputWidth, inputHeight, 3)
valSolutionSet = np.array([i[1] for i in valSet])

modelName = 'model/network.model'
model = network()

#savedEvalSet = np.load("evalSet.npy")
#evalImageSet = np.array([i[0] for i in savedEvalSet]).reshape(-1, inputWidth, inputHeight, 1)
#evalSolutionSet = np.array([i[1] for i in savedEvalSet])


for epochs in range (1,10):
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