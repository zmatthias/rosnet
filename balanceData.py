import numpy as np
from random import shuffle
import os

train_data = np.load('trainSet.npy')


lefts = []
rights = []
forwards = []

shuffle(train_data)

for data in train_data:
    img = data[0]
    joystickInput = data[1]

    if (joystickInput == [1,0,0]):
        print(joystickInput)
        print("left")
        lefts.append([img,joystickInput])

    if (joystickInput == [0,0,1]):
        print(joystickInput)
        print("right")
        rights.append([img,joystickInput])

    if (joystickInput == [0,1,0]):
        print(joystickInput)
        print("forwards")
        forwards.append([img,joystickInput])

    else:
        print('no matches')

forwards = forwards[:len(lefts)][:len(rights)]
lefts = lefts[:len(forwards)]
rights = rights[:len(forwards)]

print(len(forwards))
print(len(lefts))
print(len(rights))

final_data = forwards + lefts + rights
shuffle(final_data)

try:
    os.remove("trainSet.npy")
except OSError:
    pass

np.save('trainSet.npy', final_data)