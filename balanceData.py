import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle

train_data = np.load('trainSet.npy')

df = pd.DataFrame(train_data)
print(df.head())
print(Counter(df[1].apply(str)))

lefts = []
rights = []
forwards = []

shuffle(train_data)

for data in train_data:
    img = data[0]
    choice = data[1]

 #   if choice[1] > 0.1:
 #       lefts.append([img,choice])
 #   elif abs(choice[1]) <= 0.1:
 #       forwards.append([img,choice])
 #   elif choice[1] < -0.1:
 #       rights.append([img,choice])
 #   else:
 #       print('no matches')


    if choice[0] > 0.1:
        #links
        lefts.append([img,choice])
    elif abs(choice[0]+choice[1]) < 0.1:
        forwards.append([img,choice])

    elif choice[1] < -0.1:
        # rechts
        rights.append([img,choice])
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

np.save('trainSetBalanced.npy', final_data)