import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle
import cv2
#from dataAUgment import image_flip


##num_data = 6
##train_data = np.array([])
##for i in range(1, num_data+1):
##    data = "waypoint_TrainingData_{}.npy".format(i)
##    new_data = np.load(data, allow_pickle=True)
##    print (len(new_data))
##    train_data = np.concatenate((train_data, new_data))
##    print (len(train_data))

train_data_1 = np.load('waypoint_TrainingData_1.npy', allow_pickle=True)
train_data_2 = np.load('waypoint_TrainingData_2.npy', allow_pickle=True)
train_data_3 = np.load('waypoint_TrainingData_3.npy', allow_pickle=True)
train_data_4 = np.load('waypoint_TrainingData_4.npy', allow_pickle=True)
train_data_5 = np.load('waypoint_TrainingData_5.npy', allow_pickle=True)
train_data_6 = np.load('waypoint_TrainingData_6.npy', allow_pickle=True)
train_data = np.concatenate((train_data_1, train_data_2, train_data_3, train_data_4, train_data_5, train_data_6))

df = pd.DataFrame(train_data)
print (df.head())
print (Counter(df[1].apply(str)))




#df.to_csv(r'choice.csv', index=False)

length = len(train_data)
print ('Total # of data :', length)

lefts = []
rights = []
forwards = []
reverse = []
nothing = []
brake = []

# We want to equalise forward, left and right controlling
for data in train_data:
    img = data[0] #0th index has image
    choice = data[1] #1st index has [A,W,D, S, SpaceBar] value choices

    if choice == [1, 0, 0, 0, 0]:
        lefts.append([img, choice])
        print ('LEFT')
        flipped_img = cv2.flip(img, 1)
        flipped_choice = [0, 0, 1, 0, 0]
        rights.append([flipped_img, flipped_choice])
        print ('Added to RIGHT')
    elif choice == [0, 1, 0, 0, 0]:
        forwards.append([img, choice])
        print ('FORWARD')
    elif choice == [0, 0, 1, 0, 0]:
        rights.append([img, choice])
        print ('RIGHT')
        flipped_img = cv2.flip(img, 1)
        flipped_choice = [1, 0, 0, 0, 0]
        lefts.append([flipped_img, flipped_choice])
        print ('Added to LEFT')
    elif choice == [0, 0, 0, 1, 0]:
        reverse.append([img, choice])
        print ('REVERSE')
    elif choice == [0, 0, 0, 0, 1]:
        brake.append([img, choice])
        print ('BRAKE')
    else:
        nothing.append([img, choice])
        print ('STOP!!!!!')

    length -= 1
    print ('Data left :', length)



shuffle(forwards)
shuffle(lefts)
shuffle(rights)
shuffle(reverse)
shuffle(nothing)
shuffle(brake)

print ("Fowards : {}, Lefts : {}, Rights : {}, Nothing : {}, Reverse : {}, Brake : {}".format(len(forwards), len(lefts), len(rights), len(nothing), len(reverse), len(brake)))

forwards = forwards[:int(1.5*len(lefts))]
nothing = nothing[:int(0.5*len(lefts))]

print ("Fowards : {}, Lefts : {}, Rights : {}, Nothing : {}".format(len(forwards), len(lefts), len(rights), len(nothing)))
                                                      
final_data = np.concatenate((forwards, lefts, rights, nothing)) #combining all the arrays

df = pd.DataFrame(final_data)
df.to_csv(r'GPS-data.csv', index=False)

##
##shuffle(final_data)
##print (len(final_data))
##np.save('Final_Training_Data_v1.npy', final_data)
##




