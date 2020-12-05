import numpy as np
import matplotlib.pyplot as plt
import cv2
from random import shuffle

train_data = np.load('Final_Training_Data_v1.npy', allow_pickle=True)
print (type(train_data))

new_left = []
new_right = []
new_forward = []
#new_nothing = []


for data in train_data:
    image = data[0]
    choice = data[1]

    if choice == [1, 0, 0, 0, 0]:
        new_left.append([image, [1, 0, 0]])
        flipped_img = cv2.flip(image, 1)
        flipped_choice = [0, 0, 1]
        new_right.append([flipped_img, flipped_choice])
    elif choice == [0, 1, 0, 0, 0]:
        new_forward.append([image, [0, 1, 0]])
    elif choice == [0, 0, 1, 0, 0]:
        new_right.append([image, [0, 0, 1]])
        flipped_img = cv2.flip(image, 1)
        flipped_choice = [1, 0, 0]
        new_left.append([flipped_img, flipped_choice])
    else:
        continue

shuffle(new_forward)
new_forward = new_forward[:int(1.5*len(new_left))]
    
total_data = np.concatenate((new_left, new_right, new_forward))
print ("Left : {}, Right : {}, Forward : {}".format(len(new_left), len(new_right), len(new_forward)))

print (len(total_data))
#fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
#ax.hist(total_data, bins=3, width=0.05)
#ax.set_title('Training Data')

np.save('Only_3_choice_data.npy', total_data)



