
import numpy as np
from alexnet import alexnet2
from random import shuffle


WIDTH = 80*2
HEIGHT = 60*2
LR = 1e-3
EPOCHS = 25
#MODEL_NAME = 'pygta5-car-{}-{}-{}-epochs-300K-data.model'.format(LR, 'alexnetv2', EPOCHS)
MODEL_NAME = 'Only_3_choice_data.npy' 
    
model = alexnet2(WIDTH, HEIGHT, LR, output=3)

def train_test_split(data, test_percentage):
    shuffle(data)
    length = len(data)
    test_length = int(length*test_percentage)
    train = data[:-test_length]
    test = data[-test_length:]
    return train, test

# We have total 22 training data npy files (1-22)
#hm_data = 22
total = 1    
    
for i in range(EPOCHS):
    for i in range(1,total+1):
        #train_data = np.load('sentdex_data/training_data/training_data-{}-balanced.npy'.format(i), allow_pickle=True)
        train_data = np.load(MODEL_NAME, allow_pickle=True)
        train, test = train_test_split(train_data, 0.2)
        print (len(train), len(test))

        X = np.array([i[0] for i in train]).reshape(-1, WIDTH, HEIGHT, 3) #index 0 has the features
        Y = np.array([i[1] for i in train]) #index 1 has the labels
        print (X.shape, Y.shape)

        test_x = np.array([i[0] for i in test]).reshape(-1, WIDTH, HEIGHT, 3)
        test_y = np.array([i[1] for i in test])
        print (test_x.shape, test_y.shape)

        model.fit({'input': X}, {'targets': Y}, n_epoch=1, validation_set=({'input': test_x}, {'targets': test_y}), 
                 snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

        model.save(MODEL_NAME)


# tensorboard is stored as 'log' folder inside the folder containing alexnet.py

# >> tensorboard --logdir=foo:C:/Users/LENOVO/Desktop/gtaVpy/log
# Step1 : Run the above in cmd/terminal
# Step2 : It will give you a local host (like 6006). Use that in web browser to access the TENSORBOARD


model.save(MODEL_NAME)
