import numpy as np
import cv2
import time
from grabscreen import grab_screen
from getkeys import key_check #to check the keys we press while controlling the vehicle
import os
from directkeys import PressKey, ReleaseKey, W, A, S, D
from alexnet import alexnet, alexnet2
import tensorflow as tf
from path_color_coder import lane_finder

WIDTH = 80*2
HEIGHT = 60*2
LR = 1e-3
EPOCHS = 25

MODEL_NAME = 'gta5-model_1'

def straight(t=0.5):
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)
    time.sleep(t)
    ReleaseKey(W)

def left(t):
    PressKey(A)
    PressKey(W) #always want to go forward
    ReleaseKey(D)
    time.sleep(t)
    ReleaseKey(A)
    ReleaseKey(W)
    

def right(t):
    PressKey(D)
    PressKey(W)
    ReleaseKey(A)
    time.sleep(t)
    ReleaseKey(D)
    ReleaseKey(W)
    


model = alexnet2(WIDTH, HEIGHT, LR)
model.load(MODEL_NAME)

    
def main():
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)
        
    last_time = time.time()
    paused = False
    
    while True:

        if not paused:
            screen = grab_screen(region=(0,510,170,640))
            
            screen = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR) 
            screen = lane_finder(screen)
            screen = cv2.resize(screen, (160, 120))
            

            print('Frame took {} seconds'.format(time.time()-last_time))
            last_time = time.time()

            prediction = model.predict([screen.reshape(WIDTH, HEIGHT, 3)])[0] #pass in the feature image (real time image --> the screen)
            moves = list(np.around(prediction)) #round off to 0 or 1 (the original output would be like [0.9334, 0.02334, 0.0564] and it would make it [1, 0, 0])
        
            print (moves, prediction)

            if moves == [1,0,0]:
                if prediction >= 0.85:
                    left(t=0.5)
                elif 0.5 < prediction < 0.8:
                    left(t=0.2)
                else:
                    left(t=0.05)
            elif moves == [0,1,0]:
                if prediction >= 0.85:
                    straight(t=0.5)
                elif 0.5 < prediction < 0.8:
                    straight(t=0.2)
                else:
                    straight(t=0.05)
            elif moves == [0,0,1]:
                if prediction >= 0.85:
                    right(t=0.5)
                elif 0.5 < prediction < 0.8:
                    right(t=0.2)
                else:
                    right(t=0.05)
            else:
                PressKey(W)
                time.sleep(0.2)
                ReleaseKey(W)

        keys = key_check()

        if 'T' in keys:
            if paused:
                paused = False
                time.sleep(1)
            else:
                paused = True
                ReleaseKey(A)
                ReleaseKey(W)
                ReleaseKey(D)
                time.sleep(1)


main()

