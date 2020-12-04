from grabscreen import grab_screen
import numpy as np
import cv2
import time
import os
from getkeys import key_check
from path_color_coder import lane_finder

def keys_to_output(keys):
    #[A:left, W:straight, D:right, S:reverse, ' ':brakes] boolean values
    output = [0, 0, 0, 0, 0] #initially all A,W,D,S, SpaceBar are zeros

    if 'A' in keys:
        output[0] = 1
    elif 'D' in keys:
        output[2] = 1
    elif 'W' in keys:
        output[1] = 1
    elif 'S' in keys:
        output[3] == 1
    elif ' ' in keys:
        output[4] == 1
    else:
        output = [0, 0, 0, 0, 0]
        
    return output


file_name = 'waypoint_TrainingData_2.npy' #store our training data

if os.path.isfile(file_name):
    print ('File exists, loading previous data!')
    training_data = list(np.load(file_name))
else:
    print ('File does not exist, starting fresh')
    training_data = []

    
#####     RECORD DATA      ###########
def main():
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)
        
    last_time = time.time()

    while True:
        
        screen = grab_screen(region=(0,510,170,640))
        #convert from RGB to BGR since lane finder used BGR to HSV
        screen = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR) 
        new_screen = lane_finder(screen)
        
        screen = cv2.resize(screen, (160, 120))
    
        keys = key_check()
        output = keys_to_output(keys)

        training_data.append([screen, output])

        print('Frame took {} seconds'.format(time.time()-last_time))
        last_time = time.time()

        
        # Reminding every 500 training data collected
        if len(training_data) % 500 == 0:
            print (len(training_data))
            np.save(file_name, training_data)
        
        
        cv2.imshow('window', new_screen)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

main()
