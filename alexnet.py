####### pip install tflearn ######

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization

# Architecture : https://www.google.com/search?q=alex+alexnet&hl=en&sxsrf=ALeKk03QCZRSZik2JFyhU1nmkkCi7oX9Lw:1606926995508&source=lnms&tbm=isch&sa=X&ved=2ahUKEwiM3_zF3a_tAhUU6nMBHTWsAlkQ_AUoAXoECA4QAw&biw=1280&bih=578#imgrc=j8y8gy88lYqI-M

def alexnet(width, height, lr):
    network = input_data(shape=[None, width, height, 1], name='input') #(w,h) is (80, 60) set by us previously

    #conv_2d(input, # of filters, each filter/kernel size n for nxn, strides, activation)
    network = conv_2d(network, 96, 11, strides=4, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    
    network = conv_2d(network, 256, 5, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 256, 3, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
                                      #units
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.7)
    network = fully_connected(network, 3, activation='softmax')

    # Linear/Logistic Regression
    network = regression(network, optimizer='momentum',
                         loss='categorical_crossentropy',
                         learning_rate=lr, name='targets')

    # Gives a summary
    model = tflearn.DNN(network, checkpoint_path='model_alexnet',
                        max_checkpoints=1, tensorboard_verbose=2, tensorboard_dir='log')

    return model

###############################################################################

def alexnet2(width, height, lr, output=3):
    #network = input_data(shape=[None, width, height, 1], name='input') #(w,h) is (160, 120) set by us previously
    network = input_data(shape=[None, width, height, 3], name='input') #Alexnet default is 224x224x3 but iam passing 160x120x3

    #conv_2d(input, # of filters, each filter/kernel size n for nxn, strides, activation)
    network = conv_2d(network, 96, 11, strides=4, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    
    network = conv_2d(network, 256, 5, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 256, 3, activation='relu')
    network = max_pool_2d(network, 3, strides=2)

    network = conv_2d(network, 256, 5, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)

    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 256, 3, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
                                      #units
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.7)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.7)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.7)
    network = fully_connected(network, output, activation='softmax')

    # Linear/Logistic Regression
    network = regression(network, optimizer='momentum',
                         loss='categorical_crossentropy',
                         learning_rate=lr, name='targets')

    # Gives a summary
    model = tflearn.DNN(network, checkpoint_path='model_alexnet',
                        max_checkpoints=1, tensorboard_verbose=2, tensorboard_dir='log')

    return model
