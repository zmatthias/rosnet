import tflearn
from tflearn.layers.estimator import regression
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression


def network():

    width = 128
    height = 72
    lr = 0.001
    network = tflearn.input_data(shape=[None, width,height,3], name='input')
    network = fully_connected(network, 10, activation='relu')
    network = dropout(network, 0.5)
    network = fully_connected(network, 10, activation='relu')
    network = dropout(network, 0.5)

    #network = conv_2d(network, 96, 11, strides=4, activation='relu')
    #network = max_pool_2d(network, 3, strides=2)
    #network = local_response_normalization(network)
    #network = conv_2d(network, 256, 5, activation='relu')
    #network = max_pool_2d(network, 3, strides=2)
    #network = local_response_normalization(network)
    #network = conv_2d(network, 384, 3, activation='relu')
    #network = conv_2d(network, 384, 3, activation='relu')
    #network = conv_2d(network, 256, 3, activation='relu')
    #network = max_pool_2d(network, 3, strides=2)
    #network = local_response_normalization(network)
    #network = fully_connected(network, 4096, activation='tanh')
    #network = dropout(network, 0.5)
    #network = fully_connected(network, 4096, activation='tanh')
    #network = dropout(network, 0.5)

    network = tflearn.fully_connected(network, 1, activation='softmax')
    network = regression(network, optimizer='momentum',
                         loss='binary_crossentropy',
                         learning_rate=lr, name='targets')

    model = tflearn.DNN(network, checkpoint_path='model_simplenet',
                        max_checkpoints=1, tensorboard_verbose=2, tensorboard_dir='log')
    return model