import tflearn
from tflearn.layers.estimator import regression
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression


def network():

    width = 128
    height = 72
    #lr = 0.1
    network = tflearn.input_data(shape=[None, width,height,3], name='input')

    #network = conv_2d(network, 64, 3, activation='relu')
    #network = max_pool_2d(network, 3, strides=4)
   # network = conv_2d(network, 64, 3, activation='relu')
   # network = max_pool_2d(network, 3, strides=4)
   # network = conv_2d(network, 64, 3, activation='relu')
   #network = local_response_normalization(network)
    network = fully_connected(network, 10, activation='relu')
    network = dropout(network, 0.7)
    network = fully_connected(network, 10, activation='relu')
    network = dropout(network, 0.7)
    network = tflearn.fully_connected(network, 3, activation='softmax')

    momentum = tflearn.Momentum(learning_rate=0.01, lr_decay=0.96, decay_step=100)

    network = regression(network, optimizer=momentum,
                         loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(network, checkpoint_path='model_simplenet',
                        max_checkpoints=1, tensorboard_verbose=2, tensorboard_dir='log')
    return model