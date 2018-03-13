import tflearn
from tflearn.layers.estimator import regression
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

def network():

    width = 128
    height = 72
    network = tflearn.input_data(shape=[None, width,height,3], name='input')
    network = tflearn.batch_normalization(network)


    network = fully_connected(network, 10, activation='relu')
    network = dropout(network, 0.6)
    network = fully_connected(network, 10, activation='relu')
    network = dropout(network, 0.6)
    network = tflearn.fully_connected(network, 3, activation='softmax')

    momentum = tflearn.Momentum(learning_rate=0.0)
   # momentum = tflearn.Momentum(learning_rate=0.000001, lr_decay=0.80, decay_step=400)

    network = regression(network, optimizer=momentum,
                         loss='categorical_crossentropy', name='targets')
                          #loss='weak_cross_entropy_2d', name='targets')

    model = tflearn.DNN(network, checkpoint_path='model_simplenet',
                        max_checkpoints=1, tensorboard_verbose=2, tensorboard_dir='log')
    return model
