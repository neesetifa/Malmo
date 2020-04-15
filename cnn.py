# -*- coding: utf-8 -*-
import tensorflow as tf
from keras.layers import (Activation, Conv2D, Dense, Flatten, Input)
from keras.layers.merge import dot
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K

'''
Code reference from:
"Beat Atari with Deep Reinforcement Learning!" by Adrien Lucas Ecoffet
Following is the link:
https://becominghuman.ai/lets-build-an-atari-ai-part-1-dqn-df57e8ff3b26
'''

def cnn_model(frame=4, input_shape=[5,5], num_actions=5):  
    with tf.name_scope('deep_q_network'):
        with tf.name_scope('input'):
            # 5*5*4
            input_state = Input(shape=(frame, input_shape[0], input_shape[1]))
            input_action = Input(shape=(num_actions,))

        with tf.name_scope('conv1'):
            conv1 = Conv2D(16, (3, 3), data_format='channels_first', kernel_initializer='glorot_uniform', activation='relu', padding='valid', strides=(1, 1))(input_state)

        with tf.name_scope('conv2'):
            conv2 = Conv2D(32, (2, 2), data_format='channels_first', kernel_initializer='glorot_uniform', activation='relu', padding='valid', strides=(1, 1))(conv1)

        with tf.name_scope('fc'):
            flattened = Flatten()(conv2)
            dense1 = Dense(128, kernel_initializer='glorot_uniform', activation='relu')(flattened)

        with tf.name_scope('output'):
            q_values = Dense(num_actions, kernel_initializer='glorot_uniform', activation=None)(dense1)
            q_v = dot([q_values, input_action], axes=1)

        network_model = Model(inputs=[input_state, input_action], outputs=q_v)
        q_values_func = K.function([input_state], [q_values])

    network_model.summary()

    return network_model, q_values_func

#print summary
#cnn_model()


def nn_model(frame=4, input_shape=[5,5], num_actions=5):  

    with tf.name_scope('deep_q_network'):
        with tf.name_scope('input'):
            # 5*5*4
            input_state = Input(shape=(frame, input_shape[0], input_shape[1]))
            input_action = Input(shape=(num_actions,))
   
 
        #本层参数初始化,用的是glorot_uniform,这个选项是默认的
        with tf.name_scope('fc2'):
            flattened = Flatten()(input_state)
            dense2 = Dense(128, kernel_initializer='glorot_uniform', activation='relu')(flattened)

        with tf.name_scope('output'):
            q_values = Dense(num_actions,activation=None)(dense2)
            q_v = dot([q_values, input_action], axes=1)


        network_model = Model(inputs=[input_state, input_action], outputs=q_v) #方案1,输入state,action,输出一个q_value
        q_values_func = K.function([input_state], [q_values])  #方案2,输入一个state,输出一系列[action,q_value]

    network_model.summary()

    return network_model, q_values_func

nn_model()