# -*- coding: utf-8 -*-
import numpy as np
import sys
import keras.models
import keras.backend as K
import random

'''
举例,有数组a
a=np.array([[0,9,1],[2,3,4],[1,3,2],[9,7,1]])
我知道b=np.array([1,1,0,2]),是每一行我想要的index
即我想要获得数组[9,3,1,1]
构造数组
c=np.arange(0,4)
d=np.stack((c,b),axis=0)
d为 [[0,1,2,3],[1,1,0,2]]

a[list(d)]即为一维的[9,3,1,1]
'''
    


class Agent:
    def __init__(self,network_model, q_values_func,memoryD,test_or_train,agent_model):
        self.q_network=network_model
        
        self.target_network=keras.models.clone_model(network_model)
        self.target_network.set_weights(network_model.get_weights())
        
        #for cnn
        #self.target_q_values_func=K.function([self.target_network.layers[0].input], [self.target_network.layers[5].output])
        #for nn
        self.target_q_values_func=K.function([self.target_network.layers[0].input], [self.target_network.layers[3].output])

        
        self.q_values_func=q_values_func
        self.memoryD=memoryD
        
        self.history_frame=[None]*4
        self.num_step=0
        self.update_times=0
        
        self.test_or_train=test_or_train
        self.agent_model=agent_model
    
    def do_compile(self,optimizer,loss_func):
        self.q_network.compile(optimizer=optimizer, loss=loss_func)
        self.target_network.compile(optimizer=optimizer, loss=loss_func)

    def load_weights(self, weights_file_name):
        self.q_network.load_weights(weights_file_name)
        self.update_target_network()

    def update_target_network(self):
        self.target_network.set_weights(self.q_network.get_weights())
    
    
    #用于test, 完全信赖model
    def greedy_policy(self,q_values):
        return np.argmax(q_values)
    
    
    # 用于train,即并不完全信任model,让agent有机会随机选择动作
    def greedy_epsilon_policy(self,q_values):
        epsilon=0.05
        rnd = random.random()
        if rnd<=epsilon:
            return random.randint(0, 4)
        
        return np.argmax(q_values)
    # 用于train,让agent随机选择的几率慢慢下降,从0.999下降到0.05,即我们对model选择的信任度随着训练时长逐渐上升
    def linear_greedy_epsilon_policy(self,q_values):
        if self.num_step>949000:
            epsilon=0.05
        else:
            epsilon=0.999-0.000001*self.num_step
            
        rnd = random.random()
        if rnd<=epsilon:
            return random.randint(0, 4)
        
        return np.argmax(q_values)
    
    
    def select_action(self,state):
        state=np.expand_dims(np.asarray(state),0)
        q_values=self.q_values_func([state])[0]
        return self.greedy_epsilon_policy(q_values)
        '''
        if self.test_or_train=='train':
            return self.greedy_epsilon_policy(q_values)
            #return self.linear_greedy_epsilon_policy(q_values)
        elif self.test_or_train=='test':
            return self.greedy_policy(q_values)
        else:
            print('In dqn.py, select_action function, wrong model!')
            sys.exit(0)
        '''
    
    def transform_actions(self,actions):
        one_hot_action = np.zeros((len(actions), 5), dtype='float32')
        one_hot_action[np.arange(len(actions), dtype='int'), actions] = 1

        return one_hot_action
    
    def process_new_frame(self,frame):
        if self.history_frame[0]==None:
            self.history_frame=[frame,frame,frame,frame]
        else:
            self.history_frame[0:3]=self.history_frame[1:]
            self.history_frame[-1]=frame
        return np.array(self.history_frame)
    
    
    
    '''
    def update(self):
        if self.agent_model=='dqn':
            self.update_dqn()
        elif self.agent_model=='ddqn':
            self.update_ddqn()
        else:
            print('In dqn.py, update function, wrong model!')
            sys.exit(0)
    '''
    
    def update(self):
        states, actions, rewards, new_states, is_terminals=self.memoryD.get_sample()
        # i.e.  action 3 -->  [0,0,0,1,0]
        actions=self.transform_actions(actions)
        
        #用旧的网络w-,选择值最高的q_value
        q_values=self.target_q_values_func([new_states])[0]
        max_q_values = np.max(q_values, axis=1)
        
        #把是terminal的q_value置为0
        max_q_values[is_terminals] = 0
        # gamma衰减因子  设置 0.99
        targets = rewards + 0.99* max_q_values
        targets = np.expand_dims(targets, axis=1)
        #在一个batch的数据上进行一次参数更新
        #因为我们是以一个batch,aka 32个经验数组去update的
        self.q_network.train_on_batch([states, actions], targets)
        
        if self.num_step%19000==0:
            self.update_times+=1
            self.update_target_network()
            '''
            #新的=====================================================================================
            if self.update_times>0 and self.update_times%30==0:
                self.q_network.save_weights('./model_weights_round_%d.h5' % (self.update_times))
            #=========================================================================================
            '''
    '''  
    为什么Double DQN?     
    https://blog.csdn.net/mike112223/article/details/92843720
    有时候会学习到不现实的高动作价值，这是因为在估计动作价值的时候包含了一步最大值操作，这会使得agent倾向于过估计
    当然过分乐观的价值估计其实本质上并不是一个问题。如果所有的values都均匀的高于相对的参考actions，那么对于policy来说并不会有影响
    然而问题就是过估计是不均匀的，而且并不集中在我们想要更多去学习的states里
    '''
    def update_ddqn(self):
        states, actions, rewards, new_states, is_terminals=self.memoryD.get_sample()
        # i.e.  action 3 -->  [0,0,0,1,0]
        actions=self.transform_actions(actions)
        
        #用新的权重w选择动作.  即选择q_value最大值对应的动作
        q_values=self.q_values_func([new_states])[0]
        max_actions = np.argmax(q_values, axis=1)
        
        #用获得的动作组合成index方便下面使用
        tmp=np.arange(0,len(max_actions))
        index=np.stack((tmp,max_actions),axis=0)
        
        #用旧的权重w-获得q_values
        target_q_values=self.target_q_values_func([new_states])[0]
        #不用最大值作为max_q_value,而是用刚才动作的index作为max_q_value
        max_q_values=target_q_values[list(index)]
        
        
        #把是terminal的q_value置为0
        max_q_values[is_terminals] = 0
        # gamma衰减因子  设置 0.99
        targets = rewards + 0.99* max_q_values
        targets = np.expand_dims(targets, axis=1)
        #在一个batch的数据上进行一次参数更新
        #因为我们是以一个batch,aka 32个经验数组去update的
        self.q_network.train_on_batch([states, actions], targets)
        
        if self.num_step%19000==0:
            self.update_times+=1
            self.update_target_network()
            #新的=====================================================================================
            if self.update_times>0 and self.update_times%30==0:
                self.q_network.save_weights('./model_weights_round_%d.h5' % (self.update_times))
            #=========================================================================================
        

        
                