# -*- coding: utf-8 -*-
import numpy as np

class MemoryD:

    def __init__(self,frame_size):

        #设置数组总大小
        self.max_size = 100000
        #设置每帧大小
        self.frame_size=frame_size
        #设置每4帧组成一个状态
        self.frame_num = 4
        self.mem_size = (self.max_size + self.frame_num-1);
        
        # two pointers        
        self.start = 0
        # End point to the next position.
        # The content doesn't change when end points at it but change
        # when end points move forward from it.
        self.end = 0
        #数组是否满
        self.full = False
        
        #map size 5*5
        self.mem_frame = np.ones((self.mem_size,self.frame_size , self.frame_size), dtype=np.float32)
        self.mem_action = np.ones(self.mem_size, dtype=np.int8)
        self.mem_reward = np.ones(self.mem_size, dtype=np.float32)
        self.mem_terminal = np.ones(self.mem_size, dtype=np.bool)
        

    def append(self, frame, action, reward, is_terminal):
        if self.start == 0 and self.end == 0: # the first frame
            # 1 2 3 S E
            #因为是first time 所以第一帧重复4次
            for i in range(self.frame_num-1):
                self.mem_frame[i] = frame
                self.start = (self.start + 1) % self.mem_size
            self.mem_frame[self.start] = frame
            self.mem_action[self.start] = action
            self.mem_reward[self.start] = reward
            self.mem_terminal[self.start] = is_terminal
            self.end = (self.start + 1) % self.mem_size
        else:
            # Case 1:  1 2 3 S ... E
            # Case 2:  ... E 1 2 3 S ...
            self.mem_frame[self.end] = frame
            self.mem_action[self.end] = action
            self.mem_reward[self.end] = reward
            self.mem_terminal[self.end] = is_terminal
            self.end = (self.end + 1) % self.mem_size
            if self.end > 0 and self.end < self.start:
                self.full = True

            if self.full:
                self.start = (self.start + 1) % self.mem_size

    #batch_size,  要求抽样的数目
    def get_sample(self, batch_size=32, indexes=None):
        if self.end == 0 and self.start == 0:
            # state, action, reward, next_state, is_terminal
            return None, None, None, None, None
        else:
            #count 能够抽取的样本数
            count = 0
            #数组还没有满
            if self.end > self.start:
                count = self.end - self.start
            else:
                count = self.max_size

            if count <= batch_size:
                indices = np.arange(0, count-1)
            else:
                #indices range is 0 ... count-2
                indices = np.random.randint(0, count-1, size=batch_size)

            # 4 is the current state frame because of our design
            indices_5 = (self.start + indices + 1) % self.mem_size
            indices_4 = (self.start + indices) % self.mem_size
            indices_3 = (self.start + indices - 1) % self.mem_size
            indices_2 = (self.start + indices - 2) % self.mem_size
            indices_1 = (self.start + indices - 3) % self.mem_size
            frame_5 = self.mem_frame[indices_5]
            frame_4 = self.mem_frame[indices_4]
            frame_3 = self.mem_frame[indices_3]
            frame_2 = self.mem_frame[indices_2]
            frame_1 = self.mem_frame[indices_1]

            state_list = np.array([frame_1, frame_2, frame_3, frame_4])
            state_list = np.transpose(state_list, [1,0,2,3])

            next_state_list = np.array([frame_2, frame_3, frame_4, frame_5])
            next_state_list = np.transpose(next_state_list, [1,0,2,3])

            action_list = self.mem_action[indices_4]
            reward_list = self.mem_reward[indices_4]
            terminal_list = self.mem_terminal[indices_4]

            return state_list, action_list, reward_list, next_state_list, terminal_list

    def clear(self):
        self.start = 0
        self.end = 0
        self.full = False
