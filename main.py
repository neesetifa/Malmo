# ----------------------------------- Reference ----------------------------------------
#
# XML Schema Documentation:
# https://microsoft.github.io/malmo/0.21.0/Schemas/MissionHandlers.html
# MalmoPython.AgentHost() documentation:
# http://microsoft.github.io/malmo/0.16.0/Documentation/classmalmo_1_1_agent_host.html
#
# --------------------------------------------------------------------------------------

import os
import sys
import json
import time
import math 
import errno
import random
import MalmoPython
import malmoutils # For video recording



#=========Core part import====================
from memoryD import MemoryD
from cnn import cnn_model,nn_model
from dqn import Agent
from helper import mean_huber_loss, get_frame
from keras.optimizers import Adam
#=============================================



### 5 X 5  
mapblock = '''
    <DrawingDecorator>
        <DrawCuboid x1="-5" y1="200" z1="-5" x2="5" y2="200" z2="5" type="lava"/>
        <DrawCuboid x1="-5" y1="226" z1="-5" x2="5" y2="230" z2="5" type="glowstone"/>
        <DrawCuboid x1="-4" y1="201" z1="-4" x2="4" y2="225" z2="4" type="stained_glass" colour="RED"/>
        <DrawCuboid x1="-2" y1="202" z1="-2" x2="2" y2="220" z2="2" type="air"/>
        <DrawEntity x="-1" y="202" z="2" type="Zombie"/>
        <DrawEntity x="1" y="202" z="2" type="Villager"/>
    </DrawingDecorator>
'''

# ----------------------------------- Functions ----------------------------------------

def GetMissionXML( mapblock, agent_host ):
    '''Generate Mission XML'''
    return '''<?xml version="1.0" encoding="UTF-8" ?>
    <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
        <About>
            <Summary>Save Villagers!</Summary>
        </About>
        <ServerSection>
            <ServerInitialConditions>
                <Time>
                    <StartTime>9000</StartTime>
                </Time>
            </ServerInitialConditions>
            <ServerHandlers>
                <FlatWorldGenerator forceReset="false" generatorString="3;7,220*1,5*3,2;3;,biome_1" />
                ''' + mapblock + '''
                <ServerQuitFromTimeUp timeLimitMs="45000"/>
                <ServerQuitWhenAnyAgentFinishes />
            </ServerHandlers>
        </ServerSection>
        <AgentSection mode="Survival">
            <Name>Agent Sirius</Name>
            <AgentStart>
                <Placement x="0" y="202" z="0"/>
                <Inventory>
                    <InventoryObject type="diamond_sword" slot="0" quantity="1"/>
                </Inventory>
            </AgentStart>
            <AgentHandlers>
                <ObservationFromNearbyEntities>
                    <Range name="Mobs" xrange="5" yrange="1" zrange="5" update_frequency="1"/>
                </ObservationFromNearbyEntities>
                <DiscreteMovementCommands>
                    <ModifierList type="deny-list">
                        <command>attack</command>
                        <command>move</command>
                        <command>turn</command>
                    </ModifierList>
                </DiscreteMovementCommands>
                <ContinuousMovementCommands>
                    <ModifierList type="allow-list">
                        <command>attack</command>
                        <command>move</command>
                        <command>turn</command>
                    </ModifierList>
                </ContinuousMovementCommands>
                </AgentHandlers>
        </AgentSection>

    </Mission>'''
    
def generate_random_start_position():
    result = []
    for _ in range(3):
        temp_list = []
        temp_list.append(random.uniform(-1.7, 2.7))
        temp_list.append(random.uniform(-1.7, 2.7))
        result.append(temp_list)
    return result

def check_overlap_position(state):
    overlap = False
    agent = state[0]
    villager = state[1]
    zombie = state[2]
    if math.sqrt((agent[0] - villager[0])**2 + (agent[1] - villager[1])**2) <= 0.5:
        overlap = True
    if math.sqrt((agent[0] - zombie[0])**2 + (agent[1] - zombie[1])**2) <= 0.5:
        overlap = True
    if math.sqrt((zombie[0] - villager[0])**2 + (zombie[1] - villager[1])**2) <= 0.5:
        overlap = True
    return overlap

def act(action):
    try:
        if action == 0: 
            # moving forward
            agent_host.sendCommand( "attack 1")
            agent_host.sendCommand( "move 0.6")
                   
        if action == 1:
            # moving backward
            agent_host.sendCommand( "attack 1")
            agent_host.sendCommand( "move -0.3")
        if action == 2:
            # turn left
            agent_host.sendCommand( "attack 1")
            agent_host.sendCommand("turn -0.5")
            #time.sleep(0.25)
            #agent_host.sendCommand("turn 0")
        if action == 3:
            # turn right
            agent_host.sendCommand( "attack 1")
            agent_host.sendCommand("turn 0.5")
            #time.sleep(0.25)
            #agent_host.sendCommand("turn 0")
        if action == 4:
            # stay still
            agent_host.sendCommand( "attack 1") 
    except RuntimeError as e:
        print("Failed to send command:", e)
        sys.exit()

def get_info(ob):   
    # Collecting information from world state observation
    information_list = []
    reward = 0 
    for item in ob['Mobs']:
        if item['name'] in {'Agent Sirius', 'Zombie', 'Villager'}:
            information_list.append(item['name'])          # Name of the entity
    information_list = sorted(information_list) # get infomation in order of agent, villager, and zombie
    # print(information_list)
    if "Villager" not in information_list:
        reward -= 50
    elif "Zombie" not in information_list:
        reward += 40
    elif "Agent Sirius" not in information_list:
        reward -= 40
    return len(information_list)<3, reward

def get_arr(ob):   
    # Collecting information from world state observation
    information_list = []
    info_set = set()
    for item in ob['Mobs']:
        temp_list = []
        if item['name'] in {'Agent Sirius', 'Zombie', 'Villager'} and item['name'] not in info_set:
            temp_list.append(item['name'])          # Name of the entity
            temp_list.append(item['x'])             
            temp_list.append(item['z'])
            information_list.append(temp_list)
            info_set.add(item['name'])
    information_list = sorted(information_list) # get infomation in order of agent, villager, and zombie
    #print(information_list)
    if len(information_list) >= 4:
        for i in information_list:
            if i == []:
                information_list.remove(i)
    if len(information_list) == 3:
        array = information_list
        del array[0][0]
        del array[1][0]
        del array[2][0]
        # print(array)
        return array
    if len(information_list) == 2:
        array = []
        if information_list[0][0] == "Agent Sirius":
            array.append([information_list[0][1], information_list[0][2]])
            if information_list[1][0] == "Villager":
                array.append([information_list[1][1], information_list[1][2]])
                array.append([-99, -99])
            elif information_list[1][0] == "Zombie":
                array.append([-99, -99])
                array.append([information_list[1][1], information_list[1][2]])
        else:
            array.append([-99, -99])
            array.append([information_list[0][1], information_list[0][2]])
            array.append([information_list[1][1], information_list[1][2]])
        return array
    if len(information_list) <= 1 :
        return [[-99, -99], [-99, -99], [-99, -99]]


def get_angle(ob,yaw_angle):
    if ob['Mobs'][0]['name']=='Agent Sirius' and ob['Mobs'][0]['yaw']!=yaw_angle[0]:
        diff=ob['Mobs'][0]['yaw']-yaw_angle[0]
        yaw_angle[0]=ob['Mobs'][0]['yaw']
        yaw_angle[1]+=diff
        if yaw_angle[1]>=360:
            yaw_angle[1]-=360
        elif yaw_angle[1]<0:
            yaw_angle[1]+=360

    return yaw_angle[1]

  
def set_reward(ob, agent_life, villager_life, zombie_life):
    # Collecting information from world state observation
    information_list = []
    reward = 0
    #if villager_life[0] != 0:
        #reward += 0.02
    for item in ob['Mobs']:
        temp_list = []
        if item['name'] in {'Agent Sirius', 'Villager', 'Zombie'}:
            temp_list.append(item['name'])          # Name of the entity
            temp_list.append(item['life'])          # Life of the entity
            information_list.append(temp_list)
    information_list = sorted(information_list) # get infomation in order of agent, villager, and zombie
    # print(information_list)
    if len(information_list) != 3:
        return reward
    if information_list[0][1] < agent_life[0]:
        #print(information_list[0][1] ,agent_life[0])
        reward -= 5
        agent_life[0] = information_list[0][1]
    if information_list[1][1] < villager_life[0]:
        reward -= 15
        villager_life[0] = information_list[1][1]
    if information_list[2][1] < zombie_life[0]:
        reward += 10
        zombie_life[0] = information_list[2][1]

    return reward

    
# ------------------------------ Variable Declaration ----------------------------------
NUM_OF_ZOMBIES = 1
NUM_OF_VILLAGERS = 1
agent_host = MalmoPython.AgentHost()
malmoutils.parse_command_line(agent_host)
validate = True

num_reps = 300

#=======core part initialization====================================
#input size 5*5, you can change the size here
memory=MemoryD(5)
network_model, q_values_func = nn_model(input_shape=[5,5])

agent=Agent(network_model, q_values_func,memory,'train','ddqn')
#set learning rate to be 0.00025
agent.do_compile(optimizer=Adam(lr=0.00025), loss_func=mean_huber_loss)
agent.memoryD.clear()
#===================================================================



for iRepeat in range(num_reps):
    my_mission_record = malmoutils.get_default_recording_object(agent_host, "./Mission_{}".format(iRepeat + 1))
    #my_mission_record = MalmoPython.MissionRecordSpec('./' + "Mission_" + str(iRepeat) + ".tgz")
    #my_mission_record.recordRewards()
    #my_mission_record.recordMP4(24,400000)
    #my_mission_record.recordObservations()
    my_mission = MalmoPython.MissionSpec(GetMissionXML(mapblock, agent_host), validate)

    max_retries = 3
    for retry in range(max_retries):
        try:
            agent_host.startMission(my_mission, my_mission_record)
            break
        except RuntimeError as e:
            if retry == max_retries - 1:
                print("Error starting mission:",e)
                exit(1)
            else:
                time.sleep(2)

    print("Please wait while we are setting up the exciting mission", end = ' ')
    world_state = agent_host.getWorldState()
    while not world_state.has_mission_begun:
        print(".", end = "")
        time.sleep(0.1)
        world_state = agent_host.getWorldState()
        if len(world_state.errors):
            print()
            for error in world_state.errors:
                print("Error:", error.text)
                exit()
    print()

# ----------------------------------- Main Body ----------------------------------------
    agent_life = [20]
    villager_life = [20]
    zombie_life = [20]
    yaw_angle=[0,0]
    
    initial_frame = generate_random_start_position()
    while check_overlap_position(initial_frame) == True:
        initial_frame = generate_random_start_position()
    #=========core initialzation====================
    t=0
    total_reward=0
    #                     agent,        villeges,      zombies,   angle,size
    frame = get_frame(initial_frame[0],initial_frame[1],initial_frame[2],0)
    #===============================================
    
    waiting=0
    while True:
        #waiting+=1
        world_state = agent_host.peekWorldState()
        #print("waiting or stuck")
        if world_state.number_of_observations_since_last_state >0 or waiting>1000:
            break
    if waiting>1000:
        continue
    
    #================================================
    while True:
    
        
        #=========core part is here====================================
        t+=1
        agent.num_step+=1
        
        state=agent.process_new_frame(frame)
        # execute action 
        action=agent.select_action(state)
        act(action)

        # get enviroment
        #ob_copy=None
        if world_state.number_of_observations_since_last_state > 0:
            msg = world_state.observations[-1].text
            ob = json.loads(msg)
            ob_copy=ob.copy()
        else:
            ob=ob_copy
        angle=get_angle(ob,yaw_angle)
        # get array of entity coords            
        new_arr = get_arr(ob)
        if new_arr != None:
            #                     agent,      villeges,  zombies,   angle,size
            new_frame = get_frame(new_arr[0],new_arr[1],new_arr[2],angle)
        else:
            new_frame = get_frame([-99, -99], [-99, -99], [-99, -99],angle)
        # check terminal state
        reward = 0
        is_t, t_reward = get_info(ob)
        reward += t_reward
        if not world_state.is_mission_running:# or is_t:
            is_terminal = True
        else:
            is_terminal = False

        # check reward
        reward += set_reward(ob, agent_life, villager_life, zombie_life)
        total_reward += reward
        #print(action," Total score:", total_reward, "     Score:", reward)
        agent.memoryD.append(frame,action,reward,is_terminal)
        if agent.num_step>20000:
            if agent.num_step%19000==0:
                agent.update()
            
        if is_terminal:# or t> 10000:
            break
            
        frame=new_frame
        #===============================================================
            

            
        world_state = agent_host.getWorldState()
        
    
    print("=============== Mission", str(iRepeat + 1), "has stopped ==================")
    print("Agent step",agent.num_step,"Update times",agent.update_times)
    # time.sleep(0.5) 
