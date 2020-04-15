import keras.backend as K


def huber_loss(y_true, y_pred, max_grad=1.):
    diff = y_true - y_pred
    diff_squared = diff * diff
    max_grad_squared = max_grad * max_grad
    return max_grad_squared * (K.sqrt(1. + diff_squared/max_grad_squared) - 1.)


def mean_huber_loss(y_true, y_pred, max_grad=1.):
    loss = huber_loss(y_true, y_pred, max_grad=max_grad)
    return K.mean(loss, axis=None)



#原始================================================================
#地图坐标没改的情况下要用这个
def get_frame(agent,villeges,zombies,angle):
    frame=[[4.0]*5 for _ in range(5)]
    if agent[0]==-99:
        print('sb')
    if agent[0]!=-99:
        agent_x,agent_y=transform_state(agent[0],agent[1])
        frame[agent_x][agent_y]=angle/360
    if villeges[0]!=-99:
        villeges_x,villeges_y=transform_state(villeges[0],villeges[1])
        frame[villeges_x][villeges_y]=2.0
    if zombies[0]!=-99:
        zombies_x,zombies_y=transform_state(zombies[0],zombies[1])
        frame[zombies_x][zombies_y]=3.0
    
    return frame


def transform_state(x,y):
    if x>=3:
        x=2.5
    if y>=3:
        y=2.5
    if x<=-2:
        x=-1.5
    if y<=-2:
        y=-1.5
    return int(x+2),int(y+2)

'''


def get_frame(agent,villeges,zombies,angle,size):
    frame=[[4.0]*size for _ in range(size)]
    if agent[0]==-99:
        print('sb')
    if agent[0]!=-99:
        agent_x,agent_y=transform_state(agent[0],agent[1],size)
        frame[agent_x][agent_y]=angle/360
    if villeges[0]!=-99:
        villeges_x,villeges_y=transform_state(villeges[0],villeges[1],size)
        frame[villeges_x][villeges_y]=2.0
    if zombies[0]!=-99:
        zombies_x,zombies_y=transform_state(zombies[0],zombies[1],size)
        frame[zombies_x][zombies_y]=3.0
    
    return frame



def transform_state(x,y,size):
    if x>=size:
        x=size-0.5
    if y>=size:
        y=size-0.5
    if x<=0:
        x=0.5
    if y<=0:
        y=0.5
    return int(x),int(y)
'''