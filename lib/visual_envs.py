import numpy as np
import random
import itertools
import scipy.ndimage
import scipy.misc
import matplotlib.pyplot as plt
from numpy.random import rand

from scipy.ndimage import gaussian_filter

class gameOb():
    def __init__(self,coordinates,size,color,reward,name):
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.size = size
        self.color = color
        self.reward = reward
        self.name = name

#To implement:
# VisualIntEnv

class VisualIntEnv():
    def __init__(self, size, delay = 1, p1 = 0.1, p2 = 0.01, p3 = 0.01, int_p2 = 0.1, int_p3 = 0.1, max_steps = 20, obs_steps= 19, chain_prob = 0.5):
        self.sizeX = size
        self.sizeY = size
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.int_p2 = int_p2
        self.int_p3 = int_p3
        self.alpha = 0.5
        self.max_steps = max_steps
        self.obs_steps = obs_steps
        self.actions = 2
        self.N = 3 #Number of objects
        self.bg = np.zeros([size,size])
        self.chain_prob = chain_prob
        self.reset()
                
    def reset(self):
        #Choose the topology randomly with each reset
        self.is_chain = rand() > self.chain_prob
        self.timestep = 0
        self.state = np.zeros(self.N+1)
        rendered_state, rendered_state_big = self.renderEnv()
        self.xhistory = np.zeros((self.max_steps, self.N+1))
        return rendered_state, rendered_state_big

    def renderEnv(self):
        s = np.zeros([self.sizeY,self.sizeX])
        #For each object... find a location and render its state
        for idx in range(self.N):
            obj_x = int((idx * self.sizeX)/float(self.N)) 
            obj_y = obj_x
            s[obj_y, obj_x] = self.state[idx]

        #Plot response indicator
        s_i = np.zeros([self.sizeY,self.sizeX])
        obj_x = int(self.sizeX/float(self.N))
        obj_y = 0
        s_i[obj_y, obj_x] = self.state[-1]

        a = gaussian_filter(s, sigma = 1, mode = 'wrap')
        a = np.tile(a[:,:,None], (1,1,3))
        #Add response indicator pixels to red channel
        a[:,:,0] += s_i
        a_big = scipy.misc.imresize(a, [32,32,3], interp='nearest')
        return a, a_big

    def step(self,action):
        #Choose spontaneous activity
        y1 = rand() < self.p1
        y2 = rand() < self.p2
        y3 = rand() < self.p3

        #Introduce interventions that help distinguish the two causal graphs
        z2 = (action == 0)
        z3 = (action == 1)

        ##########
        #Dynamics#
        ##########

        #Choose if node A is active
        x1 = y1
        #Choose if node B is active
        x2 = y2 + (1-y2)*self.xhistory[max(0, self.timestep - 1), 0]
        if z2:          #Overwrite if intervening
            x2 = 1

        #Depending on topology, choose if node C is active
        if self.is_chain:
            x3 = y3 + (1-y3)*self.xhistory[max(0, self.timestep - 1), 1]
        else:
            x3 = y3 + (1-y3)*self.xhistory[max(0, self.timestep - 2), 0]
        if z3:          #Overwrite if intervening
            x3 = 1

        y1 = 1. if self.timestep >= self.obs_steps else 0.

        state = np.array([x1, x2, x3, y1])

        #Decay
        self.state = np.minimum(1, (1-self.alpha)*self.state + self.alpha*state)
        self.xhistory[self.timestep, :] = self.state
        self.timestep += 1
        if self.timestep >= self.max_steps:
            done = True
        else:
            done = False
        #If in the 'action phase', then the action is meant to indicate which topology it thinks is correct
        if self.timestep >= self.obs_steps:
            reward = float(action == self.is_chain)
        else:
            reward = 0.0
        #Render states to agent to see....
        rendered_state, rendered_state_big = self.renderEnv()
        return rendered_state, rendered_state_big, reward, done

class VisualObsIntEnv():
    def __init__(self, size, delay = 1, p1 = 0.1, p2 = 0.01, p3 = 0.01, int_p2 = 0.1, int_p3 = 0.1, max_steps = 20, obs_steps= 20, chain_prob = 0.5):
        self.sizeX = size
        self.sizeY = size
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.int_p2 = int_p2
        self.int_p3 = int_p3
        self.alpha = 0.5
        self.max_steps = max_steps
        self.obs_steps = obs_steps
        self.actions = 2
        self.N = 3 #Number of objects
        self.bg = np.zeros([size,size])
        self.chain_prob = chain_prob
        a,a_big = self.reset()
        plt.imshow(a_big,interpolation="nearest") 
                
    def reset(self):
        #Choose the topology randomly with each reset
        self.is_chain = rand() > self.chain_prob
        self.timestep = 0
        self.state = np.zeros(self.N+2)
        rendered_state, rendered_state_big = self.renderEnv()
        self.xhistory = np.zeros((self.max_steps, self.N+2))
        return rendered_state, rendered_state_big

    def renderEnv(self):
        s = np.zeros([self.sizeY,self.sizeX])
        #For each object... find a location and render its state
        for idx in range(self.N):
            obj_x = int((idx * self.sizeX)/float(self.N)) 
            obj_y = obj_x
            s[obj_y, obj_x] = self.state[idx]

        #For the intervention objects... plot those too!
        s_i = np.zeros([self.sizeY,self.sizeX])
        for idx in range(2):
            obj_x = int(((idx+1) * self.sizeX)/float(self.N))
            obj_y = int((idx * self.sizeX)/float(self.N))
            s_i[obj_y, obj_x] = self.state[self.N+idx]

        a = gaussian_filter(s, sigma = 1, mode = 'wrap')
        a = np.tile(a[:,:,None], (1,1,3))
        #Add intervention indicator pixels to red channel
        a[:,:,0] += s_i
        a_big = scipy.misc.imresize(a, [32,32,3], interp='nearest')
        return a, a_big

    def step(self,action):
        #Choose spontaneous activity
        y1 = rand() < self.p1
        y2 = rand() < self.p2
        y3 = rand() < self.p3

        #Introduce interventions that help distinguish the two causal graphs
        z2 = rand() < self.int_p2
        z3 = rand() < self.int_p3

        ##########
        #Dynamics#
        ##########

        #Choose if node A is active
        x1 = y1
        #Choose if node B is active
        x2 = y2 + (1-y2)*self.xhistory[max(0, self.timestep - 1), 0]
        if z2:          #Overwrite if intervening
            x2 = 1

        #Depending on topology, choose if node C is active
        if self.is_chain:
            x3 = y3 + (1-y3)*self.xhistory[max(0, self.timestep - 1), 1]
        else:
            x3 = y3 + (1-y3)*self.xhistory[max(0, self.timestep - 2), 0]
        if z3:          #Overwrite if intervening
            x3 = 1

        state = np.array([x1, x2, x3, z2, z3])

        #Decay
        self.state = np.minimum(1, (1-self.alpha)*self.state + self.alpha*state)
        self.xhistory[self.timestep, :] = self.state
        self.timestep += 1
        if self.timestep >= self.max_steps:
            done = True
        else:
            done = False
        #If in the 'action phase', then the action is meant to indicate which topology it thinks is correct
        if self.timestep >= self.obs_steps:
            reward = float(action == self.is_chain)
        else:
            reward = 0.0
        #Render states to agent to see....
        rendered_state, rendered_state_big = self.renderEnv()
        return rendered_state, rendered_state_big, reward, done

class VisualObsEnv():
    def __init__(self, size, delay = 1, p1 = 0.1, p2 = 0.01, p3 = 0.01, int_p2 = 0.1, int_p3 = 0.1, max_steps = 20, obs_steps= 20, chain_prob = 0.5):
        self.sizeX = size
        self.sizeY = size
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.int_p2 = int_p2
        self.int_p3 = int_p3
        self.alpha = 0.5
        self.max_steps = max_steps
        self.obs_steps = obs_steps
        self.actions = 2
        self.N = 3 #Number of objects
        self.bg = np.zeros([size,size])
        self.chain_prob = chain_prob
        a,a_big = self.reset()
        plt.imshow(a_big,interpolation="nearest") 
                
    def reset(self):
        #Choose the topology randomly with each reset
        self.is_chain = rand() > self.chain_prob
        self.timestep = 0
        self.state = np.zeros(self.N)
        rendered_state, rendered_state_big = self.renderEnv()
        self.xhistory = np.zeros((self.max_steps, self.N))
        return rendered_state, rendered_state_big

    def renderEnv(self):
        s = np.zeros([self.sizeY,self.sizeX])
        #For each object... find a location and render its state
        for idx in range(self.N):
            obj_x = int((idx * self.sizeX)/float(self.N)) 
            obj_y = obj_x
            s[obj_y, obj_x] = self.state[idx]
        a = gaussian_filter(s, sigma = 1, mode = 'wrap')
        a = np.tile(a[:,:,None], (1,1,3))
        a_big = scipy.misc.imresize(a, [32,32,3], interp='nearest')
        return a, a_big

    def step(self,action):
        #Choose spontaneous activity
        y1 = rand() < self.p1
        y2 = rand() < self.p2
        y3 = rand() < self.p3

        #Introduce interventions that help distinguish the two causal graphs
        z2 = rand() < self.int_p2
        z3 = rand() < self.int_p3

        ##########
        #Dynamics#
        ##########

        #Choose if node A is active
        x1 = y1
        #Choose if node B is active
        x2 = y2 + (1-y2)*self.xhistory[max(0, self.timestep - 1), 0]
        if z2:          #Overwrite if intervening
            x2 = 1

        #Depending on topology, choose if node C is active
        if self.is_chain:
            x3 = y3 + (1-y3)*self.xhistory[max(0, self.timestep - 1), 1]
        else:
            x3 = y3 + (1-y3)*self.xhistory[max(0, self.timestep - 2), 0]
        if z3:          #Overwrite if intervening
            x3 = 1

        state = np.array([x1, x2, x3])

        #Decay
        self.state = np.minimum(1, (1-self.alpha)*self.state + self.alpha*state)
        self.xhistory[self.timestep, :] = self.state
        self.timestep += 1
        if self.timestep >= self.max_steps:
            done = True
        else:
            done = False
        #If in the 'action phase', then the action is meant to indicate which topology it thinks is correct
        if self.timestep >= self.obs_steps:
            reward = float(action == self.is_chain)
        else:
            reward = 0.0
        #Render states to agent to see....
        rendered_state, rendered_state_big = self.renderEnv()
        return rendered_state, rendered_state_big, reward, done

#Here not enough information is provided to solve the problem
#Three objects. 
class VisualConfoundedEnv():
    def __init__(self, size, delay = 1, p1 = 0.1, p2 = 0.01, p3 = 0.01, max_steps = 20, obs_steps= 20, chain_prob = 0.5):
        self.sizeX = size
        self.sizeY = size
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.alpha = 0.5
        self.max_steps = max_steps
        self.obs_steps = obs_steps
        self.actions = 2
        self.N = 3 #Number of objects
        self.bg = np.zeros([size,size])
        self.chain_prob = chain_prob
        a,a_big = self.reset()
        plt.imshow(a_big,interpolation="nearest") 
                
    def reset(self):
        #Choose the topology randomly with each reset
        self.is_chain = rand() > self.chain_prob
        self.timestep = 0
        self.state = np.zeros(self.N)
        rendered_state, rendered_state_big = self.renderEnv()
        self.xhistory = np.zeros((self.max_steps, self.N))
        return rendered_state, rendered_state_big

    def renderEnv(self):
        s = np.zeros([self.sizeY,self.sizeX])
        #For each object... find a location and render its state
        for idx in range(self.N):
            obj_x = int((idx * self.sizeX)/float(self.N)) 
            obj_y = obj_x
            s[obj_y, obj_x] = self.state[idx]
        a = gaussian_filter(s, sigma = 1, mode = 'wrap')
        a = np.tile(a[:,:,None], (1,1,3))
        a_big = scipy.misc.imresize(a, [32,32,3], interp='nearest')
        return a, a_big

    def step(self,action):
        #Choose spontaneous activity
        y1 = rand() < self.p1
        y2 = rand() < self.p2
        y3 = rand() < self.p3

        ##########
        #Dynamics#
        ##########

        #Choose if node A is active
        x1 = y1
        #Choose if node B is active
        x2 = y2 + (1-y2)*self.xhistory[max(0, self.timestep - 1), 0]
        #Depending on topology, choose if node C is active
        if self.is_chain:
            x3 = y3 + (1-y3)*self.xhistory[max(0, self.timestep - 1), 1]
        else:
            x3 = y3 + (1-y3)*self.xhistory[max(0, self.timestep - 2), 0]
        state = np.array([x1, x2, x3])

        #Decay
        self.state = np.minimum(1, (1-self.alpha)*self.state + self.alpha*state)
        self.xhistory[self.timestep, :] = self.state
        self.timestep += 1
        if self.timestep >= self.max_steps:
            done = True
        else:
            done = False
        #If in the 'action phase', then the action is meant to indicate which topology it thinks is correct
        if self.timestep >= self.obs_steps:
            reward = float(action == self.is_chain)
        else:
            reward = 0.0
        #Render states to agent to see....
        rendered_state, rendered_state_big = self.renderEnv()
        return rendered_state, rendered_state_big, reward, done

class gameEnv():
    def __init__(self,partial,size,goal_color):
        self.sizeX = size
        self.sizeY = size
        self.actions = 4
        self.objects = []
        self.partial = partial
        self.bg = np.zeros([size,size])
        a,a_big = self.reset(goal_color)
        plt.imshow(a_big,interpolation="nearest")
        
    def getFeatures(self):
        return np.array([self.objects[0].x,self.objects[0].y]) / float(self.sizeX)
        
    def reset(self,goal_color):
        self.objects = []
        self.goal_color = goal_color
        self.other_color = [1 - a for a in self.goal_color]
        self.orientation = 0
        self.hero = gameOb(self.newPosition(0),1,[0,0,1],None,'hero')
        self.objects.append(self.hero)
        for i in range(self.sizeX-1):
            bug = gameOb(self.newPosition(0),1,self.goal_color,1,'goal')
            self.objects.append(bug)
        for i in range(self.sizeX-1):
            hole = gameOb(self.newPosition(0),1,self.other_color,0,'fire')
            self.objects.append(hole)
        state,s_big = self.renderEnv()
        self.state = state
        return state,s_big

    def moveChar(self,action):
        # 0 - up, 1 - down, 2 - left, 3 - right, 4 - 90 counter-clockwise, 5 - 90 clockwise
        hero = self.objects[0]
        blockPositions = [[-1,-1]]
        for ob in self.objects:
            if ob.name == 'block': blockPositions.append([ob.x,ob.y])
        blockPositions = np.array(blockPositions)
        heroX = hero.x
        heroY = hero.y
        penalize = 0.
        if action < 4 :
            if self.orientation == 0:
               direction = action            
            if self.orientation == 1:
               if action == 0: direction = 1
               elif action == 1: direction = 0
               elif action == 2: direction = 3
               elif action == 3: direction = 2
            if self.orientation == 2:
               if action == 0: direction = 3
               elif action == 1: direction = 2
               elif action == 2: direction = 0
               elif action == 3: direction = 1
            if self.orientation == 3:
               if action == 0: direction = 2
               elif action == 1: direction = 3
               elif action == 2: direction = 1
               elif action == 3: direction = 0
        
            if direction == 0 and hero.y >= 1 and [hero.x,hero.y - 1] not in blockPositions.tolist():
                hero.y -= 1
            if direction == 1 and hero.y <= self.sizeY-2 and [hero.x,hero.y + 1] not in blockPositions.tolist():
                hero.y += 1
            if direction == 2 and hero.x >= 1 and [hero.x - 1,hero.y] not in blockPositions.tolist():
                hero.x -= 1
            if direction == 3 and hero.x <= self.sizeX-2 and [hero.x + 1,hero.y] not in blockPositions.tolist():
                hero.x += 1     
        if hero.x == heroX and hero.y == heroY:
            penalize = 0.0
        self.objects[0] = hero
        return penalize
    
    def newPosition(self,sparcity):
        iterables = [ range(self.sizeX), range(self.sizeY)]
        points = []
        for t in itertools.product(*iterables):
            points.append(t)
        for objectA in self.objects:
            if (objectA.x,objectA.y) in points: points.remove((objectA.x,objectA.y))
        location = np.random.choice(range(len(points)),replace=False)
        return points[location]

    def checkGoal(self):
        hero = self.objects[0]
        others = self.objects[1:]
        ended = False
        for other in others:
            if hero.x == other.x and hero.y == other.y and hero != other:
                self.objects.remove(other)
                if other.reward == 1:
                    self.objects.append(gameOb(self.newPosition(0),1,self.goal_color,1,'goal'))
                    return other.reward,False
                else: 
                    self.objects.append(gameOb(self.newPosition(0),1,self.other_color,0,'fire'))
                    return other.reward,False
        if ended == False:
            return 0.0,False

    def renderEnv(self):
        if self.partial == True:
            padding = 2
            a = np.ones([self.sizeY+(padding*2),self.sizeX+(padding*2),3])
            a[padding:-padding,padding:-padding,:] = 0
            a[padding:-padding,padding:-padding,:] += np.dstack([self.bg,self.bg,self.bg])
        else:
            a = np.zeros([self.sizeY,self.sizeX,3])
            padding = 0
            a += np.dstack([self.bg,self.bg,self.bg])
        hero = self.objects[0]
        for item in self.objects:
            a[item.y+padding:item.y+item.size+padding,item.x+padding:item.x+item.size+padding,:] = item.color
            #if item.name == 'hero':
            #    hero = item
        if self.partial == True:
            a = a[(hero.y):(hero.y+(padding*2)+hero.size),(hero.x):(hero.x+(padding*2)+hero.size),:]
        a_big = scipy.misc.imresize(a,[32,32,3],interp='nearest')
        return a,a_big

    def step(self,action):
        penalty = self.moveChar(action)
        reward,done = self.checkGoal()
        state,s_big = self.renderEnv()
        if reward == None:
            print(done)
            print(reward)
            print(penalty)
            return state,(reward+penalty),done
        else:
            goal = None
            for ob in self.objects:
                if ob.name == 'goal':
                    goal = ob
            return state,s_big,(reward+penalty),done,[self.objects[0].y,self.objects[0].x],[goal.y,goal.x]
