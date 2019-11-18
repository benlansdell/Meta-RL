import numpy as np
from numpy.random import rand

class ConfoundedEnv(object):
    def __init__(self, delay = 1, p1 = 0.1, p2 = 0.01, p3 = 0.01, max_steps = 20, obs_steps= 20, chain_prob = 0.5):
        self.delay = delay
        self.N = 3
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.max_steps = max_steps
        self.obs_steps = obs_steps
        self.chain_prob = chain_prob
        self.reset()
        
    def reset(self):
        self.timestep = 0
        
        #Choose the environment: chain or fork
        #self.is_chain = 1      #Sanity check to test learning with deterministic
        self.is_chain = rand() > self.chain_prob
        
        self.xhistory = np.zeros((self.max_steps, self.N))
        return [0]*self.N

    def step(self,action):
        #Choose spontaneous activity
        y1 = rand() < self.p1
        y2 = rand() < self.p2
        y3 = rand() < self.p3
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
        self.xhistory[self.timestep, :] = state
        self.timestep += 1
        if self.timestep >= self.max_steps:
            done = True
        else:
            done = False
        #If in the 'action phase', then the action is meant to indicate which topology is thinks is correct
        if self.timestep >= self.obs_steps:
            reward = float(action == self.is_chain)
        else:
            reward = 0.0
        return reward, done, self.timestep, state

class ObsIntEnv(ConfoundedEnv):
    def __init__(self, delay = 1, p1 = 0.01, p2 = 0.01, p3 = 0.01, int_p2 = 0.2, int_p3 = 0.0,\
                                             max_steps = 10, obs_steps= 10, chain_prob = 0.5):
        self.delay = delay
        self.N = 5
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.int_p2 = int_p2
        self.int_p3 = int_p3
        self.max_steps = max_steps
        self.obs_steps = obs_steps
        self.chain_prob = chain_prob
        self.reset()
        
    def step(self,action):
        #Choose spontaneous activity
        y1 = rand() < self.p1
        y2 = rand() < self.p2
        y3 = rand() < self.p3

        #Introduce interventions that help distinguish the two causal graphs
        z2 = rand() < self.int_p2
        z3 = rand() < self.int_p3

        #Choose if node A is active
        x1 = y1
        #Choose if node B is active
        x2 = y2 + (1-y2)*self.xhistory[max(0, self.timestep - self.delay), 0]
        if z2:          #Overwrite if intervening
            x2 = 1
        #Depending on topology, choose if node C is active
        if self.is_chain:
            x3 = y3 + (1-y3)*self.xhistory[max(0, self.timestep - self.delay), 1]
        else:
            x3 = y3 + (1-y3)*self.xhistory[max(0, self.timestep - 2*self.delay), 0]
        if z3:          #Overwrite if intervening
            x3 = 1

        state = np.array([x1, x2, x3, z2, z3])
        self.xhistory[self.timestep, :] = state
        self.timestep += 1
        if self.timestep >= self.max_steps:
            done = True
        else:
            done = False
        #If in the 'action phase', then the action is meant to indicate which topology is thinks is correct
        if self.timestep >= self.obs_steps:
            reward = float((action == self.is_chain))
        else:
            reward = 0.
        return reward, done, self.timestep, state

class IntEnv(ConfoundedEnv):
    def __init__(self, delay = 1, p1 = 0.01, p2 = 0.01, p3 = 0.01, int_p2 = 0.2, int_p3 = 0.0,\
                                             max_steps = 10, obs_steps= 9, chain_prob = 0.5):
        self.delay = delay
        self.N = 4
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.int_p2 = int_p2
        self.int_p3 = int_p3
        self.max_steps = max_steps
        self.obs_steps = obs_steps
        self.chain_prob = chain_prob
        self.reset()
        
    def step(self,action):
        #Choose spontaneous activity
        y1 = rand() < self.p1
        y2 = rand() < self.p2
        y3 = rand() < self.p3

        #Introduce interventions that help distinguish the two causal graphs
        z2 = (action == 0)
        z3 = (action == 1)

        #Choose if node A is active
        x1 = y1
        #Choose if node B is active
        x2 = y2 + (1-y2)*self.xhistory[max(0, self.timestep - self.delay), 0]
        if z2:          #Overwrite if intervening
            x2 = 1
        #Depending on topology, choose if node C is active
        if self.is_chain:
            x3 = y3 + (1-y3)*self.xhistory[max(0, self.timestep - self.delay), 1]
        else:
            x3 = y3 + (1-y3)*self.xhistory[max(0, self.timestep - 2*self.delay), 0]
        if z3:          #Overwrite if intervening
            x3 = 1

        y1 = 1. if self.timestep >= self.obs_steps else 0.

        state = np.array([x1, x2, x3, y1])
        self.xhistory[self.timestep, :] = state
        self.timestep += 1
        if self.timestep >= self.max_steps:
            done = True
        else:
            done = False
        #If in the 'answer phase', then the action is meant to indicate which topology is thinks is correct,
        #otherwise the actions are just interventions on the variables, and no reward is given
        if self.timestep >= self.max_steps:
            reward = float((action == self.is_chain))
        else:
            reward = 0.
        return reward, done, self.timestep, state

class ObsEnv(ConfoundedEnv):
    def __init__(self, delay = 1, p1 = 0.1, p2 = 0.01, p3 = 0.01, int_p2 = 0.1, int_p3 = 0.1,\
                                             max_steps = 20, obs_steps= 20, chain_prob = 0.5):
        self.delay = delay
        self.N = 3
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.int_p2 = int_p2
        self.int_p3 = int_p3
        self.max_steps = max_steps
        self.obs_steps = obs_steps
        self.chain_prob = chain_prob
        self.reset()
        
    def step(self,action):
        #Choose spontaneous activity
        y1 = rand() < self.p1
        y2 = rand() < self.p2
        y3 = rand() < self.p3

        #Introduce interventions that help distinguish the two causal graphs
        z2 = rand() < self.int_p2
        z3 = rand() < self.int_p3

        #Choose if node A is active
        x1 = y1
        #Choose if node B is active
        x2 = y2 + (1-y2)*self.xhistory[max(0, self.timestep - self.delay), 0]
        if z2:          #Overwrite if intervening
            x2 = 1
        #Depending on topology, choose if node C is active
        if self.is_chain:
            x3 = y3 + (1-y3)*self.xhistory[max(0, self.timestep - self.delay), 1]
        else:
            x3 = y3 + (1-y3)*self.xhistory[max(0, self.timestep - 2*self.delay), 0]
        if z3:          #Overwrite if intervening
            x3 = 1

        state = np.array([x1, x2, x3])
        self.xhistory[self.timestep, :] = state
        self.timestep += 1
        if self.timestep >= self.max_steps:
            done = True
        else:
            done = False
        #If in the 'action phase', then the action is meant to indicate which topology is thinks is correct
        if self.timestep >= self.obs_steps:
            reward = float((action == self.is_chain))
        else:
            reward = 0.
        return reward, done, self.timestep, state
