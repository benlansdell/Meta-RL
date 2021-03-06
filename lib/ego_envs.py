import numpy as np
import random
import itertools
import scipy.ndimage
import scipy.misc
import matplotlib.pyplot as plt
from numpy.random import rand

from scipy.ndimage import gaussian_filter

from PIL import Image
import skimage.transform

from lib.helper import set_image_ego, make_gif

#trial_env(PushButtonsCardinalEnv, './tests/test_cardinal_env.gif')
#trial_env(PushManyButtonsEnv, './tests/test_manybuttons_env.gif')
#trial_env(PushButtonsEnv, './tests/test_randomsize_buttons_env.gif', randomize_size = True)
#trial_env(PushButtonsEnv, './tests/test_randomsize_test_buttons_env.gif', randomize_size = True, test = True)

def trial_env(Env, fn_out, n_trials = 5, randomize_size = False, test = False, env_dim = 5):
    """Run the given environment for a given number of trials.
    """
    #Create env
    frames = []
    env = Env(size = env_dim, randomize_size = randomize_size)
    time_per_step = 0.25
    for _ in range(n_trials):
        d = False
        episode_reward = 0
        s,s_big = env.reset(test = test)
        t = 0
        while not d:
            #Take an action according to random policy
            a = np.random.choice(env.actions)
            _,s1_big,r,d = env.step(a)
            episode_reward += r
            t += 1
            frames.append(set_image_ego(s1_big,episode_reward,t))         

    images = np.array(frames)
    make_gif(images, fn_out, duration=len(images)*time_per_step,true_image=True)

class StepOnLightsEnv():
    """Agent can move around, and step on one of the lights to block it ('intervening' on it). 

    Reward is given when, after receiving a go cue, it steps on the right box.
    """
    def __init__(self, size = 8, delay = 1, p1 = 0.5, p2 = 0.3, p3 = 0.3, int_p2 = 0.1, int_p3 = 0.1, max_steps = 20, obs_steps= 19, chain_prob = 0.5):
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
        self.actions = 5
        self.N = 3 #Number of objects
        self.bg = np.zeros([size,size])
        self.chain_prob = chain_prob
        self.reset()
                
    def reset(self):
        #Choose the topology randomly with each reset
        self.is_chain = rand() > self.chain_prob
        self.timestep = 0
        self.agent_pos = np.floor(rand(2)*self.sizeX).astype(int)
        self.state = np.zeros(self.N+1)
        rendered_state, rendered_state_big = self.renderEnv()
        self.xhistory = np.zeros((self.max_steps, self.N+1))
        return rendered_state, rendered_state_big

    def renderEnv(self, brightness = 0.2):
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
        s_i[obj_y, obj_x] = self.state[-1]*brightness

        a = gaussian_filter(s, sigma = 1, mode = 'wrap')
        a = np.tile(a[:,:,None], (1,1,3))

        #Draw the agent on green channel
        agent_y = self.agent_pos[0]
        agent_x = self.agent_pos[1]
        a_i = np.zeros([self.sizeY,self.sizeX])
        a_i[agent_y, agent_x] = brightness

        #Add response indicator pixels to red channel
        a[:,:,0] += s_i
        a[:,:,1] += a_i
        #a_big = scipy.misc.imresize(a, [32,32,3], interp='nearest')
        a_big = skimage.transform.resize(a, [32, 32, 3], order = 0)*255.0
        return a, a_big

    def step(self,action):

        #Here the state is given... 
        #Thus the action interacts with the current state
        #The rules are as follows:
        # - Move the agent around, can move anywhere, just not ouside the environment
        if action == 0:
            #Move up
            self.agent_pos[0] = max(0, self.agent_pos[0]-1)
        elif action == 1:
            #Move down
            self.agent_pos[0] = min(self.sizeX-1, self.agent_pos[0]+1)
        elif action == 2:
            #Move left
            self.agent_pos[1] = max(0, self.agent_pos[0]-1)
        elif action == 3:
            #Move right
            self.agent_pos[1] = min(self.sizeX-1, self.agent_pos[0]+1)
        #Action 4 here does nothing (push button)

        #Choose spontaneous activity
        y1 = rand() < self.p1
        y2 = rand() < self.p2
        y3 = rand() < self.p3

        #Introduce interventions that help distinguish the two causal graphs
        #This now depends on the location of the agent
        on_object = np.zeros(self.N)
        for idx in range(self.N):
            obj_x = int((idx * self.sizeX)/float(self.N)) 
            obj_y = obj_x
            if (obj_y, obj_x) == (self.agent_pos[0], self.agent_pos[1]):
                on_object[idx] = 1

        ##########
        #Dynamics#
        ##########

        #Choose if node A is active
        x1 = y1
        #Choose if node B is active
        x2 = y2 + (1-y2)*self.xhistory[max(0, self.timestep - 1), 0]
        if on_object[1]:          #Overwrite if intervening. Here the interventions block the light...
            x2 = 0

        #Depending on topology, choose if node C is active
        if self.is_chain:
            x3 = y3 + (1-y3)*self.xhistory[max(0, self.timestep - 1), 1]
        else:
            x3 = y3 + (1-y3)*self.xhistory[max(0, self.timestep - 2), 0]
        if on_object[2]:          #Overwrite if intervening. Here the interventions block the light
            x3 = 0

        y1 = 1. if self.timestep >= self.obs_steps else 0.

        state = np.array([x1, x2, x3, y1])

        #Decay activity
        self.state = np.minimum(1, (1-self.alpha)*self.state + self.alpha*state)
        self.xhistory[self.timestep, :] = self.state
        self.timestep += 1
        if self.timestep >= self.max_steps:
            done = True
        else:
            done = False

        #If in the 'reward phase', then the action is meant to indicate which topology it thinks is correct
        #Here this is rewarded if the position is the 
        if self.timestep >= self.obs_steps:
            reward = 0.0
            #If on object 2:
            obj_loc = int((1 * self.sizeX)/float(self.N)) 
            if (obj_loc, obj_loc) == (self.agent_pos[0], self.agent_pos[1]):
                if self.is_chain:
                    reward = 1.
                else:
                    reward = -1.
            #If on object 3:
            obj_loc = int((1 * self.sizeX)/float(self.N)) 
            if (obj_loc, obj_loc) == (self.agent_pos[0], self.agent_pos[1]):
                if self.is_chain:
                    reward = -1.
                else:
                    reward = 1.
        else:
            reward = 0.0
        #Render states to agent to see....
        rendered_state, rendered_state_big = self.renderEnv()
        return rendered_state, rendered_state_big, reward, done

class PushButtonsEnv():
    """Agent can observe a ball bouncing around. The ball has some slight randomness in the angle it bounces off the walls.
    The ball can interact with buttons that open a door. When the go cue is given, the agent must figure out which button is
    the one that opens the door and quickly move to it before get the reward in time. 

    This is quite challenging to just randomly solve...

    Reward is given when, after receiving a go cue, it reaches the exit square
    """
    def __init__(self, size = 5, delay = 1, obs_steps = 30, max_steps = 40, randomize_size = False, max_size = 7):
        self.alpha = 0.5
        self.max_steps = max_steps
        self.obs_steps = obs_steps
        self.actions = 4
        self.n_buttons = 3
        self.open_count = 6 #Number of time steps the door is open after button pushed
        self.randomize_size = randomize_size
        self.size = size
        if self.randomize_size: self.max_size = max_size
        else: self.max_size = size
        self._make_geometry()
        self.reset()

    def _make_geometry(self, test = False):

        if test: size = self.max_size+1
        elif self.randomize_size: size = np.random.randint(self.size, self.max_size+1)
        else: size = self.size
        self.sizeX = size
        self.sizeY = size
        self.bg = np.zeros([size,size])

        #self.b1_pos = (0,2)
        #self.b2_pos = (2,0)
        #self.b3_pos = (4,2)
        #self.door_pos = (2,4)
        self.b1_pos = (0,2-self.size+size)
        self.b2_pos = (2-self.size+size,0)
        self.b3_pos = (size-1,2-self.size+size)
        self.door_pos = (2-self.size+size,size-1)

        self.render_button_pos = np.zeros((self.n_buttons, 2), dtype = int)

        #self.render_button_pos[0,:] = [0,3]
        #self.render_button_pos[1,:] = [3,0]
        #self.render_button_pos[2,:] = [6,3]
        #self.render_door_pos = (3,6)
        self.render_button_pos[0,:] = [0,self.b1_pos[1]+1]
        self.render_button_pos[1,:] = [self.b2_pos[0]+1,0]
        self.render_button_pos[2,:] = [self.b3_pos[0]+2,self.b3_pos[1]+1]
        self.render_door_pos = (self.door_pos[0]+1,self.door_pos[1]+2)

        self.button_positions = [self.b1_pos, self.b2_pos, self.b3_pos]

    def reset(self, test = False):            

        self._make_geometry(test)
        #Choose the topology randomly with each reset
        self.exit_button = self.chooseNewTarget()
        self.timestep = 0
        #Choose the ball's position and target randomly
        self.door_open = 0
        self.button_on = [0,0,0]
        self.ball_pos = np.random.randint(self.sizeX, size = 2)
        self.ball_target = self.chooseNewTarget()
        self.agent_pos = np.random.randint(self.sizeX, size = 2)
        self.state = np.zeros(self.n_buttons+1)
        rendered_state, rendered_state_big = self.renderEnv()
        self.xhistory = np.zeros((self.max_steps, self.n_buttons+1))
        return rendered_state, rendered_state_big

    def chooseNewTarget(self):
        return np.random.randint(self.n_buttons)

    def vel(self, x):
        if x < 0:
            return -1
        elif x > 0:
            return 1
        else:
            return 0

    def renderEnv(self, brightness = 0.2):

        s = np.zeros([self.max_size+1+2,self.max_size+1+2])
        #s = np.zeros([self.sizeY+2,self.sizeX+2])
        #Buttons
        for idx in range(self.n_buttons):
            s[(int(self.render_button_pos[idx,0]),int(self.render_button_pos[idx,1]))] = 0.5
        #Door
        s[self.render_door_pos] = 0.5

        #If in obs mode:
        if self.timestep <= self.obs_steps:
            #Ball position
            s[1+self.ball_pos[0], 1+self.ball_pos[1]] = 1
        #If in action mode:
        else:
            #Agent position, and light indicating if exit is open or not
            s[1+self.agent_pos[0], 1+self.agent_pos[1]] = brightness

        a = np.tile(s[:,:,None], (1,1,3))
        #Draw indicators on different channels
        if self.door_open:
            a[self.render_door_pos[0],self.render_door_pos[1],0] = 1
        for idx in range(self.n_buttons):
            if self.button_on[idx]:
                a[self.render_button_pos[idx,0],self.render_button_pos[idx,1],2] = 1

        if self.timestep > self.obs_steps:
            a[0,0,1] = 1

        #a_big = scipy.misc.imresize(a, [32,32,3], interp='nearest')
        #a_big = Image.fromarray(a).resize(size = [32, 32])
        a_big = skimage.transform.resize(a, [32, 32, 3], order = 0)*255.0
        return a, a_big

    def step(self,action):
        #The environment dynamics are all here...
        reward = 0.0

        #Reset the buttons and coutndown door timer
        for idx in range(self.n_buttons):
            if self.button_on[idx]:
                self.button_on[idx] = 0
        if self.door_open > 0:
            self.door_open -= 1

        #If in obs mode... the ball moves around
        #######
        if self.timestep <= self.obs_steps:
            #Move it towards target
            #Calculate difference vector
            dy = self.vel(self.ball_pos[0] - self.button_positions[self.ball_target][0])
            dx = self.vel(self.ball_pos[1] - self.button_positions[self.ball_target][1])
            self.ball_pos[0] -= dy
            self.ball_pos[1] -= dx

            #If ball is at a target, choose a new target randomly
            for idx, pos in enumerate(self.button_positions):
                if self.ball_pos[0] == pos[0] and self.ball_pos[1] == pos[1]:
                    self.button_on[idx] = 1

                    if self.exit_button == idx:
                        self.door_open = self.open_count
                        #To make things a bit easier for this environment, these transitions are also rewarded...
                        reward = 1.0
                    self.ball_target = self.chooseNewTarget()

        #If in action mode... the agent can move around
        #######
        else:
            if action == 0:
                #Move up
                self.agent_pos[0] = max(0, self.agent_pos[0]-1)
            elif action == 1:
                #Move down
                self.agent_pos[0] = min(self.sizeX-1, self.agent_pos[0]+1)
            elif action == 2:
                #Move left
                self.agent_pos[1] = max(0, self.agent_pos[1]-1)
            elif action == 3:
                #Move right
                self.agent_pos[1] = min(self.sizeX-1, self.agent_pos[1]+1)

            #If agent is on a button then set indicator to 1
            #If button is a target, then open door and set timer
            for idx, pos in enumerate(self.button_positions):
                if self.agent_pos[0] == pos[0] and self.agent_pos[1] == pos[1]:
                    self.button_on[idx] = 1
                    if self.exit_button == idx:
                        self.door_open = self.open_count

            #If agent is at open door, then give reward
            if self.agent_pos[0] == self.door_pos[0] and self.agent_pos[1] == self.door_pos[1]:
                if self.door_open:
                    reward = 10.0

        #Update environment state...
        state = np.hstack((self.button_on, self.door_open))
        if self.timestep < self.max_steps:
            self.xhistory[self.timestep,:] = state

        self.timestep += 1
        if self.timestep >= self.max_steps:
            done = True
        else:
            done = False

        #Render states to agent to see....
        rendered_state, rendered_state_big = self.renderEnv()
        return rendered_state, rendered_state_big, reward, done

class PushManyButtonsEnv(PushButtonsEnv):
    """Agent can observe a ball bouncing around. The ball has some slight randomness in the angle it bounces off the walls.
    The ball can interact with buttons that open a door. When the go cue is given, the agent must figure out which button is
    the one that opens the door and quickly move to it before get the reward in time. 

    This is quite challenging to just randomly solve...

    Reward is given when, after receiving a go cue, it reaches the exit square
    """

    def __init__(self, size = 5, delay = 1, obs_steps = 25, max_steps = 40, randomize_size = False, max_size = 7):
        self.alpha = 0.5
        self.max_steps = max_steps
        self.obs_steps = obs_steps
        self.actions = 4
        self.n_buttons = 6
        self.open_count = 6 #Number of time steps the door is open after button pushed
        self.randomize_size = randomize_size
        self.size = size
        if self.randomize_size: self.max_size = max_size
        else: self.max_size = size
        self._make_geometry()
        self.reset()

    def _make_geometry(self):

        self.sizeX = self.size
        self.sizeY = self.size
        self.bg = np.zeros([self.size,self.size])

        self.b1_pos = (0,1)
        self.b2_pos = (1,0)
        self.b3_pos = (4,1)
        self.b4_pos = (0,3)
        self.b5_pos = (3,0)
        self.b6_pos = (4,3)

        self.door_pos = (2,4)

        self.render_button_pos = np.zeros((self.n_buttons, 2), dtype = int)

        self.render_button_pos[0,:] = [0,2]
        self.render_button_pos[1,:] = [2,0]
        self.render_button_pos[2,:] = [6,2]
        self.render_button_pos[3,:] = [0,4]
        self.render_button_pos[4,:] = [4,0]
        self.render_button_pos[5,:] = [6,4]

        self.render_door_pos = (3,6)
        
        self.button_positions = [self.b1_pos, self.b2_pos, self.b3_pos,\
                                self.b4_pos, self.b5_pos, self.b6_pos]

        self.test_button = 2

    def reset(self, test = False):            

        #Choose the topology randomly with each reset
        self.exit_button = self.chooseNewTarget(test = test)        
        self.timestep = 0
        #Choose the ball's position and target randomly
        self.door_open = 0
        self.button_on = [0]*self.n_buttons
        self.ball_pos = np.random.randint(self.sizeX, size = 2)
        self.ball_target = self.chooseNewTarget()
        self.agent_pos = np.random.randint(self.sizeX, size = 2)
        self.state = np.zeros(self.n_buttons+1)
        rendered_state, rendered_state_big = self.renderEnv()
        self.xhistory = np.zeros((self.max_steps, self.n_buttons+1))
        return rendered_state, rendered_state_big

    def chooseNewTarget(self, test = None):
        if test is None:
            return np.random.randint(self.n_buttons)
        elif test is True:
            return self.test_button
        else:
            b = np.random.randint(self.n_buttons)
            while b == self.test_button:
                b = np.random.randint(self.n_buttons)
            return b

class PushButtonsCardinalEnv(PushButtonsEnv):
    """Agent can observe a ball bouncing around. The ball has some slight randomness in the angle it bounces off the walls.
    The ball can interact with buttons that open a door. When the go cue is given, the agent must figure out which button is
    the one that opens the door and quickly move to it before get the reward in time. 

    This is quite challenging to just randomly solve...

    Reward is given when, after receiving a go cue, it reaches the exit square.

    In this environment, the ball can only move in 4 cardinal directions, while the agent can move
    in all 8 directions (like a king in chess). This tests to see if the agent is copying the white
    box (which only moves cardinally), or takes advantage of its extended action space.
    """

    def __init__(self, size = 5, delay = 1, obs_steps = 30, max_steps = 40, randomize_size = None):
        super(PushButtonsCardinalEnv, self).__init__(size, delay, obs_steps, max_steps)
        self.actions = 8

    def step(self,action):
        #The environment dynamics are all here...
        reward = 0.0

        #Reset the buttons and coutndown door timer
        for idx in range(self.n_buttons):
            if self.button_on[idx]:
                self.button_on[idx] = 0
        if self.door_open > 0:
            self.door_open -= 1

        #If in obs mode... the ball moves around
        #######
        if self.timestep <= self.obs_steps:
            #Move it towards target
            #Calculate difference vector

            #Only move one of dx and dy...
            diff_y = self.ball_pos[0] - self.button_positions[self.ball_target][0]
            diff_x = self.ball_pos[1] - self.button_positions[self.ball_target][1]
            dy = self.vel(diff_y)
            dx = self.vel(diff_x)
            move_x = abs(diff_x) > abs(diff_y)
            if move_x: self.ball_pos[1] -= dx
            else: self.ball_pos[0] -= dy

            #If ball is at a target, choose a new target randomly
            for idx, pos in enumerate(self.button_positions):
                if self.ball_pos[0] == pos[0] and self.ball_pos[1] == pos[1]:
                    self.button_on[idx] = 1

                    if self.exit_button == idx:
                        self.door_open = self.open_count
                        #To make things a bit easier for this environment, these transitions are also rewarded...
                        reward = 1.0
                    self.ball_target = self.chooseNewTarget()

        #If in action mode... the agent can move around
        #######
        else:
            if action == 0:
                #Move up
                self.agent_pos[0] = max(0, self.agent_pos[0]-1)
            elif action == 1:
                #Move down
                self.agent_pos[0] = min(self.sizeX-1, self.agent_pos[0]+1)
            elif action == 2:
                #Move left
                self.agent_pos[1] = max(0, self.agent_pos[1]-1)
            elif action == 3:
                #Move right
                self.agent_pos[1] = min(self.sizeX-1, self.agent_pos[1]+1)
            elif action == 4:
                #Move up-left
                self.agent_pos[0] = max(0, self.agent_pos[0]-1)
                self.agent_pos[1] = max(0, self.agent_pos[1]-1)
            elif action == 5:
                #Move down-left
                self.agent_pos[0] = min(self.sizeX-1, self.agent_pos[0]+1)
                self.agent_pos[1] = max(0, self.agent_pos[1]-1)
            elif action == 6:
                #Move up-right
                self.agent_pos[0] = max(0, self.agent_pos[0]-1)
                self.agent_pos[1] = min(self.sizeX-1, self.agent_pos[1]+1)
            elif action == 7:
                #Move down-right
                self.agent_pos[0] = min(self.sizeX-1, self.agent_pos[0]+1)
                self.agent_pos[1] = min(self.sizeX-1, self.agent_pos[1]+1)
            #If agent is on a button then set indicator to 1
            #If button is a target, then open door and set timer
            for idx, pos in enumerate(self.button_positions):
                if self.agent_pos[0] == pos[0] and self.agent_pos[1] == pos[1]:
                    self.button_on[idx] = 1
                    if self.exit_button == idx:
                        self.door_open = self.open_count

            #If agent is at open door, then give reward
            if self.agent_pos[0] == self.door_pos[0] and self.agent_pos[1] == self.door_pos[1]:
                if self.door_open:
                    reward = 10.0

        #Update environment state...
        state = np.hstack((self.button_on, self.door_open))
        if self.timestep < self.max_steps:
            self.xhistory[self.timestep,:] = state

        self.timestep += 1
        if self.timestep >= self.max_steps:
            done = True
        else:
            done = False

        #Render states to agent to see....
        rendered_state, rendered_state_big = self.renderEnv()
        return rendered_state, rendered_state_big, reward, done

