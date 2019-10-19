import threading
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.signal
from PIL import Image
from PIL import ImageDraw 
from PIL import ImageFont
from helper import *

import tensorflow.contrib.slim as slim 

from random import choice
from time import sleep
from time import time

from numpy.random import rand

from confounding_envs import *
from lib.networks import AC_Network_Confounding as AC_Network

class Worker():
    def __init__(self,game,name,a_size,trainer,model_path,global_episodes):
        self.name = "worker_" + str(name)
        self.number = name        
        self.model_path = model_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter("train_"+str(self.number))

        state_size = game.N

        #Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_AC = AC_Network(a_size,state_size,self.name,trainer)
        self.update_local_ops = update_target_graph('global',self.name)        
        self.env = game
        
    def train(self,rollout,sess,gamma,bootstrap_value):
        #print(rollout)
        #rollout = np.array(rollout)
        actions = np.array([r[0] for r in rollout])
        rewards = np.array([r[1] for r in rollout])
        timesteps = np.array([r[2] for r in rollout])
        states = np.array([r[5] for r in rollout])
        prev_rewards = [0] + rewards[:-1].tolist()
        prev_actions = [0] + actions[:-1].tolist()
        values = np.array([r[4] for r in rollout])
        
        self.pr = prev_rewards
        self.pa = prev_actions
        # Here we take the rewards and values from the rollout, and use them to 
        # generate the advantage and discounted returns. 
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus,gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages,gamma)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        rnn_state = self.local_AC.state_init
        feed_dict = {self.local_AC.target_v:discounted_rewards,
            self.local_AC.prev_rewards:np.vstack(prev_rewards),
            self.local_AC.prev_actions:prev_actions,
            self.local_AC.curr_states:states,
            self.local_AC.actions:actions,
            self.local_AC.timestep:np.vstack(timesteps),
            self.local_AC.advantages:advantages,
            self.local_AC.state_in[0]:rnn_state[0],
            self.local_AC.state_in[1]:rnn_state[1]}
        v_l,p_l,e_l,g_n,v_n,_ = sess.run([self.local_AC.value_loss,
            self.local_AC.policy_loss,
            self.local_AC.entropy,
            self.local_AC.grad_norms,
            self.local_AC.var_norms,
            self.local_AC.apply_grads],
            feed_dict=feed_dict)
        return v_l / len(rollout),p_l / len(rollout),e_l / len(rollout), g_n,v_n
        
    def work(self,gamma,sess,coord,saver,train):
        episode_count = sess.run(self.global_episodes)
        print(episode_count)
        total_steps = 0
        print("Starting worker " + str(self.number))
        with sess.as_default(), sess.graph.as_default():                 
            while not coord.should_stop():
                sess.run(self.update_local_ops)
                episode_buffer = []
                episode_values = []
                episode_frames = []
                episode_reward = [0,0]
                episode_step_count = 0
                d = False
                r = 0
                a = 0
                t = 0
                state = self.env.reset()
                rnn_state = self.local_AC.state_init
                
                #print("Starting episode")

                while d == False:
                    #Take an action using probabilities from policy network output.
                    a_dist,v,rnn_state_new = sess.run([self.local_AC.policy,self.local_AC.value,self.local_AC.state_out], 
                        feed_dict={
                        self.local_AC.prev_rewards:[[r]],
                        self.local_AC.timestep:[[t]],
                        self.local_AC.prev_actions:[a],
                        self.local_AC.curr_states:[state],
                        self.local_AC.state_in[0]:rnn_state[0],
                        self.local_AC.state_in[1]:rnn_state[1]})
                    a = np.random.choice(a_dist[0],p=a_dist[0])
                    a = np.argmax(a_dist == a)
                    
                    rnn_state = rnn_state_new
                    r,d,t,s = self.env.step(a)                        
                    episode_buffer.append([a,r,t,d,v[0,0],state])
                    episode_values.append(v[0,0])
                    #episode_frames.append(set_image_bandit(episode_reward,self.env.bandit,a,t))
                    episode_reward[a] += r
                    total_steps += 1
                    episode_step_count += 1
                    state = s
                    
                self.episode_rewards.append(np.sum(episode_reward))
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))
                
                # Update the network using the experience buffer at the end of the episode.
                if len(episode_buffer) != 0 and train == True:
                    v_l,p_l,e_l,g_n,v_n = self.train(episode_buffer,sess,gamma,0.0)
                    
                # Periodically save gifs of episodes, model parameters, and summary statistics.
                if episode_count % 50 == 0 and episode_count != 0:
                    if episode_count % 500 == 0 and self.name == 'worker_0' and train == True:
                        saver.save(sess,self.model_path+'/model-'+str(episode_count)+'.cptk')
                        print("Saved Model")

                    mean_reward = np.mean(self.episode_rewards[-50:])
                    mean_length = np.mean(self.episode_lengths[-50:])
                    mean_value = np.mean(self.episode_mean_values[-50:])

                    if episode_count % 100 == 0 and self.name == 'worker_0':
                        print("Mean reward: %f"%float(mean_reward))                            
                    #    self.images = np.array(episode_frames)
                    #    make_gif(self.images,'./frames/image'+str(episode_count)+'.gif',
                    #        duration=len(self.images)*0.1,true_image=True)

                    summary = tf.Summary()
                    summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                    summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                    summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
                    if train == True:
                        summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
                        summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
                        summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
                        summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                        summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
                    self.summary_writer.add_summary(summary, episode_count)

                    self.summary_writer.flush()
                if self.name == 'worker_0':
                    sess.run(self.increment)
                episode_count += 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--name', type=str, default="model_confounding")
    parser.add_argument('--env', type=str, default = "obs")
    parser.add_argument('--loadmodel', type=bool, default=False)
    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--gamma', type=float, default=0.8)
    parser.add_argument('--epochs', type=int, default=200)

    args = parser.parse_args()

    epochs = args.epochs
    gamma = args.gamma # discount rate for advantage estimation and reward discounting
    a_size = 2 # Agent can choose one of two topologies
    load_model = args.loadmodel
    train = args.train
    model_path = './models/' + args.name + '_' + args.env

    if args.env == "obs":
        #This environment has 3 variables. Here interventions are taking place, but no indicators are given. Thus the statistics
        #of the variables _may_ be enough to figure out the structure. 
        env = ObsEnv
    elif args.env == "confounded":
        #This is a completely ambiguous environment, there should not be enough information here to be able to solve the problem
        env = ConfoundedEnv
    elif args.env == "obs_int_env"
        #This environment has 5 variables. 3 for the states, and 2 that function as 'intervention' indicators
        #This should be the easiest environment to learn
        env = ObsIntEnv
    else:
        raise ValueError, "Not valid environment name"
    
    e = env()
    state_size = e.N
    
    tf.reset_default_graph()
    
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        
    if not os.path.exists('./frames'):
        os.makedirs('./frames')
        
    with tf.device("/cpu:0"): 
        global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
        trainer = tf.train.AdamOptimizer(learning_rate=1e-3)
        master_network = AC_Network(a_size,state_size,'global',None) # Generate global network
        #num_workers = multiprocessing.cpu_count() # Set workers ot number of available CPU threads
        num_workers = 1
        workers = []
        # Create worker classes
        for i in range(num_workers):
            workers.append(Worker(env(), i, a_size, trainer, model_path, global_episodes))
        saver = tf.train.Saver(max_to_keep=5)
    
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        if load_model == True:
            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(model_path)
            saver.restore(sess,ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())
            
        worker_threads = []
        for worker in workers:
            worker_work = lambda: worker.work(gamma,sess,coord,saver,train)
            thread = threading.Thread(target=(worker_work))
            thread.start()
            worker_threads.append(thread)
        coord.join(worker_threads)
    
    if __name__ == "main":
        main()