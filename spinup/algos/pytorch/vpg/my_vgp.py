import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import gym
import time
from torch.distributions.categorical import Categorical
import time
#import spinup.algos.pytorch.vpg.core as core
#from spinup.utils.logx import EpochLogger
#from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
#from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs

def vpg(env_fn, actor_critic=None, ac_kwargs=dict(),  seed=0, 
        steps_per_epoch=4000, epochs=50, gamma=0.99, pi_lr=3e-4,
        vf_lr=1e-3, train_v_iters=80, lam=0.97, max_ep_len=1000,
        logger_kwargs=dict(), save_freq=10):
    

    # Random seed
    #seed += 10000 * proc_id() + seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env = gym.make(env_fn)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # Prepare initial parameteres phi0 and theta0 and corresponding neural nets
    def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
        # Build a feedforward neural network. (Multilayer perceptron)
        layers = []
        for j in range(len(sizes)-1):
            act = activation if j < len(sizes)-2 else output_activation
            #print((sizes[j], sizes[j+1]))
            layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
        return nn.Sequential(*layers)

    hidden_sizes = [32]
    pi_net = mlp(sizes=[obs_dim]+hidden_sizes+[act_dim])
    value_net = mlp(sizes=[obs_dim]+[32]+[1])

    # make function to compute action distribution
    def get_policy(obs):
        #print('observations', obs)
        #time.sleep(1)
        logits = pi_net(obs)
        return Categorical(logits=logits)

    # make action selection function (outputs int actions, sampled from policy)
    def get_action(obs):
        return get_policy(obs).sample().item()

    # make function to compute value of a state
    def get_value(obs):
        return value_net(obs)

    def pi_loss(trajectory_number, advantages, observations, actions):
        logp = get_policy(observations).log_prob(actions)
        g = - (advantages*logp).sum()/trajectory_number
        return g

    def value_loss(reward_to_go, observations):
        error = (get_value(observations) - reward_to_go)**2
        return error.mean()


    # make optimizer
    pi_optimizer = Adam(pi_net.parameters(), lr=pi_lr)
    value_optimizer = Adam(value_net.parameters(), lr=vf_lr)

    # for number of epochs

    trajectories = []


    def train_one_epoch():
        tau = 0
        trajectory_number = 0

        observations = []
        actions = []
        rewards = []
        reward_to_go = []
        advantages = []

        trajectory_times = []

        while tau < steps_per_epoch: # For each trajectory

            observation = env.reset()
            observations.append(observation)

            done = False

            for t in range(max_ep_len): # For each step in the trajectory
                #Collect a set of trajectories acting on the environment

                action = get_action(torch.as_tensor(observation, dtype=torch.float32))
                #print(observation, type(observation))
                #time.sleep(1.5)
                observation, reward, done, _ = env.step(action)


                actions.append(action)
                rewards.append(reward)

                if done or t == max_ep_len-1:
                    tau += t+1
                    last_trajectory_lenght = t+1
                    break
                else:
                    observations.append(observation)

            
            G = 0

            trajectory_reward_to_go = []
            #Compute rewards to go, using Monte Carlo G updates (page 92 Sutton and Barto)
            for t in range(tau-1,tau-last_trajectory_lenght-1,-1):
                G = gamma*G+rewards[t]
                trajectory_reward_to_go.append(G)
            
            #Since the reward_to_go is calculated backwards, we have to reverse the list and then append to the list of rewards to go
            trajectory_reward_to_go.reverse()
            reward_to_go += trajectory_reward_to_go

            #Compute advantage estimates A = Q(s,a)-V(s)
            #print(tau, len(observations))
            #time.sleep(1.5)
            for t in range(tau-last_trajectory_lenght, len(observations)):

                if t == len(observations)-1 and done == True:
                    V_s_t = 0
                else:
                    V_s_t = get_value(torch.as_tensor(observations[t+1], dtype=torch.float32))

                #print('len advantages',len(advantages))
                #print('len observations',len(observations))
                advantages.append(rewards[t]+ gamma*V_s_t - get_value(torch.as_tensor(observations[t], dtype=torch.float32)))

            # Calculate total time of steps in this epoch, and the number of trajectories
            trajectory_times.append(tau)
            trajectory_number += 1

        print('len(advantages),len(rewards),len(observations),len(actions)')
        print(len(advantages),len(rewards),len(observations),len(actions))

        observations = torch.as_tensor(observations, dtype=torch.float32)
        actions = torch.as_tensor(actions, dtype=torch.float32)
        rewards = torch.as_tensor(rewards, dtype=torch.float32)
        reward_to_go = torch.as_tensor(reward_to_go, dtype=torch.float32)
        advantages = torch.as_tensor(advantages, dtype=torch.float32)

        # Normalize the advantages
        adv_mean = torch.mean(advantages)
        adv_std = torch.std(advantages)
        advantages = (advantages-adv_mean)/adv_std

        # Estimate gradient policy
        # g = 1/D_k * sum_k sum_t grad_theta log_probs A
        pi_optimizer.zero_grad()
        loss_pi = pi_loss(trajectory_number, advantages, observations, actions)
        loss_pi.backward()
        pi_optimizer.step()

        # Fit value function using several steps of gradient descent
        for _ in range(train_v_iters):
            value_optimizer.zero_grad()
            loss_value = value_loss(reward_to_go, observations)
            loss_value.backward()
            value_optimizer.step()
        
        # Create a list of trajectory lengths
        trajectory_lengths = [trajectory_times[0]]
        for i in range(1,len(trajectory_times)):
            trajectory_lengths.append(trajectory_times[i]-trajectory_times[i-1])

        return loss_pi, loss_value, reward_to_go, trajectory_lengths
    
    # training loop
    for i in range(epochs):
        loss_pi, loss_value, reward_to_go, trajectory_lengths = train_one_epoch()
        print('epoch: %3d \t pi_loss: %.3f \t value_loss: %.3f \t return: %.3f \t trajectory_lengths: %.3f'%
                (i, loss_pi, loss_value, np.mean(np.array(reward_to_go)), np.mean(np.array(trajectory_lengths))))

vpg('CartPole-v0')