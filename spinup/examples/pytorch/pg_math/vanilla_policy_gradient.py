import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gym
from gym.spaces import Discrete, Box
import time

def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    # Build a feedforward neural network.
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layer = nn.Linear(sizes[j], sizes[j+1])
        #nn.init.xavier_uniform_(layer.weight)
        layers += [layer, act()]
    return nn.Sequential(*layers)

def reward_to_go(rews,gamma):
    # taken from file 2_rtg_pg.py
    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (gamma*rtgs[i+1] if i+1 < n else 0)
    return rtgs

def train(env_name='CartPole-v0', hidden_sizes=[64,64], pi_lr=3e-4, vf_lr=1e-3, epochs=150, 
        batch_size=4000, render=False, gamma=1, lam = 0.97, train_v_iters=200):

    # make environment, check spaces, get obs / act dims
    env = gym.make(env_name)
    assert isinstance(env.observation_space, Box), \
        "This example only works for envs with continuous state spaces."
    assert isinstance(env.action_space, Discrete), \
        "This example only works for envs with discrete action spaces."

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    # make core of policy and value networks
    pi_net = mlp(sizes=[obs_dim]+hidden_sizes+[n_acts])
    value_net = mlp(sizes=[obs_dim]+hidden_sizes+[1])#, output_activation=nn.Tanh)


    # make function to compute action distribution
    def get_policy(obs):
        logits = pi_net(obs)
        return Categorical(logits=logits)

    # make action selection function (outputs int actions, sampled from policy)
    def get_action(obs):
        return get_policy(obs).sample().item()

    # make function to compute value of a state
    def get_value(obs):
        return value_net(obs)

    # make loss function whose gradient, for the right data, is policy gradient
    def compute_pi_loss(obs, act, adv):
        logp = get_policy(obs).log_prob(act)
        return -(logp * adv).mean()

    # Set up function for computing value loss
    def compute_v_loss(obs, rtg):
        return ((get_value(obs) - rtg)**2).mean()

    # Helper function to calculate cumulative discounted sum
    def discount_cumsum(x, discount):
        y = np.zeros(len(x))
        G = 0
        for i in reversed(range(len(x))):
            G = x[i] + discount*G
            y[i] = G
        return y

    # Compute advantages
    def compute_episode_advantages(rews, vals, gamma=gamma, lam=lam):
        deltas = []
        for i in range(len(rews)-1):
            deltas.append(rews[i]+gamma*vals[i+1]-vals[i])
        deltas.append(rews[i]-vals[i]) # the value of the termination state is 0.
        return discount_cumsum(deltas, gamma * lam)

    # Normalize a list
    def normalize(l):
        l = (torch.as_tensor(l, dtype=torch.float32) - torch.mean(torch.as_tensor(l, dtype=torch.float32)))/ (torch.std(torch.as_tensor(l, dtype=torch.float32))+1e-8)
        return list(l)

    # make optimizer
    pi_optimizer = Adam(pi_net.parameters(), lr=pi_lr, weight_decay = .1)
    value_optimizer = Adam(value_net.parameters(), lr=vf_lr, weight_decay=.1)

    # for training policy
    def train_one_epoch():
        # make some empty lists for logging.
        batch_obs = []          # for observations
        batch_acts = []         # for actions
        batch_reward_to_go = []         # for measuring episode returns
        batch_lens = []         # for measuring episode lengths
        batch_advantages = []   # for advantages Q(s,a) - V(s) = R + gamma*V(s') - V(s)

        # reset episode-specific variables
        obs = env.reset()       # first obs comes from starting distribution
        done = False            # signal from environment that episode is over
        episode_rewards = []            # list for rewards accrued throughout ep
        episode_obs = []
        batch_total_reward = 0

        # render first episode of each epoch
        finished_rendering_this_epoch = False

        # collect experience by acting in the environment with current policy
        while True:
            # rendering
            if (not finished_rendering_this_epoch) and render:
                env.render()

            prev_obs = obs
            batch_obs.append(prev_obs)
            episode_obs.append(prev_obs)
                    
            # calculate value of initial step
            prev_val = get_value(torch.as_tensor(prev_obs, dtype = torch.float32))

            # act in the environment
            act = get_action(torch.as_tensor(prev_obs, dtype=torch.float32))
            obs, rew, done, _ = env.step(act)

            # save action and reward
            batch_acts.append(act)
            episode_rewards.append(rew)
            batch_total_reward += rew

            if done: # that is, if the episode ended

                # batch length
                ep_len = len(episode_rewards)
                batch_lens.append(ep_len)

                # batch_reward_to_go
                batch_reward_to_go += list(reward_to_go( episode_rewards, gamma))

                # batch advantages
                vals = list(get_value(torch.as_tensor(episode_obs, dtype=torch.float32)))
                batch_advantages += list( compute_episode_advantages(rews = episode_rewards, vals = vals))

                # end experience loop if we have enough of it
                if len(batch_obs) >= batch_size:
                    assert(len(batch_advantages) == len(batch_acts) == len(batch_obs) == len(batch_reward_to_go))
                    break

                # reset episode-specific variables
                obs, done, episode_rewards, episode_obs = env.reset(), False, [], []

                # won't render again this epoch
                finished_rendering_this_epoch = True
        
        # normalize the advantages to have mean 0 and std 1. This is done in the spinningup algorithm
        batch_advantages = normalize(batch_advantages)
    
        # take a single policy gradient update step
        pi_optimizer.zero_grad()
        batch_loss_pi = compute_pi_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32),
                                  act=torch.as_tensor(batch_acts, dtype=torch.int32),
                                  adv=torch.as_tensor(batch_advantages, dtype=torch.float32),
                                  )
        batch_loss_pi.backward()
        pi_optimizer.step()

        # take several steps fitting the value function
        for _ in range(train_v_iters):
            value_optimizer.zero_grad()
            batch_loss_v = compute_v_loss(obs = torch.as_tensor(batch_obs, dtype=torch.float32),
                                          rtg = torch.as_tensor(batch_reward_to_go, dtype=torch.float32)
                                          )
            batch_loss_v.backward()
            value_optimizer.step()

        batch_mean_reward = batch_total_reward / len(batch_reward_to_go)

        return batch_loss_pi, batch_loss_v, batch_mean_reward, batch_lens

    # training loop
    for i in range(epochs):
        batch_loss_pi, batch_loss_v, batch_mean_reward, batch_lens = train_one_epoch()
        print('epoch: %3d \t pi_loss: %.3f \t value_loss: %.3f \t return: %.3f \t ep_len: %.3f \t'%
                (i, batch_loss_pi, batch_loss_v, batch_mean_reward, np.mean(batch_lens)))

train(env_name='CartPole-v0')