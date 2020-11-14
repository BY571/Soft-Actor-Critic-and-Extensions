import numpy as np
import random
import gym
from collections import deque
import torch
import time
from torch.utils.tensorboard import SummaryWriter
import argparse
from  files import MultiPro
from files.Agent import Agent

def evaluate(frame, eval_runs=5, capture=False, render=False):
    """
    Makes an evaluation run with the current epsilon
    """

    reward_batch = []
    for i in range(eval_runs):
        state = eval_env.reset()
        if render: eval_env.render()
        rewards = 0
        while True:
            action = agent.act(np.expand_dims(state, axis=0))
            action_v = np.clip(action*action_high, action_low, action_high)
            state, reward, done, _ = eval_env.step(action_v[0])
            rewards += reward
            if done:
                break
        reward_batch.append(rewards)
    if capture == False:   
        writer.add_scalar("Reward", np.mean(reward_batch), frame)
    return np.mean(reward_batch)


def run(frames=1000, eval_every=1000, eval_runs=5, worker=1):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    i_episode = 1
    state = envs.reset()
    score = 0    
    if ERE:
        episode_K = 0
        eta_0 = 0.996
        eta_T = 1.0
        #episodes = 0
        max_ep_len = 500 # original = 1000
        c_k_min = 2500 # original = 5000

    for frame in range(1, frames+1):
        # evaluation runs
        if frame % eval_every == 0 or frame == 1:
            eval_reward = evaluate(frame*worker, eval_runs)

        action = agent.act(state)
        action_v = np.clip(action*action_high, action_low, action_high)
        next_state, reward, done, _ = envs.step(action_v) #returns np.stack(obs), np.stack(action) ...

        for s, a, r, ns, d in zip(state, action, reward, next_state, done):
            agent.step(s, a, r, ns, d, frame, ERE)
        
        if ERE:
            eta_t = eta_0 + (eta_T - eta_0)*(frame/(frames+1))
            episode_K +=1
        state = next_state
        score += np.mean(reward)
        
        if done.any():
            if ERE:
                for k in range(1,episode_K):
                    c_k = max(int(agent.memory.__len__()*eta_t**(k*(max_ep_len/episode_K))), c_k_min)
                    agent.ere_step(c_k)
            scores_window.append(score)       # save most recent score
            scores.append(score)              # save most recent score
            writer.add_scalar("Average100", np.mean(scores_window), frame*worker)
            print('\rEpisode {}\tFrame {} \tAverage100 Score: {:.2f}'.format(i_episode*worker, frame*worker, np.mean(scores_window)), end="")
            #if i_episode % 100 == 0:
            #    print('\rEpisode {}\tFrame \tReward: {}\tAverage100 Score: {:.2f}'.format(i_episode*worker, frame*worker, round(eval_reward,2), np.mean(scores_window)), end="", flush=True)
            i_episode +=1 
            state = envs.reset()
            score = 0
            episode_K = 0              




parser = argparse.ArgumentParser(description="")
parser.add_argument("-env", type=str,default="Pendulum-v0", help="Environment name, default = Pendulum-v0")
parser.add_argument("-per", type=int, default=0, choices=[0,1], help="Adding Priorizied Experience Replay to the agent if set to 1, default = 0")
parser.add_argument("-munchausen", type=int, default=0, choices=[0,1], help="Adding Munchausen RL to the agent if set to 1, default = 0")
parser.add_argument("-dist", "--distributional", type=int, default=0, choices=[0,1], help="Using a distributional IQN Critic if set to 1, default=0")
parser.add_argument("-ere", type=int, default=0, choices=[0,1], help="Adding Emphasizing Recent Experience to the agent if set to 1, default = 0")
parser.add_argument("-info", type=str, help="Information or name of the run")
parser.add_argument("-d2rl", type=int, choices=[0,1], default=0, help="Uses Deep Actor and Deep Critic Networks if set to 1 as described in the D2RL Paper: https://arxiv.org/pdf/2010.09163.pdf, default=0")
parser.add_argument("-frames", type=int, default=10000, help="The amount of training interactions with the environment, default is 100000")
parser.add_argument("-eval_every", type=int, default=1000, help="Number of interactions after which the evaluation runs are performed, default = 1000")
parser.add_argument("-eval_runs", type=int, default=3, help="Number of evaluation runs performed, default = 1")
parser.add_argument("-seed", type=int, default=0, help="Seed for the env and torch network weights, default is 0")
parser.add_argument("-lr_a", type=float, default=5e-4, help="Actor learning rate of adapting the network weights, default is 5e-4")
parser.add_argument("-lr_c", type=float, default=5e-4, help="Critic learning rate of adapting the network weights, default is 5e-4")
parser.add_argument("-a", "--alpha", type=float, help="entropy alpha value, if not choosen the value is leaned by the agent")
parser.add_argument("-layer_size", type=int, default=256, help="Number of nodes per neural network layer, default is 256")
parser.add_argument("-repm", "--replay_memory", type=int, default=int(1e6), help="Size of the Replay memory, default is 1e6")
parser.add_argument("-bs", "--batch_size", type=int, default=256, help="Batch size, default is 256")
parser.add_argument("-t", "--tau", type=float, default=1e-2, help="Softupdate factor tau, default is 1e-2")
parser.add_argument("-g", "--gamma", type=float, default=0.99, help="discount factor gamma, default is 0.99")
parser.add_argument("--saved_model", type=str, default=None, help="Load a saved model to perform a test run!")
parser.add_argument("-w", "--worker", type=int, default=1, help="Number of parallel worker, default = 1")
args = parser.parse_args()

if __name__ == "__main__":
    env_name = args.env
    seed = args.seed
    frames = args.frames
    worker = args.worker
    GAMMA = args.gamma
    TAU = args.tau
    HIDDEN_SIZE = args.layer_size
    BUFFER_SIZE = int(args.replay_memory)
    BATCH_SIZE = args.batch_size * worker
    LR_ACTOR = args.lr_a         # learning rate of the actor 
    LR_CRITIC = args.lr_c        # learning rate of the critic
    FIXED_ALPHA = args.alpha
    saved_model = args.saved_model
    ERE = args.ere
    D2RL = args.d2rl

    writer = SummaryWriter("runs/"+args.info)
    envs = MultiPro.SubprocVecEnv([lambda: gym.make(args.env) for i in range(args.worker)])
    eval_env = gym.make(args.env)
    envs.seed(seed)
    eval_env.seed(seed+1)
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(device))
    
    action_high = eval_env.action_space.high[0]
    action_low = eval_env.action_space.low[0]
    state_size = eval_env.observation_space.shape[0]
    action_size = eval_env.action_space.shape[0]
    agent = Agent(state_size=state_size, action_size=action_size, per=args.per, ere=args.ere, munchausen=args.munchausen, distributional=args.distributional,
                 D2RL=D2RL, random_seed=seed, hidden_size=HIDDEN_SIZE, BATCH_SIZE=BATCH_SIZE, BUFFER_SIZE=BUFFER_SIZE, GAMMA=GAMMA,
                 FIXED_ALPHA=FIXED_ALPHA, lr_a=LR_ACTOR, lr_c=LR_CRITIC, tau=TAU, worker=worker, device=device,  action_prior="uniform") #"normal"
    
    t0 = time.time()
    if saved_model != None:
        agent.actor_local.load_state_dict(torch.load(saved_model))
        evaluate(frame=None, capture=False)
    else:    
        run(frames = args.frames//args.worker,
            eval_every=args.eval_every//args.worker,
            eval_runs=args.eval_runs,
            worker=args.worker)
    t1 = time.time()
    eval_env.close()
    print("training took {} min!".format((t1-t0)/60))
