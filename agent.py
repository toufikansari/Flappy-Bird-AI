import flappy_bird_gymnasium
import gymnasium as gym
from dqn import DQN
from exprience_reply import ReplyMemory
import torch
import itertools
import yaml
import torch.nn as nn
import torch.optim as optim

if torch.backends.mps.is_available():
    device ="mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

class Agent:
    def __init__(self, params_set):
        self.param-set = params_set
        with open("parameter.ymal", "r") as f:
            all_param_set = yaml.safe_load(f)
            params = all_param_set[params_set]

        self.alpha = params['alpha']
        self.gamma = params["gamma"]

        self.epsilon_init = params["epsilon_init"]
        self.epsilon_min = params["epsilon_min"]
        self.epsilon_decay = params["epsilon_decay"]

        self.replay_memory_size = params["replay_memory_size"]
        self.mini_batch_size = params["mini_batch_size"]

        self.reward_theresold = params["reward_theresold"]
        self.network_sync_rate = params["network_sync_rate"]

        self.loss_fn = nn.MSELoss()
        self.optimizer = None

    def run(self, is_training=True, render=False):
        
        env = gym.make("FlappyBird-v0", render_mode="human" if render else None)

        num_state = env.observation_space.shape[0] # input dimension
        num_action = env.action_space.n # output dimnesion

        policy_dqn = DQN(num_state, num_action).to(device)

        if is_training:
            memory = ReplyMemory(self.replay_memory_size)

        for episode in itertools.count():
            state, _ = env.reset()
            episode_rewards = 0
            terminated = False

            while not terminated:
            
                action = env.action_space.sample()

                
                next_state, reward, terminated, _, _ = env.step(action)

                if is_training:
                    memory.append((state, action ,new_state,  reward , terminated))

                state = new_state
                episode_rewards += reward

            print(f"for episdoe={episode+1} total rewrd={episode_rewards} ")
            # env.close()  -- i want to manually stop