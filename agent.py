import flappy_bird_gymnasium
import gymnasium as gym
from dqn import DQN
from exprience_reply import ReplyMemory
import torch
import random
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
            epsilon = self.epsilon_init

            target_dqn =  DQN(num_state, num_action).to(device)
            # copy the wt & bias vals from policy  => target
            target_dqn.load_state_dict(policy_dqn.state_dict())

            staps =  0
            self.optimizer = optim.Adam(policy_dqn.parameters(), lr=self.alpha)

        for episode in itertools.count():
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float , device=device)

            episode_rewards = 0
            terminated = False

            while not terminated:
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample() # explore
                    action = torch.tensor(action, dtype=torch.long , device=device)
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax() # exploid

                
                next_state, reward, terminated, _, _ = env.step(action.item())

                #creat our tensor
                reward = torch.tensor(reward, dtype=torch.float , device=device)
                next_state = torch.tensor(next_state, dtype=torch.float , device=device)

                if is_training:
                    memory.append((state, action, new_state,  reward , terminated))
                    steps += 1

                state = new_state
                episode_rewards += reward

            print(f"for episdoe={episode+1} total rewrd={episode_rewards} & epsilon={epsilon} ")

            if is_training:
                #epsilin dice
                epsilon = max(epsilon *self.epsilon_decay, self.epsilon_min)

            if is_training and len(memory) > self.mini_batch_size:
                #get sample
                mini_batch = memory.sample(self.mini_batch_size)

                optimize(mini_batch, policy_dqn, target_dqn)

                #sync the network
                if steps > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    steps = 0
            # env.close()  -- i want to manually stop
    
    def optimize(self ,mini_batch, policy_dqn, target_dqn ):
        # get batch of expriences
        states , actions, next_state, rewards, termination = zip(*mini_batch)

        states = torch.stack(states)
        actions = torch.stack(actions)
        next_state = torch.stack(next_state)
        rewards = torch.stack(rewards)
        termination = torch.tensor(termination).float().to(device)

        # calculation target Q value from current ploicy
        with torch.no_grad():
            target_q = rewards + (1-termination) * self.gamma * target_dqn(next_state).max(dim=1)[0]


        # calculate y_pred i.e Q_value from current policy

        current_q = policy_dqn(states).gether(dim=1, index=actions.unsqueeze(dim=1)).squeeze()

        # loss
        loss = self.loss_fn(current_q, target_q)

        # optimezer model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
       

            
    