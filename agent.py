import flappy_bird_gymnasium
import gymnasium as gym
from dqn import DQN
from exprience_reply import ReplyMemory
import torch

if torch.backends.mps.is_available():
    device ="mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def run(self, is_training=True, render=False):
    env = gym.make("FlappyBird-v0", render_mode="human" if rander else None)

    num_state = env.observation_space.shape[0] # input dimension
    num_action = env.action_space.n # output dimnesion

    policy_dqn = DQN(num_state, num_action).to(device)

    obs, _ = env.reset()

    if is_training:
        memory = ReplyMemory(10000)
    while True:
        # Next action:
        # (feed the observation to your agent here)
        action = env.action_space.sample()

        # Processing: terminated => done
        next_state, reward, done, _, _ = env.step(action)

        if is_training:
            memory.append((state, action ,new_state,  reward , terminated))
        
        # Checking if the player is still alive
        if done:
            break

        env.close()