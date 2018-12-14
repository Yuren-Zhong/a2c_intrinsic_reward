import argparse
import gym
import numpy as np
from itertools import count
from collections import namedtuple
import cv2
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from model_component.cuda import cuda_wrapper


parser = argparse.ArgumentParser(description='PyTorch actor-critic')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
args = parser.parse_args()


env = gym.make('RoadRunner-v0')
env.seed(args.seed)
torch.manual_seed(args.seed)
action_space_size = env.action_space.n

model_weight_path = 'weights\\model_weight'

# hyperparameters
learning_rate = 0.0007
gamma = 0.99
img_w = 200
img_h = 160

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

class MLPPolicy(nn.Module):
    def __init__(self):
        super(MLPPolicy, self).__init__()

        self.affine1 = nn.Linear(4, 128)
        self.action_head = nn.Linear(128, 2)
        self.value_head = nn.Linear(128, 1)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(action_scores, dim=-1), state_values

class CNNPolicy(nn.Module):
    def __init__(self):
        super(CNNPolicy, self).__init__()
        self.conv1 = cuda_wrapper(nn.Conv2d(3, 32, 8, stride=4, padding=2))
        self.conv2 = cuda_wrapper(nn.Conv2d(32, 64, 4, stride=2, padding=1))
        self.conv3 = cuda_wrapper(nn.Conv2d(64, 64, 3, stride=1, padding=1))
        self.fc = cuda_wrapper(nn.Linear(img_w*img_h, 16))

        self.action_head = cuda_wrapper(nn.Linear(16, action_space_size))
        self.value_head = cuda_wrapper(nn.Linear(16, 1))

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = cuda_wrapper(F.relu(self.conv1(x)))
        x = cuda_wrapper(F.relu(self.conv2(x)))
        x = cuda_wrapper(F.relu(self.conv3(x)))
        x = cuda_wrapper(x.view(-1, img_w*img_h))
        x = cuda_wrapper(F.relu(self.fc(x)))
        
        action_scores = cuda_wrapper(self.action_head(x))
        state_values = cuda_wrapper(self.value_head(x))
        return cuda_wrapper(F.softmax(action_scores, dim=-1)), state_values


model = CNNPolicy()
# model.load_state_dict(torch.load("weights\model_weight6650"))
# model.eval()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
eps = np.finfo(np.float32).eps.item()

def select_action(state):
    state = preprocess(state)
    state = cuda_wrapper(state)
    probs, state_value = model(state)
    m = Categorical(probs)
    action = m.sample()
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value[0]))
    return action.item()

def finish_episode():
    R = 0
    saved_actions = model.saved_actions
    policy_losses = []
    value_losses = []
    rewards = []
    for r in model.rewards[::-1]:
        R = r + gamma * R
        rewards.insert(0, R)
    rewards = torch.tensor(rewards)
    rewards = cuda_wrapper(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    for (log_prob, value), r in zip(saved_actions, rewards):
        reward = r - value.item()
        policy_losses.append(-log_prob * reward)
        value_losses.append(F.smooth_l1_loss(cuda_wrapper(value), cuda_wrapper(torch.tensor([r]))))
    optimizer.zero_grad()
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    loss.backward()
    optimizer.step()
    del model.rewards[:]
    del model.saved_actions[:]

def preprocess(img):

    def rgb2gray(rgb):
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

    img = img / 255.0
    img = cv2.resize(img, dsize=(img_w, img_h))
    img = torch.from_numpy(np.array([img.transpose(2,0,1)])).float()

    return img

def main():
    running_reward = 0
    for i_episode in count(1):
        state = env.reset()
        for t in range(10000):  # Don't infinite loop while learning
            action = select_action(state)
            state, reward, done, _ = env.step(action)
            if args.render:
                env.render()
            model.rewards.append(reward)
            if done:
                break


        running_reward = running_reward * 0.99 + t * 0.01
        finish_episode()

        print('Episode {}\tLast Reward: {:2d}\tAverage Reward: {:.2f}'.format(
              i_episode, t, running_reward))
        if i_episode % args.log_interval == 0:
            sys.stdout.flush()
            torch.save(model.state_dict(), model_weight_path+str(i_episode))
        if i_episode > 5e7:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


if __name__ == '__main__':
    main()
