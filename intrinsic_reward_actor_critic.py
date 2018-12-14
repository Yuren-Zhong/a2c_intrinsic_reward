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

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
args = parser.parse_args()


env = gym.make('RoadRunner-v0')
env.seed(args.seed)
torch.manual_seed(args.seed)
action_space_size = env.action_space.n

policy_net_weight_path = 'weights\\policy_net_weight'
inner_net_weight_path = 'weights\\inner_net_weight'

# hyperparameters
learning_rate_alpha = 0.0007
learning_rate_beta = 0.0007
lamda = 0.01
gamma = 0.99
img_w = 200
img_h = 160

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

class MLPIntrinsic(nn.Module):
    def __init__(self):
        super(Intrinsic, self).__init__()
        self.fc1 = nn.Linear(5, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 1) 
        self.activ = nn.ReLU()

        self.rewards = []
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.activ(x)
        x = self.fc2(x)
        x = self.activ(x)
        x = self.fc3(x)
        return self.activ(x)

class CNNIntrinsic(nn.Module):
    def __init__(self):
        super(CNNIntrinsic, self).__init__()
        self.conv1 = cuda_wrapper(nn.Conv2d(3, 32, 8, stride=4, padding=2))
        self.conv2 = cuda_wrapper(nn.Conv2d(32, 64, 4, stride=2, padding=1))
        self.conv3 = cuda_wrapper(nn.Conv2d(64, 64, 3, stride=1, padding=1))
        self.fc = cuda_wrapper(nn.Linear(img_w*img_h, 16))

        self.head = cuda_wrapper(nn.Linear(17, 1))

        self.rewards = []

    def forward(self, x, y):
        x = cuda_wrapper(F.relu(self.conv1(x)))
        x = cuda_wrapper(F.relu(self.conv2(x)))
        x = cuda_wrapper(F.relu(self.conv3(x)))
        x = cuda_wrapper(x.view(-1, img_w*img_h))
        x = cuda_wrapper(F.relu(self.fc(x)))
        x = cuda_wrapper(x[0])
        return cuda_wrapper(self.head(torch.cat((x, cuda_wrapper(y.type(torch.FloatTensor))))))

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

policy_net = CNNPolicy()
# policy_net.load_state_dict(torch.load("weights\policy_net_weight4100"))
# policy_net.eval()
inner_net = CNNIntrinsic()
# inner_net.load_state_dict(torch.load("weights\inner_net_weight4100"))
# inner_net.eval()
optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate_alpha)
hidden = optim.Adam(inner_net.parameters(), lr=learning_rate_beta)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    probs, state_value = policy_net(state)
    m = Categorical(probs)
    action = m.sample()
    policy_net.saved_actions.append(SavedAction(m.log_prob(action), state_value[0]))
    return action.item()


def finish_episode():
    saved_actions = policy_net.saved_actions
    policy_losses = []
    value_losses = []
    inner_losses = []
    rewards_ex = []
    rewards_in = []
    
    R = 0
    for r in policy_net.rewards[::-1]:
        R = r + gamma * R
        rewards_ex.insert(0, R)
    R = 0
    for r in inner_net.rewards[::-1]:
        R = r + gamma * R
        rewards_in.insert(0, R)

    rewards_ex = cuda_wrapper(torch.tensor(rewards_ex))
    rewards_ex = (rewards_ex - rewards_ex.mean()) / (rewards_ex.std() + eps)
    rewards_in = cuda_wrapper(torch.tensor(rewards_in))
    rewards_in = (rewards_in - rewards_in.mean()) / (rewards_in.std() + eps)

    for (log_prob, value), r, reward_in in zip(saved_actions, rewards_ex, rewards_in):
        reward_ex = r - value.item()
        policy_losses.append(- log_prob * reward_ex - log_prob * lamda * reward_in)
        value_losses.append(F.smooth_l1_loss(cuda_wrapper(value), cuda_wrapper(torch.tensor([r]))))
        inner_losses.append(- log_prob * reward_ex)
    
    optimizer.zero_grad()
    policy_loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    policy_loss.backward(retain_graph=True)
    optimizer.step()
    
    hidden.zero_grad()
    inner_loss = torch.stack(inner_losses).sum()
    inner_loss.backward()
    hidden.step()

    del policy_net.rewards[:]
    del policy_net.saved_actions[:]
    del inner_net.rewards[:]

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
            state = preprocess(state)
            state = cuda_wrapper(state)
            action = select_action(state)
            reward_in = inner_net(state, cuda_wrapper(torch.from_numpy(np.array([action]))))
            state, reward_ex, done, _ = env.step(action)
            policy_net.rewards.append(reward_ex)
            inner_net.rewards.append(reward_in)
            if done:
                break

        running_reward = running_reward * 0.99 + t * 0.01 
        finish_episode()

        print('Episode {}\tLast Reward: {:2d}\tAverage Reward: {:.2f}'.format(
              i_episode, t, running_reward))
        if i_episode % args.log_interval == 0:
            sys.stdout.flush()
            torch.save(policy_net.state_dict(), policy_net_weight_path+str(i_episode))
            torch.save(inner_net.state_dict(), inner_net_weight_path+str(i_episode))
        if i_episode > 5e7:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break

if __name__ == '__main__':
    main()
