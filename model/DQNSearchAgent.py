import torch
import torch.nn as nn
import numpy as np
import random
import torch.optim as optim
from collections import deque

class DQN(nn.Module):
    def __init__(self, state_dim, num_actions,fc1_unit=64,fc2_unit = 64):
        """
        Initialize parameters and build model.
        Params
        =======
            state_dim (int): Dimension of each state
            num_actions (int): Dimension of each action
            seed (int): Random seed
            fc1_unit (int): Number of nodes in first hidden layer
            fc2_unit (int): Number of nodes in second hidden layer
        """
        super(DQN,self).__init__() ## calls __init__ method of nn.Module class

        self.fc1 = nn.Linear(state_dim, fc1_unit)
        self.fc2 = nn.Linear(fc1_unit, fc2_unit)
        self.fc3 = nn.Linear(fc2_unit, num_actions)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, state):
        """
        Build a network that maps state -> action values.
        """
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Agent(nn.Module):
    def __init__(self, args,device,rbt_model, rbt_tokenizer):
        super(Agent, self).__init__()  ## calls __init__ method of nn.Module class
        self.max_len=args.max_len

        self.model=rbt_model
        self.tokenizer=rbt_tokenizer
        self.state_dim = len(self.tokenizer.get_vocab()) # 50265
        self.num_actions = 3
        self.args=args
        self.device=device

    def text2emb(self, sent):
        # Obtain the input ids from the tokenizer
        encoded_input = self.tokenizer(sent, return_tensors='pt', add_special_tokens=True, max_length=self.max_len, truncation=True,padding=True)
        with torch.no_grad():
            input_ids = self.model(**encoded_input.to(self.device), output_hidden_states=True)
        # same as self.model(torch.tensor(self.tokenizer.encode(sent[0])).unsqueeze(0).to(device))

        # Apply average pooling to obtain a tensor of shape (BSZ, 50257)
        output=torch.mean(input_ids[0], dim=1)

        return output

    def act(self, state, epsilon, policy_net):
        """Returns action for given state as per current policy

        Params
        =======
            state (array_like): current state
            epsilon (float): for epsilon-greedy action selection

        """
        if random.random() > epsilon:
            state_emb=self.text2emb(state)
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                q_value = policy_net(state_emb)

            print("q_value",q_value)
            actions = q_value.max(1)[1]
        else:
            actions = torch.tensor(random.choices([0, 1, 2], k=self.args.bsz)).to(self.device) #.action_space.n=3, {deletion, insertion, replacement}
        return actions

    def forward(self, state):
        # Apply the linear regression model to obtain a tensor of shape (2, 50257)
        # Obtain the input ids from the tokenizer
        input_ids= self.text2emb(state)

        # Apply average pooling to obtain a tensor of shape (2, 50257)
        x = torch.mean(input_ids[0], dim=1)

        # Apply the linear regression model to obtain a tensor of shape (2, 1)
        x=x.view(-1, self.input_dim)
        x = torch.relu(self.dropout(self.fc1(x)))
        output = self.fc2(x) #action

        return output


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(states), actions, rewards, np.concatenate(next_states), dones

    def __len__(self):
        return len(self.buffer)
