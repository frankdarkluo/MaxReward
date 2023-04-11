import torch
import torch.nn as nn
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    def __init__(self, editor,args,num_actions=3):
        super(Agent, self).__init__()  ## calls __init__ method of nn.Module class
        self.max_len=editor.max_len
        self.model=editor.model
        self.tokenizer=editor.tokenizer
        self.state_dim = len(self.tokenizer.get_vocab())
        self.num_actions = num_actions
        self.args=args
        self.replay_buffer = ReplayBuffer(args.buffer_size)
        self.path=args.path

        # Q-Network
        self.policy_net = DQN(self.state_dim, num_actions).to(device)
        self.target_net = DQN(self.state_dim, num_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=args.lr)

    def text2emb(self, sent):
        # Obtain the input ids from the tokenizer
        encoded_input = self.tokenizer(sent, return_tensors='pt', add_special_tokens=True, max_length=self.max_len, truncation=True,padding=True)
        input_ids = self.model(**encoded_input.to(device), output_hidden_states=True)
        # same as self.model(torch.tensor(self.tokenizer.encode(sent[0])).unsqueeze(0).to(device))

        # Apply average pooling to obtain a tensor of shape (BSZ, 50257)
        output=torch.mean(input_ids[0], dim=1)

        return output

    def act(self, state, epsilon):
        """Returns action for given state as per current policy

        Params
        =======
            state (array_like): current state
            epsilon (float): for epsilon-greedy action selection

        """
        if random.random() > epsilon:
            state_emb=self.text2emb(state)
            q_value = self.policy_net(state_emb)
            actions = q_value.max(1)[1]
        else:
            actions = torch.tensor(random.choices([0, 1, 2], k=self.args.bsz)).to(device) #.action_space.n=3, {deletion, insertion, replacement}
        return actions

    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load_model(self, path):
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        return self.target_net

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

    def compute_td_loss(self):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.args.buffer_size)

        actions=torch.asarray(actions).to(device)
        rewards=torch.asarray(rewards).to(device)
        dones=torch.asarray(dones).to(device)+0 # From T/F to be 0/1

        self.policy_net.train()
        self.target_net.eval()
        q_values = self.policy_net(self.text2emb(states.tolist()))
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_net(self.text2emb(next_states.tolist()))
            next_q_value = next_q_values.max(1)[0]

        reward=torch.tensor(rewards).to(device)

        # Max DQN
        expected_q_value = torch.max(reward, next_q_value * (1 - dones))

        loss = (q_value - expected_q_value.data).pow(2).mean() # MSE loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.policy_net, self.target_net, self.args.tau)

        # ------------------- update target network ------------------- #
        return loss

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        =======
            local model (PyTorch model): weights will be copied from
            target model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter

        """
        for target_param, local_param in zip(target_model.parameters(),
                                             local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)
        self.save_model(self.path)


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
