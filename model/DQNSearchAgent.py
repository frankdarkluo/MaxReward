import torch
import torch.nn as nn
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import random






class DQNAgent(nn.Module):
    def __init__(self, editor,num_actions=None,input=None):
        super(DQNAgent,self).__init__()
        self.max_len=editor.max_len
        self.model=editor.model
        self.tokenizer=editor.tokenizer
        self.input = input
        self.num_actions = num_actions

        self.input_dim = len(self.tokenizer.get_vocab())
        # Define a linear regression model
        self.fc1 = torch.nn.Linear(self.input_dim, 128)
        self.fc2 = torch.nn.Linear(128, self.num_actions)
        self.dropout = torch.nn.Dropout(0.5)

    def text2emb(self, sent):
        encoded_input = self.tokenizer(sent, return_tensors='pt', add_special_tokens=True, max_length=self.max_len, truncation=True,padding=True)
        input_ids = self.model(**encoded_input.to(device), output_hidden_states=True)
        # same as self.model(torch.tensor(self.tokenizer.encode(sent[0])).unsqueeze(0).to(device))
        return input_ids


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * \
                       np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def act(self, state, epsilon):
        if random.random() > epsilon:
            q_value = self.forward(state)
            action = q_value.max(1)[1].item()
        else:
            action = random.randrange(self.num_actions) #.action_space.n=3, {deletion, insertion, replacement}
        return action

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




