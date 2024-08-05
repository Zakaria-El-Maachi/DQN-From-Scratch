import torch
from torch import tensor
from torch.nn  import Linear, ReLU, MSELoss, Module, Dropout
from torch.optim import AdamW, lr_scheduler
from collections import deque, namedtuple
import random
import numpy as np
from tqdm import tqdm

Experience = namedtuple("Experience", ["state", "action", "reward", "nextState", "done"])    


class QCartpole(Module):
    def __init__(self, dropout=0.1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fc1 = Linear(4,32)
        self.relu = ReLU()
        self.fc2 = Linear(32, 16)
        self.relu2 = ReLU()
        self.fc3 = Linear(16, 2)
        self.dropout = Dropout(dropout)

    
    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu2(self.fc2(x)))
        return self.fc3(x)


class DQN:
    def __init__(self, env, lr=1e-3, discount=0.95, batchSize=64, updateSteps=500 ,buffermax=50000, timesteps=20000, learningStarts=500, policyNetwork=QCartpole, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        

        self.policy = policyNetwork()
        self.target = policyNetwork()
        
        self.replayBuffer = deque(maxlen=buffermax)
        self.batchSize = batchSize
        self.updateSteps = updateSteps
        self.discount = discount
        self.lr = lr
        self.timesteps = timesteps
        self.learningStarts = learningStarts
        self.env = env
        self.epsilon = 1
        
        self.criterion = MSELoss()
        self.optimizer = AdamW(self.policy.parameters(), lr=lr)
        self.scheduler = lr_scheduler.StepLR(optimizer=self.optimizer, step_size=500, gamma=0.9)
        
        
    def _remember(self, experience):
        self.replayBuffer.append(experience)
        
    def _backPropagation(self, predictions, labels):
        self.optimizer.zero_grad()
        loss = self.criterion(predictions, labels)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
            
    def _optimize(self):
        expBatch = random.sample(self.replayBuffer, self.batchSize)
        states = tensor(np.array([x.state for x in expBatch], dtype=np.float32))
        actions = tensor(np.array([x.action for x in expBatch]), dtype=torch.int64).unsqueeze(dim=1)
        rewards = tensor(np.array([x.reward for x in expBatch], dtype=np.float32))
        nextStates = tensor(np.array([x.nextState for x in expBatch], dtype=np.float32))
        dones = tensor(np.array([x.done for x in expBatch], dtype=np.float32))
        
        q = self.policy(states).gather(1, actions).squeeze()
        with torch.no_grad():
            qn = self.target(nextStates).max(dim=1).values
        qn[dones == True] = 0
        qTarget = rewards + self.discount * qn
        self._backPropagation(q, qTarget)
        
        
    def _targetUpdate(self):
        self.target.load_state_dict(self.policy.state_dict())
    
    def predict(self, obs):
        if isinstance(self.policy, QCartpole):
            actionLogits = self.policy(tensor(obs, dtype=torch.float32))
        else:
            actionLogits = self.policy(tensor(obs, dtype=torch.float32).unsqueeze(dim=0)).squeeze()
        return actionLogits.argmax(dim=0).item()
    
    def _greedy(self, obs):
        if random.random() <= self.epsilon:
            return random.randint(0, self.env.action_space.n-1)
        else:
            with torch.no_grad():
                return self.predict(obs)
        
        
    def _iterate(self, obs, greedy=True):
        state = obs
        action = self._greedy(state)
        obs, reward, terminated, finished, _ = self.env.step(action)
        done = terminated or finished
        self._remember(Experience(state, action, reward, obs, done))
        if done:
            obs, _ = self.env.reset()
        if greedy:
            self.epsilon = max(0.05, self.epsilon-0.03)
        return obs
       
    def learn(self):
        obs, _ = self.env.reset()
        for _ in range(self.learningStarts):
            obs = self._iterate(obs, False)
                
        obs, _ = self.env.reset()
        for i in tqdm(range(1, self.timesteps - self.learningStarts+1)):
            obs = self._iterate(obs)
            self._optimize()
            if i % self.updateSteps == 0:
                self._targetUpdate()
                