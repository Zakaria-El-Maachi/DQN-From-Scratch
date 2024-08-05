from torch.nn  import Linear, Conv2d, ReLU, Module, Dropout
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
from connect import Connect, RandomPlayer
import numpy as np

class QConnect(Module):
    def __init__(self, in_channels=1, height=6, width=7, out_channels=4, kernel_size=4, stride=1, padding=3, dropout=0.2, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv = Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding
        )
        self.reshape_size = out_channels*((height-kernel_size+2*padding)//stride + 1) * ((width-kernel_size+2*padding)//stride + 1)
        self.fc1 = Linear(self.reshape_size,768)
        self.relu = ReLU()
        self.fc2 = Linear(768, 128)
        self.relu2 = ReLU()
        self.fc3 = Linear(128, 7)
        self.dropout = Dropout(dropout)
        
    def forward(self, x):
        n = x.shape[0]
        x = x.unsqueeze(dim=1)
        x = self.conv(x)
        x = x.reshape(n, 4*10*9)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu2(self.fc2(x)))
        return self.fc3(x)
    
    
    
class ConnectTrainer(Env):
    def __init__(self, turn=1, opponent=RandomPlayer) -> None:
        super().__init__()
        self.game = Connect(turn)
        self.firstTurn = turn
        self.opponent = opponent()
        
        self.observation_space = Box(low=-1, high=1, shape=(7, 6), dtype=np.int8)
        self.action_space = Discrete(7)
        
        self.reset()
        
        
    def reset(self, seed=None, options=None):
        self.game.reset(turn=self.firstTurn)
        if self.firstTurn != 1:
            action = self.opponent.getAction(self.game)
            self.game.step(action)
        
        return self.game.board, {}
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.game.step(action)
        
        if terminated or truncated:
            return obs, reward, terminated, truncated, info
        
        action2 = self.opponent.getAction(self.game)
        obs, reward, terminated, truncated, info = self.game.step(action2)
        return obs, reward*-1, terminated, truncated, info
        
    def render(self):
        print(f"\nPlayer {(self.game.turn+1)//2 + 1}")
        print(self.game)
        


# if __name__ == "__main__":
#     import stable_baselines3 as sb
#     env = ConnectTrainer()
#     model = sb.DQN('MlpPolicy', env, verbose=1, learning_rate=1e-3, buffer_size=50000, learning_starts=500, batch_size=64, gamma=0.99, target_update_interval=500)
#     model.learn(total_timesteps=5000)

#     env = ConnectTrainer()
#     obs, _ = env.reset()
#     for i in range(20):
#         action, _states = model.predict(obs)
#         obs, rewards, terminated, finished, _ = env.step(action)
#         print(env.game.board)
#         if terminated or finished:
#             obs, _ = env.reset()

#     env.close()