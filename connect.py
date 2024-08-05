from gymnasium.spaces import Discrete
import numpy as np
from random import choice

class Connect:
    def __init__(self, turn=1) -> None:
        super().__init__()
        self.board = np.array([[0]*6 for _ in range(7)])
        self.action_space = Discrete(7)
        self.turn = turn
                
    def reset(self, board=None, turn=1):
        if board is None:
            self.board = np.array([[0]*6 for _ in range(7)])
        else:
            self.board = board
        self.turn = turn
        
    def step(self, action):
        action = int(action)
        truncated = False
        terminated = False
        if self.board[action][5] != 0:
            truncated = True
            reward = -1
        else:
            for row in range(6):
                if self.board[action][row] == 0:
                    break
            self.board[action][row] = self.turn
            reward = self._get_reward(self.board, row, action, self.turn)
            self.turn *= -1
            if reward != 0:
                terminated = True
        
        if not truncated and not terminated:
            for i in range(7):
                if self.board[i][5] == 0:
                    break
            else:
                terminated = True

        return self.board, reward, terminated, truncated, {}
    
    def copy(self):
        newEnv = Connect(self.turn)
        newEnv.reset(np.copy(self.board), self.turn)
        return newEnv
    
    @staticmethod
    def _get_reward(board, row, action, turn):
        
        sum = 0
        for i in range(max(0, action-3), min(6, action+3)+1):
            if board[i][row]==turn:
                sum += 1
            else:
                sum = 0
            if sum == 4:
                return 1
    
        sum = 0
        for i in range(max(0, row-3), min(5, row+3)+1):
            if board[action][i]==turn:
                sum += 1
            else:
                sum = 0
            if sum == 4:
                return 1
        
        sum = 0
        for i in range(max(max(-action, -row), -3), min(min(6-action, 5-row), 3)+1):
            if board[action+i][row+i]==turn:
                sum += 1
            else:
                sum = 0
            if sum == 4:
                return 1
            
        sum = 0
        for i in range(max(max(-action, row-5), -3), min(min(6-action, row), 3)+1):
            if board[action+i][row-i]==turn:
                sum += 1
            else:
                sum = 0
            if sum == 4:
                return 1
        
        return 0
    

    def __str__(self):
        game = ""
        for row in range(5, -1, -1):
            for col in range(7):
                if self.board[col][row] == 1:
                    game += "| X "
                elif self.board[col][row] == -1:
                    game += "| O "
                else:
                    game += "| . "
            game += "|\n-----------------------------\n"
        return game


class RandomPlayer:
    def getAction(self, env):
        return env.action_space.sample()

    
class MiniMax:
    def __init__(self, depth=4) -> None:
        self.depth = depth
        
    def getAction(self, env):
        nenv = env.copy()
        nenv.reset(nenv.board * nenv.turn)
        move, best = self._recurse(0, True, nenv)
        
        if best == -1:
            possibleMoves = []
            for i in range(7):
                if env.board[i][5] == 0:
                    possibleMoves.append(i)
            return choice(possibleMoves)
                    
        return move
    
    @staticmethod
    def func(move, best, action, current, maximizer):
        if maximizer:
            if best < current:
                return action, current
            return move, best
        if best > current:
            return action, current
        return move, best
    
    def _recurse(self, depth, maximizer, env):
        if depth == self.depth:
            return 0, 0
        move, best = 0, 1
        if maximizer:
            best = -1
            
        for i in range(7):
            if env.board[i][5] != 0:
                continue
            
            nenv = env.copy()
            _, reward, terminated, _, _ = nenv.step(i)
            if terminated:
                reward *= env.turn
                move, best = self.func(move, best, i, reward, maximizer)
                continue
            _, best1 = self._recurse(depth+1, not maximizer, nenv)
            move, best = self.func(move, best, i, best1, maximizer)     
            
        return move, best


if __name__ == "__main__":
    game = Connect()
    # game.reset(board=[[-1, -1, -1, 1, -1, -1], [-1, 0, 0, 0, 0, 0], [1, 1, 1, -1, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [1, 1, 1, -1, 1, 0], [0, 0, 0, 0, 0, 0]])
    # game.reset(board=[[-1, -1, -1, 1, -1, -1], [-1, -1, -1, 1, 0, 0], [1, 1, 1, -1, 0, 0], [-1, -1, -1, 0, 0, 0], [1, 1, -1, 1, 0, 0], [1, 1, 1, -1, 1, 0], [1, -1, 1, 1, 0, 0]])
    computer = MiniMax()
    
    done = False
    while not done:
        
        action = int(input("Player Move : "))
        _, reward, tr, ter, _ = game.step(action)
        print(f"Player Reward : {reward}")
        done = tr or ter
        if done:
            break
        
        action = computer.getAction(game)
        print(f"Computer Move : {action}")
        _, reward, tr, ter, _ = game.step(action)
        print(f"Computer Reward : {reward}")
        done = tr or ter
        
        print(game)
    