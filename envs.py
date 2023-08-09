from collections import deque
import gymnasium as gym

class FrozenLakeMod(gym.Wrapper):

    num_steps = 0
    rew = None

    def __init__(self, env, rew=[0,0,1]):
        super().__init__(env)
        self.rew = rew

    def step(self, action):
        self.num_steps += 1
        obs, reward, terminated, truncated, info = self.env.step(action)

        if terminated:
            #reward = 100 if reward == 1 else -100
            reward = self.rew[2] if reward == 1 else self.rew[1]
        else:
            #reward = -10
            reward = self.rew[0]

        return obs, reward, terminated, truncated, info
        


            
def plot_v_history(V_history, targets=None):
    import numpy as np
    import matplotlib.pyplot as plt
    
    history = {}
    
    num_episodes = len(V_history)
    for k in range(num_episodes):
        V = V_history[k]
        for s in V.keys():
            if s not in history.keys():
                history[s] = np.zeros(num_episodes)
            history[s][k] = V[s]
        
    for s in sorted(list(history.keys())):
        plt.plot(history[s], label=s)
        if targets is not None:
            plt.axhline(y=targets[s], linewidth=1, linestyle='--')
        
    plt.legend(bbox_to_anchor=(1,1))
    plt.show()

class MtnCarContMod(gym.Wrapper):

    def __init__(self, env, rew_struct):
        super().__init__(env)
        self.positions = deque([], maxlen=20)
        self.rew_struct = rew_struct

    def step(self, action):
        #self.num_steps += 1
        state, reward, terminated, truncated, info = self.env.step(action)
        self.positions.append(state[0])

        reward = self.rew_struct(state)
        
        return state, reward, terminated, truncated, info
