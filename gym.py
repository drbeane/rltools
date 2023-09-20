import gymnasium as ogym

def make(
    name, prob=None, rew_struct=None, size=None, 
    num_bins=None, max_angle=None, sqrt_trans=False, coord_max=None,
    record_states=False,
    **kwargs):
    
    env = ogym.make(name, disable_env_checker=True, **kwargs)
    if name == 'FrozenLake-v1':
        if rew_struct is None: rew_struct = [0,0,1]
        env = FrozenLakeMod(env=env, prob=prob, rew_struct=rew_struct)
    elif name == 'CartPole-v1':
        env = ogym.make(name, disable_env_checker=True, max_episode_steps=1000, **kwargs)
        if max_angle is None: max_angle=1.57
        env.spec.disable_env_checker = True
        env = CartPoleMod(env=env, num_bins=num_bins, sqrt_trans=sqrt_trans, 
                          max_angle=max_angle, record_states=record_states)
    elif name == 'Blackjack-v1':
        env = BlackjackMod(env=env)
    elif name == 'Taxi-v3':
        env = TaxiMod(env=env)
    elif name == 'CliffWalking-v0':
        env = CliffWalkMod(env=env)
    elif name == 'Pendulum-v1':
        env = PendulumMod(env=env)
    return env

       

class FrozenLakeMod(ogym.Wrapper):

    def __init__(self, env, prob=None, rew_struct=[0,0,1]):
        '''
        rew_struct = [step, fail, goal]
        '''
        super().__init__(env)
        self.num_steps=0
        self.rew_struct = rew_struct
        self.prob = prob
        self.status = None
        self.visited = set()
        if prob is not None:
            self.set_transitions()

    def reset(self, seed=None):
        state, info = super().reset(seed=seed)
        self.status = 'ongoing'
        self.visited = {state}
        return state, info
    
    def get_states(self):
        return range(self.observation_space.n)
        
    def step(self, action):
        self.num_steps += 1
        state, reward, terminated, truncated, info = self.env.step(action)
        
        if state not in self.visited:
            #reward += self.nsb
            self.visited.add(state)
        
        if terminated:
            if state == self.observation_space.n - 1:
                self.status = 'success'
            else:
                self.status = 'failed'
        
        return state, reward, terminated, truncated, info

    def set_T(self, T):
        p = self.prob
        q = round((1 - p) / 2, 4)
        # Non-terminal states have 3 possible transitions for each action
        if len(T) == 3:
            # Loop over 3 possible transitions
            for i in range(3):
                prob = p if i == 1 else q
                next_state = T[i][1]
                terminal = T[i][3]
                orig_reward = T[i][2]
                if terminal == False: reward = self.rew_struct[0]
                elif orig_reward == 0: 
                    reward = self.rew_struct[1]
                else: reward = self.rew_struct[2]

                T[i] = (prob, next_state, reward, terminal)
                
        return T

    def set_transitions(self):
    
        states = range(self.observation_space.n)
        actions =  range(4)
        
        for s in states:
            for a in actions:
                T = self.P[s][a]
                self.set_T(T)

    def display_policy(self, policy):
        dir_dict = {0:'←', 1:'↓', 2:'→', 3:'↑'}
        num_states = self.observation_space.n
        n = round(num_states**0.5)
        for s in range(num_states):
            if s % n == 0:
                print('+---'*n + '+')

            if s == 0: glyph = 'S'
            elif s == num_states-1: glyph = 'G'
            elif self.P[s][0][0][0] == 1.0: glyph = 'H'
            else:
                glyph = dir_dict[policy[s]]

            print(f'| {glyph} ', end='')
            if s % n == n-1:
                print('|')

            if s == num_states-1:
                print('+---'*n + '+\n')

    def display_values(self, V, digits=3, cell_width=50, cell_height=36):
        import numpy as np
        import pandas as pd
        from IPython.display import display, HTML
        num_states = self.observation_space.n
        n = round(num_states**0.5)
        V_list = [V.get(s,0) for s in range(num_states)]
        V_array = np.array(V_list).reshape((n,n)).round(digits)
        display(HTML('<b>State-Value Function</b>'))
        #print(V_array)
        
        html = '<table style="border-spacing: 0px; text-align: center">'
        for r in range(V_array.shape[0]):
            html += '<tr>'
            for c in range(V_array.shape[1]):
                state = r * V_array.shape[1] + c
                bgc = 'white'
                if state == 0:
                    bgc = 'LightBlue'
                elif state == V_array.size - 1:
                    bgc = 'LightGreen'
                elif len(self.env.P[state][0]) == 1:
                    bgc = '#bbbbbb'
                html += f'<td width={cell_width}, height={cell_height}, '
                html += f'style="border-style:solid; border-width:thin; background-color: {bgc}">'
                html += f'<b><center>{V_array[r,c]}</center></b></td>'
            html += '</td>'
        html += '</table>'

        
        display(HTML(html))

class CartPoleMod(ogym.Wrapper):
    
    def __init__(self, env, max_angle=1.57, num_bins=None, 
                 sqrt_trans=False, record_states=False):
        import numpy as np
        
        self.env = env
        self.max_angle = max_angle
        self.num_bins = num_bins
        
        self.coord_max = [2, 2.5, 0.21, 5]
        self.sqrt_trans = sqrt_trans
        
        if num_bins is not None:
            num_cuts = num_bins - 1
        
            m0, m1, m2, m3 = self.coord_max
            if self.sqrt_trans:
                m0, m1, m2, m3 = m0**0.5, m1**0.5, m2**0.5, m3**0.5
                
            self.bins0 = np.linspace(-m0, m0, num_cuts)
            self.bins1 = np.linspace(-m1, m1, num_cuts)
            self.bins2 = np.linspace(-m2, m2, num_cuts)
            self.bins3 = np.linspace(-m3, m3, num_cuts)
            
        
        self.record_states = record_states
        if record_states:
            import numpy as np
            self.state_visits = [[],[],[],[]]
            self.bin_visits = np.zeros(shape=(4,num_bins))
            
    
    def digitize_state(self, state):
        import numpy as np
        
        s0, s1, s2, s3 = state
        if self.sqrt_trans:
            s0, s1, s2, s3 = [abs(s)**0.5 if s > 0 else -abs(s)**0.5 for s in state]
                
        dig_state = np.zeros(4)
        dig_state[0] = np.digitize(s0, self.bins0)
        dig_state[1] = np.digitize(s1, self.bins1)
        dig_state[2] = np.digitize(s2, self.bins2)
        dig_state[3] = np.digitize(s3, self.bins3)
        dig_state = dig_state.astype(int)
        
        if self.record_states:
            self.log_state(dig_state, binned=True)
        return dig_state
    
    def log_state(self, state, binned=False):
        for i in range(4):
            if binned:
                self.bin_visits[i, state[i]] += 1
            else:
                self.state_visits[i].append(state[i])
        

    def reset(self, seed=None):
        state, info = super().reset(seed=seed)
        if self.record_states:
            self.log_state(state)
        if self.num_bins is not None:
            state = self.digitize_state(state)
        return state, info
    
    def step(self, action):
        '''
        Overwrite the step function. 
        '''
        import numpy as np
        import warnings
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            state, reward, terminated, truncated, info = self.env.step(action)
        # state: (pos, vel, angle, ang_vel)
        terminated = True if abs(state[2]) > self.max_angle else False
        terminated = True if abs(state[0]) > 2.4 else terminated
        
        reward = 0.0 if terminated else 1.0
        
        if self.record_states:
            self.log_state(state)
        if self.num_bins is not None:
            state = self.digitize_state(state)
            
        return state, reward, terminated, truncated, info

class CliffWalkMod(ogym.Wrapper):
    
    def __init__(self, env):
        super().__init__(env)
        self.status = None
        for s in self.P.keys():
            for a in self.P[s].keys():
                t0, t1, t2, t3 = self.P[s][a][0]
                if t2 == -100:
                    self.P[s][a][0] = (t0, t1, t2, True)
        for s in range(37, 48):
            for a in range(4):
                self.P[s][a][0] = (1.0, s, 0, True)

    def get_states(self):
        return range(self.observation_space.n)
    
    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        
        if reward == -100:
            terminated = True
        
        if terminated:
            if reward == -1:
                self.status = 'success'
            else:
                self.status = 'failed'
        
        return state, reward, terminated, truncated, info    

class BlackjackMod(ogym.Wrapper):
    
    def __init__(self, env):
        super().__init__(env)
        self.status = None
        
    def reset(self, seed=None):
        state, info = super().reset(seed=seed)
        self.status = 'ongoing'
        return state, info
    
    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        if terminated:
            if reward == 1:
                self.status = 'success'
            else:
                self.status = 'failed'
        return state, reward, terminated, truncated, info    
    
    def get_states(self):
        states = []
        for a in range(4, 31):
            for b in range(2, 12):
                for c in range(0, 2):
                    states.append((a, b, c))
        return states

class PendulumMod(ogym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def step(self, action):
        import numpy as np
        if not isinstance(action, np.ndarray) and not isinstance(action, list):
            action = [action]
        state, reward, terminated, truncated, info = self.env.step(action)
        return state, reward, terminated, truncated, info    
    
    
     
class TaxiMod(ogym.Wrapper):
    
    def __init__(self, env):
        super().__init__(env)
        self.status = 'ongoing'
    
    def get_states(self):
        return range(self.observation_space.n)

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        
        # Episodes terminates if and only if successful. 
        if terminated:
            self.status = 'success'
            
        return state, reward, terminated, truncated, info
    
    def reset(self, seed=None):
        state, info = super().reset(seed=seed)
        self.status = 'ongoing'
        return state, info

if __name__ == '__main__':
    
    env = make("CartPole-v1", render_mode='rgb_array')
    print(env.unwrapped.spec.disable_env_checker)
    
    env.reset()
    while True:
        _, _, term, _, _ = env.step(1)
        if term:
            break
        