class DPAgent:
    
    def __init__(self, env, gamma=1.0, policy=None, seed=None):
        import numpy as np
        
        #--------------------------------------------------------
        # Store environment and gamma.
        #--------------------------------------------------------
        self.env = env
        self.gamma = gamma
    
        #--------------------------------------------------------
        # Get state information from environment.
        #--------------------------------------------------------
        self.states = self.env.unwrapped.get_states()
        self.num_states = env.observation_space.n
        
        #--------------------------------------------------------
        # Get environment information from environment.
        #--------------------------------------------------------
        self.num_actions = env.action_space.n
        self.actions = range(self.num_actions)
        
        #--------------------------------------------------------
        # Store policy or generate a random policy.
        #--------------------------------------------------------
        if policy is None:
            np_state = set_np_seed(seed)
            self.policy = {s:np.random.choice(self.actions) for s in self.states}
            restore_np_state(np_state)
        else:
            self.policy = policy
  
        #--------------------------------------------------------
        # Initialize value functions to zero
        #--------------------------------------------------------      
        self.Q = {s:np.zeros(self.num_actions) for s in self.states}
        self.V = {s:0 for s in self.states}
        
    
    def select_action(self, state):
        return self.policy[state]
        
    def policy_evaluation(
        self, threshold=1e-6, max_iter=None, verbose=True, return_n=False):
        '''
        Evaluates the policy currenty attached to the agent. 
        '''
        
        #--------------------------------------------------------
        # Loop for max_iter iteratons, or until convergence.
        #--------------------------------------------------------
        if max_iter is None:
            max_iter = float('inf')    
        n = 0
        while n < max_iter:
            n += 1
            
            #--------------------------------------------------------
            # Copy current V. 
            #--------------------------------------------------------
            V_old = self.V.copy()
   
            #--------------------------------------------------------
            # Loop over all states
            # Calculate new V(s) for each s
            #--------------------------------------------------------
            for s in self.states:
                
                # Select action and then get transitions P(s,a)
                a = self.policy[s]
                transitions = self.env.unwrapped.P[s][a]
                
                # Loop over all transitions
                Vs = 0
                for prob, s_, reward, done in transitions:
                    G_estimate = reward + (0 if done else self.gamma * V_old[s_])
                    Vs += prob * G_estimate
                self.V[s] = Vs
              
            #--------------------------------------------------------
            # Check for convergence
            #--------------------------------------------------------                  
            max_diff = max([abs(V_old[s] - self.V[s]) for s in self.states])
            if max_diff < threshold:
                break
            
        if verbose:
            print(f'Policy evaluation required {n} iterations to converge.')
    
        if return_n:
            return n
    
    def calculate_Q(self):
        '''
        Calculates estimate of Q from current estimate of V
        '''
        for s in self.states:
            for a in self.actions:
                trans = self.env.unwrapped.P[s][a]
                Qsa = 0
                for prob, s_, reward, done in trans:
                    G_est = reward + (0 if done else self.gamma * self.V[s_])
                    Qsa += prob * G_est
                self.Q[s][a] = Qsa
      
        
    def policy_improvement(self):
        '''
        Uses Q est to perform greedy policy improvement on cur policy. 
        Assumes that Q and V have been estimated for policy. 
        '''
        import numpy as np
        self.calculate_Q()
        self.policy = {s : np.argmax(self.Q[s]) for s in self.states}
       
        
    def policy_iteration(self, threshold=1e-6, verbose=True):
        
        #----------------------------------------------------------
        # Loop until convergence of policy
        #----------------------------------------------------------
        n = 0
        num_eval_steps = 0
        while True:
            n += 1
            
            # Store current policy
            old_policy = self.policy.copy()
            
            # Value current policy
            num_eval_steps += self.policy_evaluation(
                threshold=threshold, max_iter=None, return_n=True, verbose=False)
            
            # Policy Improvement
            self.policy_improvement()
            #self.calculate_Q()
            
            # Check for convergence
            if old_policy == self.policy:
                break
        
        #----------------------------------------------------------
        # Print report
        #----------------------------------------------------------
        if verbose:
            print(f'Policy iteration required {n} steps to converge.')
            print(f'{num_eval_steps} steps of policy evaluation were performed.')


    def value_iteration(self, threshold=1e-6, verbose=True):
        import numpy as np
        
        #----------------------------------------------------------
        # Loop until convergence of value function
        #----------------------------------------------------------
        n = 0
        while True:
            n += 1
            
            # Store current V estimate
            V_old = self.V.copy()
            
            # Get Q from current V
            self.calculate_Q()
            
            # Determine new V from Q
            self.V = {s : np.max(self.Q[s]) for s in self.states}

            # Check for convergence
            max_diff = max([abs(V_old[s] - self.V[s]) for s in self.states])
            if max_diff < threshold:
                break
            

        #----------------------------------------------------------
        # Determine Optimal Policy
        #----------------------------------------------------------
        self.calculate_Q()
        self.policy = {s : np.argmax(self.Q[s]) for s in self.states}

        #----------------------------------------------------------
        # Display report 
        #----------------------------------------------------------
        if verbose:
            print(f'Value iteration required {n} steps to converge.')

def set_np_seed(seed):
    import numpy as np
    if seed is None: return None
    np_state = np.random.get_state()
    np.random.seed(seed)
    return np_state

def restore_np_state(np_state):
    import numpy as np
    if np_state is None: return
    np.random.set_state(np_state)


if __name__ == '__main__':
    import gymnasium as gym
    from gymnasium.envs.toy_text.frozen_lake import generate_random_map
    from rltools.envs import * 
    from rltools.utils import success_rate

    base_4x4_env = gym.make('FrozenLake-v1', render_mode='ansi')    
    env = base_4x4_env
    
    env.reset()
    print(env.render())
    
    mc = DPAgent(env, gamma=0.99)
    
    states = range(env.observation_space.n)
    direct_actions = [2, 2, 1, 0, 1, 0, 1, 0, 2, 2, 1, 0, 0, 2, 2, 0]
    random_actions = [2, 0, 1, 3, 0, 0, 2, 0, 3, 1, 3, 0, 0, 2, 1, 0]
    careful_actions = [0, 3, 3, 3, 0, 0, 3, 0, 3, 1, 0, 0, 0, 2, 2, 0]
    direct_policy = {s:a for s, a in zip(states, direct_actions)}
    random_policy = {s:a for s, a in zip(states, random_actions)}
    careful_policy = {s:a for s, a in zip(states, careful_actions)}
        
    mc.policy_evaluation(careful_policy, threshold=1e-6)
    print()
    
    #frozen_lake_show_value(env, mc.V, digits=3)
    #print()
    
    sr, info = success_rate(env, policy=careful_policy, n=1000, max_steps=1000, seed=None)
    
    print(sr)
    print(info)