class TDAgent:
    
    def __init__(self, env, gamma=1.0):
        import numpy as np
        
        #--------------------------------------------------------
        # Store environment and gamma.
        #--------------------------------------------------------
        self.env = env
        self.gamma = gamma
    
        #------------------------------------------------------------
        # Create empty V, Q, and policy
        #------------------------------------------------------------
        self.V = {}
        self.Q = {}
        self.policy = {}
        
        #------------------------------------------------------------
        # Create dict for tracking V values
        #------------------------------------------------------------
        self.V_history = []
        
    
    def select_action(self, state):
        from rltools.utils import encode_state
        s = encode_state(state)
        if s in self.policy.keys():
            return self.policy[s]
        return self.env.action_space.sample()


    def q_learning(self, episodes, max_steps=None, alpha=0.1, alpha_decay=0.0,
                   epsilon=0.0, epsilon_decay=0.0, seed=None, show_progress=False,
                   updates=None, eval_eps=100, check_success=False, 
                   alphas=None, epsilons=None, exploring_starts=False, 
                   restore_best=True, return_best=False, verbose=True
                   ):

        import numpy as np
        from tqdm.auto import tqdm
        from rltools.utils import evaluate, set_seed, unset_seed, encode_state
        
        #------------------------------------------------------------
        # Set the seed
        #------------------------------------------------------------
        np_state = set_seed(seed)
        
        #------------------------------------------------------------
        # Number of actions needed for creating entries in Q
        #------------------------------------------------------------
        num_actions = self.env.unwrapped.action_space.n

        #------------------------------------------------------------
        # Create objects for storing best results
        #------------------------------------------------------------
        best_score = -float('inf')
        best_pi = self.policy.copy()
        best_Q = self.Q.copy()

        #------------------------------------------------------------
        # Loop for specified number of episodes
        #------------------------------------------------------------
        if max_steps is None:
            max_steps = float('inf')          
        rng = tqdm(range(episodes)) if show_progress else range(episodes)
        for n in rng:
            
            if alphas is not None: alpha = alphas[n]
            if epsilons is not None: epsilon = epsilons[n]
            
            #------------------------------------------------------------
            # Reset environment and generate episode
            #------------------------------------------------------------
            gym_seed = int(np.random.choice(10**6))
            state, info = self.env.reset(seed=gym_seed)
            self.env.action_space.seed(gym_seed)
            
            if exploring_starts and len(self.Q) > 0:
                state = np.random.choice(list(self.Q.keys())) 
                state = int(state)
                self.env.unwrapped.s = state
                
            
            
            #------------------------------------------------------------
            # Loop until episode terminates or max_steps completed
            #------------------------------------------------------------
            t = 0
            while t < max_steps:
                t += 1
                
                #------------------------------------------------------------
                # Select action using epsilon greedy strategy
                #------------------------------------------------------------
                roll = np.random.uniform(0,1)
                if roll < epsilon:
                    action = self.env.action_space.sample()
                else: 
                    action = self.select_action(state)
                    
                #------------------------------------------------------------
                # Apply action
                #------------------------------------------------------------
                next_state, reward, terminated, trunc, info = self.env.step(action)
            
                #------------------------------------------------------------
                # Update Q
                #------------------------------------------------------------
                s = encode_state(state)
                s_ =  encode_state(next_state)
                if s not in self.Q.keys(): self.Q[s] = np.zeros(num_actions)
                if s_ not in self.Q.keys(): self.Q[s_] = np.zeros(num_actions)
                Qsa = self.Q[s][action]
                maxQ = np.max(self.Q[s_]) 
                
                td_target = reward + (0 if terminated else self.gamma * maxQ)
                td_error = td_target - Qsa
                
                self.Q[s][action] = Qsa + alpha * td_error
                
                #------------------------------------------------------------
                # Update policy
                #------------------------------------------------------------
                self.policy[s] = np.argmax(self.Q[s])
                
                #------------------------------------------------------------
                # Prepare for next step
                #------------------------------------------------------------
                state = next_state
                
                if terminated:
                    break
                
            #------------------------------------------------------------
            # Decay alpha and epsilon
            #------------------------------------------------------------ 
            epsilon = epsilon * (1 - epsilon_decay)
            alpha = alpha * (1 - alpha_decay)
            

            #------------------------------------------------------------
            # Report Results
            #------------------------------------------------------------
            if updates is not None and n == 0:
                col_names = 'Episode   Mean[Return]  SD[Return]  Mean[Length]  SD[Length]'
                if check_success: col_names += '  Success_Rate'
                if verbose:
                    print(col_names, '\n', '-' * len(col_names), sep='')
        
            
            if updates is not None and (n+1) % updates == 0:
                #------------------------------------------------------------
                # Evaluate Model
                #------------------------------------------------------------
                eval_seed = np.random.choice(10**6)
                stats = evaluate(
                    self.env, self, self.gamma, episodes=eval_eps,
                    max_steps=max_steps, seed=eval_seed, 
                    check_success=check_success, show_report=False
                )
                
                #------------------------------------------------------------
                # Check for new best model
                #------------------------------------------------------------
                save_msg = ''
                score = stats['mean_return'] - stats['stdev_return']
                if score > best_score:
                    best_score = score
                    best_pi = self.policy.copy()
                    best_Q = self.Q.copy()
                    save_msg = '(Saving new best model)'
                
                #------------------------------------------------------------
                # Construct output
                #------------------------------------------------------------
                out  = f'{n+1:<9}{stats["mean_return"]:>13.4f}{stats["stdev_return"]:>12.4f}'
                out += f'{stats["mean_length"]:>14.4f}{stats["stdev_length"]:>12.4f}'
                if check_success:
                    out += f'{stats["sr"]:>14.4f}'
                out += f'  {save_msg}'
                if verbose: print(out)
            
                

        #------------------------------------------------------------
        # Unset the seed
        #------------------------------------------------------------
        unset_seed(np_state)
        
        if restore_best:
            self.policy = best_pi
            self.Q = best_Q
        if return_best:
            return best_pi, best_Q
        
        for s in self.Q.keys():
            self.V[s] = np.max(self.Q[s])