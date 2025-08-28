class MCAgent:
    
    def __init__(self, env, gamma=1.0):
        import numpy as np
        
        #------------------------------------------------------------
        # Store environment and gamma
        #------------------------------------------------------------
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
        state = encode_state(state)
        if state in self.policy.keys():
            return self.policy[state]
        return self.unwrapped.env.action_space.sample()
   
   
    def prediction(self, policy, episodes, max_steps=None, alpha=None, alpha_decay=0.0, 
                   epsilon=0.0, epsilon_decay=0.0, seed=None, show_progress=False):
        '''
        Evaluates the policy currenty attached to the agent. 
        '''
        
        import numpy as np
        from tqdm.auto import tqdm
        from rltools.utils import DictAgent, generate_episode, set_seed, unset_seed
        
        agent=DictAgent(self.env, policy)
        
        #------------------------------------------------------------
        # Set the seed
        #------------------------------------------------------------
        np_state = set_seed(seed)
        
        #------------------------------------------------------------
        # Create dict to count number of visits to each state
        # Used only if alpha == None
        #------------------------------------------------------------
        visit_counts = {}
        
        #------------------------------------------------------------
        # Loop for specified number of episodes
        #------------------------------------------------------------
        if max_steps is None:
            max_steps = float('inf')
        rng = tqdm(range(episodes)) if show_progress else range(episodes)
        for n in rng:
            #------------------------------------------------------------
            # Generate episode using policy.
            # Obtain history and determine number of steps in episode
            # Generate episde needs a seed to be used with gym env 
            #------------------------------------------------------------
            gym_seed = np.random.choice(100*episodes)
            history = generate_episode(
                self.env, agent=agent, max_steps=max_steps, 
                epsilon=epsilon, seed=gym_seed, verbose=False)
            num_steps = len(history['actions'])
            
            #------------------------------------------------------------
            # Calcuate returns
            #------------------------------------------------------------
            returns = []
            Gt = 0
            for t in reversed(range(num_steps)):
                Gt = history['rewards'][t] + self.gamma * Gt
                returns.insert(0, Gt)
         
            #------------------------------------------------------------
            # Loop over episode, updating V est for each visited state
            #------------------------------------------------------------
            visited = set()
            for t in range(num_steps):

                #------------------------------------------------------------
                # Get state and exit if already visited
                #------------------------------------------------------------
                st = history['states'][t]
                if st in visited:
                    continue

                #------------------------------------------------------------
                # Update visit information
                #------------------------------------------------------------
                visited.add(st)
                visit_counts[st] = visit_counts.get(st, 0) + 1
                
                #------------------------------------------------------------
                # Perform update
                #------------------------------------------------------------
                Gt = returns[t]
                V_st = self.V.get(st, 0)
                
                if alpha is None: 
                    self.V[st] = V_st + (Gt - V_st) / visit_counts[st]
                else:
                    self.V[st] = V_st + alpha * (Gt - V_st)
            
            #------------------------------------------------------------
            # Decay alpha and epsilon
            #------------------------------------------------------------
            epsilon = epsilon * (1 - epsilon_decay)
            if alpha is not None:
                alpha = alpha * (1 - alpha_decay)
        
        #------------------------------------------------------------
        # Restore the global seed
        #------------------------------------------------------------
        unset_seed(np_state)


    def control(self, episodes, max_steps=None, alpha=None, alpha_decay=0.0, 
                epsilon=0.0, epsilon_decay=0.0, seed=None, show_progress=False, 
                updates=None, eval_eps=100, check_success=False, track_V=False,
                restore_best=True, return_best=False, alphas=None, epsilons=None, 
                exploring_starts=False, verbose=True):
        
        import numpy as np
        from tqdm.auto import tqdm
        from rltools.utils import generate_episode, evaluate, encode_state, decode_state
        from rltools.utils import set_seed, unset_seed
        
        
        #------------------------------------------------------------
        # Set the seed
        #------------------------------------------------------------
        np_state = set_seed(seed)

        #------------------------------------------------------------
        # Create dict to count number of visits to each state
        # Used only if alpha == None
        #------------------------------------------------------------
        visit_counts = {}
           
        #------------------------------------------------------------
        # Number of actions needed for creating entries in Q
        #------------------------------------------------------------
        num_actions = self.env.action_space.n

        #------------------------------------------------------------
        # Create objects for storing best results
        #------------------------------------------------------------
        best_score = -float('inf')
        best_pi = self.policy.copy()
        best_Q = self.Q.copy()
        best_V = self.V.copy()
        
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
            # Check if using exploring starts.
            #------------------------------------------------------------
            random_init_action = True if exploring_starts else False
            init_state = None
            if exploring_starts and len(self.Q) > 0:
                init_state = np.random.choice(list(self.Q.keys())) 
                init_state = decode_state(self.env, init_state)
            
            #------------------------------------------------------------
            # Generate episode using policy. 
            #------------------------------------------------------------
            episode_seed = np.random.choice(10**6)
            history = generate_episode(
                self.env, agent=self, max_steps=max_steps, 
                init_state=init_state, random_init_action=random_init_action,
                epsilon=epsilon, seed=episode_seed, verbose=False)
            
            #------------------------------------------------------------
            # Calcuate returns
            #------------------------------------------------------------
            num_steps = len(history['actions'])
            returns = []
            Gt = 0
            for t in reversed(range(num_steps)):
                Gt = history['rewards'][t] + self.gamma * Gt
                returns.insert(0, Gt)
            
            #------------------------------------------------------------
            # Loop over episode, updating Q estimate
            #------------------------------------------------------------
            visited = set()
            updated = set()
            for t in range(num_steps):
                
                #------------------------------------------------------------
                # Get state/action pair and exit if already visited
                #------------------------------------------------------------
                st, at = history['states'][t], history['actions'][t]
                
                # Encode state
                st = encode_state(st)
                
                if (st,at) in visited:
                    continue
                
                updated.add(st)

                #------------------------------------------------------------
                # Update visit information
                #------------------------------------------------------------
                visited.add((st,at))
                visit_counts[(st,at)] = visit_counts.get((st,at), 0) + 1
                
                #------------------------------------------------------------
                # Perform update
                #------------------------------------------------------------
                Gt = returns[t]
                if st not in self.Q.keys():
                    self.Q[st] = np.zeros(num_actions)
                Qsa = self.Q[st][at]
                
                if alpha is None: 
                    self.Q[st][at] = Qsa + (Gt - Qsa) / visit_counts[(st,at)]
                else:
                    self.Q[st][at] = Qsa + alpha * (Gt - Qsa) 
    
            #------------------------------------------------------------
            # Decay alpha and epsilon
            #------------------------------------------------------------
            epsilon = epsilon * (1 - epsilon_decay)
            if alpha is not None:
                alpha = alpha * (1 - alpha_decay)


            #------------------------------------------------------------
            # Update V and policy
            #------------------------------------------------------------
            #for s in self.Q.keys():
            for s in updated:
                self.V[s] = np.max(self.Q[s])
                self.policy[s] = np.argmax(self.Q[s])

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
                stats = evaluate(
                    self.env, self, self.gamma, episodes=eval_eps,
                    max_steps=max_steps, seed=episode_seed, 
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
                    best_V = self.V.copy()
                    save_msg = '(Saving new best model)'
                
                #------------------------------------------------------------
                # Construct output
                #------------------------------------------------------------
                out  = f'{n+1:<9}{stats["mean_return"]:>13.4f}{stats["stdev_return"]:>12.4f}'
                out += f'{stats["mean_length"]:>14.4f}{stats["stdev_length"]:>12.4f}'
                if check_success:
                    out += f'{stats["sr"]:>14.4f}'
                out += f'  {save_msg}'
                if verbose:
                    print(out)
                np.set_printoptions(suppress=True)
                
                
 
            
            continue # skip stuff below for now    
            if track_V:
                self.V_history.append(self.V.copy())
                
            # Check Success Rate
            if check_success is not None: 
                if i % check_success == 0:
                    sr, info = success_rate(
                        self.env, self.policy, n=100, max_steps=max_steps)
                    print(f'{i}: Success Rate: {sr:.4f}')
                    
        #------------------------------------------------------------
        # Unset the seed
        #------------------------------------------------------------
        unset_seed(np_state)
        
        #------------------------------------------------------------
        # Restore and/or return best model
        #------------------------------------------------------------
        if restore_best:
            self.policy = best_pi
            self.Q = best_Q
            self.V = best_V
        if return_best:
            return best_pi, best_Q
        
        
if __name__ == '__main__':
    import gymnasium as gym
    from gymnasium.envs.toy_text.frozen_lake import generate_random_map
    from rltools.envs import FrozenLakeMod
    from rltools.utils import generate_episode

    env = gym.make(
        'FrozenLake-v1',
        desc=generate_random_map(size=10, p=0.9, seed=1),
        max_episode_steps=1000,
        render_mode='ansi'
    )
    
    mod_env = FrozenLakeMod(env, rew=[-1,-100, 100])

    env.reset()
    print(env.render())
    
 