import time
import numpy as np
import torch
from torch.optim.lr_scheduler import ExponentialLR

class REINFORCE():

    def __init__(self, env, policy_net, gamma, dist='cat', vec_env=False):
        self.env = env
        self.policy_net = policy_net
        self.gamma = gamma
        self.reset_history()
        self.ep_count = 0
        self.return_history = []
        self.dist = dist
        self.vec_env = vec_env

    def reset_history(self):
        self.log_probs = []
        self.rewards = []

    def select_action(self, state, return_log_prob=False):

        #-------------------------------------------
        # Convert state to Tensor (from np.array)
        #-------------------------------------------
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)

        # Send state to device (CPU/GPU)
        #state = state.to(device)

        #-------------------------------------------
        # Use policy network to select action
        #-------------------------------------------
        dist_params = self.policy_net(state)
        dist_params = dist_params.squeeze()
        if self.dist == 'cat':
            prob_dist = torch.distributions.Categorical(logits=dist_params)
        elif self.dist == 'normal':
            prob_dist = torch.distributions.normal.Normal(loc=dist_params[0], scale=abs(dist_params[1]))
        action = prob_dist.sample(sample_shape=(1,))

        #--------------------------------------------
        # Return action and (optinally) log_prob
        #--------------------------------------------
        if return_log_prob:
            log_prob = prob_dist.log_prob(action) 
            return action.item(), log_prob
        else:
            return action.item()


    def generate_episode(self, max_steps=None, seed=None):

        if max_steps is None:
            max_steps = float('inf')

        t = 0
        if seed is not None:
            seed = int(seed)
            state, info = self.env.reset(seed=seed)
            self.env.action_space.seed(seed)
        else:
            state, info = self.env.reset()
        self.reset_history()
        
        history = {'states':[state], 'actions':[], 'rewards':[0], 'log_probs':[]}
        
        while t < max_steps:
            t += 1

            #--------------------------------------------
            # Select and apply action
            #--------------------------------------------
            action, log_prob = self.select_action(state, return_log_prob=True)
            if self.vec_env == False:
                state, reward, terminated, truncated, info = self.env.step(action)
            else:
                state, reward, terminated, truncated, info = self.env.step([action])
        
            #--------------------------------------------
            # Record episode information
            #--------------------------------------------
            self.rewards.append(reward)
            self.log_probs.append(log_prob)
            history['states'].append(state)
            history['actions'].append(action)
            history['rewards'].append(reward)
            history['log_probs'].append(log_prob.item())

            if terminated:
                break
            
        return history

    def train(self, episodes, alpha, alpha_decay=1.0, baseline_alpha=None,
              max_steps=None, ms_delta=0, stop_cond=None, updates=None, seed=None,
              eval_eps=100):
        
        from rltools.utils import evaluate
        
        if seed is not None:
            np.random.seed(seed)
            env_seeds = np.random.choice(episodes*10, episodes, replace=False)
            torch.manual_seed(seed)

        #--------------------------------------------
        # Create Adam optimizer
        #--------------------------------------------
        optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=alpha)
        policy_scheduler = ExponentialLR(optimizer, gamma=alpha_decay)

        baseline = 0.0
        stop = False
        show_update = False
        for n in range(episodes):
            self.ep_count += 1
            t0 = time.time()

            #--------------------------------------------
            # Create episode and calculate returns
            #--------------------------------------------
            history = self.generate_episode(max_steps=max_steps,  seed=env_seeds[n])

            T = len(self.rewards)
            returns = np.zeros(T)
            Gt = 0
            for t in reversed(range(T)):
                #Gt = self.rewards[t] + self.gamma * Gt
                Gt = self.gamma**(T - t - 1) * self.rewards[t] + Gt
                returns[t] = Gt


            #--------------------------------------------
            # Calculate Loss
            #--------------------------------------------
            ret_tensor = torch.FloatTensor(returns)
            if baseline_alpha is not None:
                baseline += baseline_alpha * (ret_tensor.mean() - baseline)
                ret_tensor -= baseline

            #print(self.log_probs)
            #ret_tensor = ret_tensor.to(device)
            log_probs = torch.cat(self.log_probs)
            loss = - torch.sum(log_probs * ret_tensor.detach())

            #--------------------------------------------
            # Gradient Descent
            #--------------------------------------------
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #--------------------------------------------
            # Calcualte average returns
            #--------------------------------------------
            self.return_history.append(returns[0])
            ret_1 = self.return_history[-1]
            ret_10 = np.mean(self.return_history[-10:])
            ret_100 = np.mean(self.return_history[-100:])

            #--------------------------------------------
            # Check for Early Stopping
            #--------------------------------------------
            if stop_cond is not None:
                if np.all(np.array(self.return_history[-stop_cond[1]:]) >= stop_cond[0]):
                    stop = True
                    show_update = True

            #--------------------------------------------
            # Output
            #--------------------------------------------
            dt = time.time() - t0
            if updates is not None:
                if (self.ep_count) % updates == 0:
                    show_update = True
            
            if show_update:
                eplen = len(str(episodes))
                out = f'Episode {self.ep_count:<{eplen}} -- Elapsed_Time: {dt:>5.2f}s,  '
                out += f'Loss:{loss.item():>12.4f},  Return: {ret_1:>7.2f},  '
                out += f'Mean_Return_10: {ret_10:>7.2f},  Mean_Return_100: {ret_100:>7.2f}'
                #print(out)
                show_update = False
                
                #positions = [s[0] for s in history['states']]
                #print(np.max(positions).round(2), np.max(self.rewards).round(2))

            #------------------------------------------------------------
            # Report Results
            #------------------------------------------------------------
            if updates is not None and n == 0:
                col_names = 'Episode   Mean[Return]  SD[Return]  Mean[Length]  SD[Length]'
                #if check_success: col_names += '  Success_Rate'
                print(col_names, '\n', '-' * len(col_names), sep='')
        
            
            if updates is not None and (n+1) % updates == 0:
                eval_seed = np.random.choice(10**6)
                stats = evaluate(self.env, self, self.gamma, episodes=eval_eps,
                                 max_steps=max_steps, seed=eval_seed)
                out  = f'{n+1:<9}{stats["mean_return"]:>13.4f}{stats["stdev_return"]:>12.4f}'
                out += f'{stats["mean_length"]:>14.4f}{stats["stdev_length"]:>12.4f}'
                #if check_success:
                #    out += f'{stats["sr"]:>14.4f}'
                
                #if verbose: print(out)
                print(out)


            #--------------------------------------------
            # Prepare for next episode
            #-------------------------------------------- 
            policy_scheduler.step() 
            max_steps += ms_delta
            if stop:
                break
            # End episode    

            