import time
import numpy as np
import torch
from torch.optim.lr_scheduler import ExponentialLR

class A2C():

    def __init__(self, env, policy_net, value_net, gamma, 
                 dist='cat', vec_env=False):
        
        self.env = env
        self.policy_net = policy_net
        self.value_net = value_net
        self.gamma = gamma
        self.reset_history()
        self.ep_count = 0
        self.total_steps = 0
        self.return_history = []
        self.dist = dist
        self.vec_env = vec_env

    def reset_history(self):
        self.log_probs = []
        self.rewards = []
        self.states = []

    def select_action(self, state, return_log_prob=False):

        #-------------------------------------------
        # Convert state to Tensor (from np.array)
        #-------------------------------------------
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)


        #-------------------------------------------
        # Use policy network to select action
        #-------------------------------------------
        dist_params = self.policy_net(state)
        if self.dist == 'cat':
            prob_dist = torch.distributions.Categorical(logits=dist_params)
        elif self.dist == 'normal':
            prob_dist = torch.distributions.normal.Normal(
                loc=dist_params[0], scale=abs(dist_params[1]))
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
            torch.manual_seed(seed)
            state, info = self.env.reset(seed=seed)
            self.env.action_space.seed(seed)
        else:
            state, info = self.env.reset()
        self.reset_history()
        
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

            if terminated:
                break

    def train(self, episodes, actor_lr, critic_lr, lr_decay=1.0, max_steps=None, 
              ms_delta=0, stop_cond=None, updates=None, seed=None):

        if seed is not None:
            np.random.seed(seed)
            env_seeds = np.random.choice(episodes*10, episodes, replace=False)
            torch.manual_seed(seed)

        if max_steps is None:
                max_steps = float('inf')

        #--------------------------------------------
        # Create Adam optimizer
        #--------------------------------------------
        policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=actor_lr)
        value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=critic_lr)
        policy_scheduler = ExponentialLR(policy_optimizer, gamma=lr_decay)
        value_scheduler = ExponentialLR(value_optimizer, gamma=lr_decay)

        stop = False
        show_update = False
        for i in range(episodes):
            self.ep_count += 1
            t0 = time.time()

            #--------------------------------------------
            # Start a new episode
            #--------------------------------------------
            if seed is not None:
                state, info = self.env.reset(seed=int(env_seeds[i]))
            else:
                state, info = self.env.reset()

            self.env.action_space.seed(int(env_seeds[i]))
            self.reset_history()

            t = 0
            I = 1
            while t < max_steps:
                t += 1; self.total_steps += 1

                #--------------------------------------------
                # Select and apply action
                #--------------------------------------------
                action, log_prob = self.select_action(state, return_log_prob=True)

                #print(f'--{action}  {log_prob}')
                
                
                if self.vec_env == False:
                    new_state, reward, terminated, truncated, info = self.env.step(action)
                else:
                    new_state, reward, terminated, truncated, info = self.env.step([action])

                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                new_state_tensor = torch.tensor(new_state, dtype=torch.float32).unsqueeze(0)

                #--------------------------------------------
                # Estimate Advantage
                #--------------------------------------------
                Vs = self.value_net(state_tensor)
                Vs_next = self.value_net(new_state_tensor)
                #advantage = reward + (1 - terminated) * self.gamma - Vs
                advantage = reward + self.gamma * Vs_next - Vs
                
                #print(f'--{reward}')
                #print(f'--{self.gamma}')
                #print(f'--{Vs_next}')
                #print(f'--{Vs}')
                #print(f'--{advantage}')

                #--------------------------------------------
                # Calculate Losses
                #--------------------------------------------
                value_net_loss = advantage.pow(2)
                #value_net_loss = - advantage.detach() * Vs
                policy_net_loss = - advantage.detach() * log_prob * I

                #print(policy_net_loss)
                #print('\n\n')
                
                
                #--------------------------------------------
                # Gradient Descent
                #--------------------------------------------
                policy_optimizer.zero_grad()
                policy_net_loss.backward()
                policy_optimizer.step()

                value_optimizer.zero_grad()
                value_net_loss.backward()
                value_optimizer.step()

                #--------------------------------------------
                # Record episode information
                #--------------------------------------------
                self.rewards.append(reward)
                self.log_probs.append(log_prob)
                self.states.append(new_state)

                state = new_state
                I *= self.gamma

                if terminated:
                    break


            #--------------------------------------------
            # Calcualte episode return
            #--------------------------------------------
            T = len(self.rewards)
            returns = np.zeros(T)
            Gt = 0
            for t in reversed(range(T)):
                Gt = self.rewards[t] + self.gamma * Gt
            self.return_history.append(Gt)

            #--------------------------------------------
            # Calcualte average returns
            #--------------------------------------------
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
                out += f'Return: {ret_1:>7.2f},  '
                out += f'Mean_Return_10: {ret_10:>7.2f},  Mean_Return_100: {ret_100:>7.2f}'
                print(out)
                show_update = False
                
                positions = [s[0] for s in self.states]
                print(np.max(positions).round(2), np.max(self.rewards).round(2))

            #--------------------------------------------
            # Prepare for next episode
            #--------------------------------------------
            policy_scheduler.step()
            value_scheduler.step()
            max_steps += ms_delta