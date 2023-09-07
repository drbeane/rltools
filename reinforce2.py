import time
import numpy as np
import torch
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class ReinforceModel(nn.Module):
    
    def __init__(self,num_action,num_input):
        super(ReinforceModel,self).__init__()
        self.num_action = num_action
        self.num_input = num_input

        self.layer1 = nn.Linear(num_input,64)
        self.layer2 = nn.Linear(64,num_action)

    def train_forward(self,x):
        x = torch.tensor(x,dtype=torch.float32,device=DEVICE).unsqueeze(0)
        x = F.relu(self.layer1(x))
        actions = F.softmax(self.layer2(x), dim=-1)
        action = self.get_action(actions)
        log_prob_action = torch.log(actions.squeeze(0))[action]
        return action,log_prob_action

    def forward(self, x):
        with torch.no_grad():
            #print(type(x))
            #x = torch.tensor(x, dtype=torch.float32, device=DEVICE)#.unsqueeze(0)
            x = F.relu(self.layer1(x))
            actions = F.softmax(self.layer2(x), dim=-1)
            actions = actions.cpu().numpy()
            action = np.argmax(actions)
        return action.item()

    def get_action(self,a):
        ACTION_SPACE = range(self.num_action)
        return np.random.choice(ACTION_SPACE,p=a.squeeze(0).detach().cpu().numpy())




class REINFORCE():

    def __init__(self, env, policy_net, gamma):
        self.env = env
        self.model = policy_net
        self.gamma = gamma


    def train(self, episodes, alpha=0.0001, max_steps=1000, updates=None,
              save_path='save/', checkpoint=False):

        from rltools.utils import evaluate
        #--------------------------------------------
        # Create Adam optimizer
        #--------------------------------------------
        optimizer = torch.optim.Adam(self.model.parameters(), lr=alpha)

        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # Add scheduler for lr decay
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        all_rewards =[]
        best_score = -float('inf')
        for n in range(episodes):

            #--------------------------------------------
            # Generate Episode
            #--------------------------------------------
            state, info = self.env.reset()
            log_probs = []
            #a=[]
            rewards = []
            #d=[]
            #s=[]
            for t in range(max_steps):
                action, log_prob = self.model.train_forward(state)
                state, rew, done, trunc, info = self.env.step(action)
                log_probs.append(log_prob)
                rewards.append(rew)
                if done:
                    all_rewards.append(np.sum(rewards))

                    #if episode%100 ==0:
                    #    print(f"EPISODE {episode} SCORE: {np.sum(r)} roll{pd.Series(all_rewards).tail(100).mean()}")
                    #    torch.save(self.model.state_dict(), 'outputs/last_params_cloud.ckpt')
                    #    if pd.Series(all_rewards).tail(100).mean()>best_rolling:
                    #        best_rolling = pd.Series(all_rewards).tail(100).mean()
                    #        print("saving...")
                    #        torch.save(self.model.state_dict(), 'outputs/best_params_cloud.ckpt')
                    break


            #--------------------------------------------
            # Calculate Returns
            #--------------------------------------------
            returns = []
            for t in range(len(rewards)):
                Gt = 0
                pw = 0
                for r_ in rewards[t:]:
                    Gt = Gt + self.gamma**pw * r_
                    pw = pw + 1
                returns.append(Gt)
            returns = np.array(returns)

            returns_tensor = torch.tensor(returns,dtype=torch.float32,device=DEVICE)
            returns_tensor = (returns_tensor - torch.mean(returns_tensor))/ (torch.std(returns_tensor))
            log_probs = torch.stack(log_probs)

            loss = -log_probs * returns_tensor
            loss = loss.sum()

            #--------------------------------------------
            # Gradient Descent
            #--------------------------------------------
            '''
            Change to optimizer.zero_grad
            '''
            self.model.zero_grad()
            loss.backward()
            optimizer.step()


            if updates is not None and n == 0:
                col_names = 'Episode   Mean[Return]  SD[Return]  Mean[Length]  '
                col_names += 'SD[Length]'
                #col_names += '  Elapsed_Time'
                #if check_success: col_names += '  Success_Rate'
                print(col_names, '\n', '-' * len(col_names), sep='')

            if updates is not None and (n+1) % updates == 0:
                #print(f"EPISODE {n+1} SCORE: {np.sum(rewards)} last100: {pd.Series(all_rewards).tail(100).mean()}  last10: {pd.Series(all_rewards).tail(10).mean()}")

                #r1, r10, r100 = np.sum(rewards), pd.Series(all_rewards).tail(10).mean(), pd.Series(all_rewards).tail(100).mean()

                results = evaluate(self.env, self, gamma=self.gamma, episodes=10, max_steps=1000, show_report=False)

                #------------------------------------------------------------
                # Check for new best model
                #------------------------------------------------------------
                score = results['mean_return'] - results['stdev_return']
                save_msg = ''
                if score > best_score:
                    best_score = score
                    torch.save(self.model.state_dict(), save_path + 'best_model.pt')
                    save_msg = '(Saving new best model)'

                #------------------------------------------------------------
                # Checkpoint
                #------------------------------------------------------------
                if checkpoint:
                    torch.save(self.model.state_dict(), save_path + f'checkpoint_{n+1:07}.pt')


                out  = f'{n+1:<9}{results["mean_return"]:>13.4f}{results["stdev_return"]:>12.4f}'
                out += f'{results["mean_length"]:>14.4f}{results["stdev_length"]:>12.4f}'
                out += save_msg
                #out += f'{dt:>14.4f}  {save_msg}'

                print(out)


    def __init__(self, env, policy_net, gamma, dist='cat', vec_env=False):
        
        self.env = env
        self.gamma = gamma
        self.reset_history()
        self.ep_count = 0
        self.return_history = []
        self.dist = dist
        self.vec_env = vec_env
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.policy_net = policy_net.to(self.device)
    

    def reset_history(self):
        self.log_probs = []
        self.rewards = []

    def select_action(self, state):

        #-------------------------------------------
        # Convert state to Tensor (from np.array)
        #-------------------------------------------
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)

        # Send state to device (CPU/GPU)
        state = state.to(self.device)

        #-------------------------------------------
        # Use policy network to select action
        #-------------------------------------------
        
        with torch.no_grad():
            dist_params = self.policy_net(state)
            #dist_params = dist_params.detach().numpy()
            dist_params = dist_params.cpu().numpy()
            
    
        action = np.argmax(dist_params)
        
        return action


    def sample_action(self, state, return_log_prob=False):

        #-------------------------------------------
        # Convert state to Tensor (from np.array)
        #-------------------------------------------
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)

        # Send state to device (CPU/GPU)
        state = state.to(self.device)

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
            action, log_prob = self.sample_action(state, return_log_prob=True)
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
              eval_eps=100, save_path='saved_models/', checkpoint=False):
        
        import os
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


        #------------------------------------------------------------
        # Create objects for storing best results
        #------------------------------------------------------------
        best_score = -float('inf')
        os.makedirs(save_path, exist_ok=True)

        baseline = 0.0
        stop = False
        t0 = time.time()
        for n in range(episodes):
            self.ep_count += 1
            

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
            ret_tensor = ret_tensor.to(self.device)
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

            #------------------------------------------------------------
            # Report Results
            #------------------------------------------------------------
            
            if updates is not None and n == 0:
                col_names = 'Episode   Mean[Return]  SD[Return]  Mean[Length]  '
                col_names += 'SD[Length]  Elapsed_Time'
                #if check_success: col_names += '  Success_Rate'
                print(col_names, '\n', '-' * len(col_names), sep='')
        
            
            if updates is not None and (n+1) % updates == 0:
                                
                #------------------------------------------------------------
                # Evaluate Model
                #------------------------------------------------------------
                eval_seed = np.random.choice(10**6)
                stats = evaluate(
                    self.env, self, self.gamma, episodes=eval_eps,
                    max_steps=max_steps, seed=eval_seed, show_report=False
                )
                
                #------------------------------------------------------------
                # Check for new best model
                #------------------------------------------------------------
                score = stats['mean_return'] - stats['stdev_return']
                save_msg = ''
                if score > best_score:
                    best_score = score
                    torch.save(self.policy_net.state_dict(), save_path + 'best_model.pt')
                    save_msg = '(Saving new best model)'
                
                #------------------------------------------------------------
                # Checkpoint
                #------------------------------------------------------------
                if checkpoint:
                    torch.save(self.policy_net.state_dict(), save_path + f'checkpoint_{n+1:07}.pt')
                
                    
                #------------------------------------------------------------
                # Construct output
                #------------------------------------------------------------
                dt = time.time() - t0  # Get elapsed time for batch of episodes
                t0 = time.time()
                
                out  = f'{n+1:<9}{stats["mean_return"]:>13.4f}{stats["stdev_return"]:>12.4f}'
                out += f'{stats["mean_length"]:>14.4f}{stats["stdev_length"]:>12.4f}'
                out += f'{dt:>14.4f}  {save_msg}'
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

        
        self.policy_net.load_state_dict(torch.load(save_path + 'best_model.pt'))
        
        