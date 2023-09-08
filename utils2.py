def set_seed(seed):
    import numpy as np
    if seed is None:
        return None
    np_state = np.random.get_state()
    np.random.seed(seed)
    return np_state

def unset_seed(np_state):
    import numpy as np
    if np_state is None:
        return None
    np.random.set_state(np_state)
    

def render_mp4(videopath):
    from base64 import b64encode
    mp4 = open(videopath, 'rb').read()
    base64_encoded_mp4 = b64encode(mp4).decode()
    return f'<video width=400 controls><source src="data:video/mp4;' \
           f'base64,{base64_encoded_mp4}" type="video/mp4"></video>'


class SB3Agent:
    def __init__(self, model, seed=None, deterministic=True, normalizer=None): 
        self.model = model
        self.seed = seed
        self.deterministic = deterministic
        self.normalizer = normalizer
    def select_action(self, state):
        if self.seed is not None:
            from stable_baselines3.common.utils import set_random_seed
            set_random_seed(self.seed)
        if self.normalizer is not None:
            state = self.normalizer(state)
        a = self.model.predict(state, deterministic=self.deterministic)[0]
        return a

class RandomAgent:
    def __init__(self, env, seed=None):
        self.env = env
        self.seed = seed
        if seed is not None:
            self.env.action_space.seed(seed)
    def select_action(self, state):
        return self.env.action_space.sample()

class DictAgent:
    def __init__(self, env, policy):
        self.env = env
        self.policy = policy
    def select_action(self, state):
        if state in self.policy.keys():
            return self.policy[state]
        return self.env.action_space.sample() 

class FnAgent:
    def __init__(self, policy):
        self.policy = policy
    def select_action(self, state):
        return self.policy(state)


def record_episode(
    env, agent, max_steps=250, seed=None, fps=30, vec_env=False, 
    folder='', filename='temp', display_video=True):

    import os
    import shutil
    import torch
    import time
    from IPython.display import display, HTML
    from gymnasium.wrappers import RecordVideo

    #--------------------------------------------------------------------------------------
    # The video will be created in a temp folder. 
    # We can't 100% control the name of this video, but we will move it later. 
    # Also, gymnasium doesn't like it if you write to an existing folder. 
    #--------------------------------------------------------------------------------------
    #if os.path.exists(f'{folder}/temp_videos'): 
    #    shutil.rmtree(f'{folder}/temp_videos')
    
    #--------------------------------------------------------------------------------------
    # Wrap the environment in a RecordVideo
    #--------------------------------------------------------------------------------------    
    vid_env = RecordVideo(env, video_folder=f'{folder}/temp_videos', 
                          name_prefix=filename, disable_logger=True)
    vid_env.metadata['render_fps'] = fps

    #--------------------------------------------------------------------------------------
    # Set seeds, if needed. Note: Seed not yet passed to agent. 
    #--------------------------------------------------------------------------------------    
    if seed is None:
        state, info = vid_env.reset()
    else:
        state, info = vid_env.reset(seed=seed)
        vid_env.action_space.seed(seed)
        torch.manual_seed(seed)

    #--------------------------------------------------------------------------------------
    # Generate an episode
    #--------------------------------------------------------------------------------------    
    for i in range(max_steps):
        action = agent.select_action(state)
        action = [action] if vec_env else action
        state, reward, terminated, truncated, info = vid_env.step(action)
        if terminated:
            break

    #--------------------------------------------------------------------------------------
    # End Recording
    #--------------------------------------------------------------------------------------    
    print(i + 1, 'steps completed.')
    vid_env.close()

    #--------------------------------------------------------------------------------------
    # Generate an episode
    #--------------------------------------------------------------------------------------    
    os.makedirs(folder, exist_ok=True)
    #if not os.path.exists(folder): os.mkdir('videos')
    #shutil.move(f'/content/temp_videos/{filename}-episode-0.mp4', f'/content/videos/{filename}.mp4')
    shutil.move(f'{folder}/temp_videos/{filename}-episode-0.mp4', f'{folder}/{filename}.mp4')
    #shutil.rmtree('temp_videos')
    time.sleep(3)
    #shutil.rmtree(f'{folder}/temp_videos')

    if display_video:
        time.sleep(1)
        html = render_mp4(f'{folder}/{filename}.mp4')
        display(HTML(html))
        print('uh..')


def create_gif(
    env, agent, actions=None, max_steps=1000, seed=None, fps=None, vec_env=False,
    folder='', filename='', processor=None, display_gif=True
    ):
    
    import os
    import torch
    import imageio
    from IPython.display import Image, display, HTML

    if actions is None:
        if max_steps is None: max_steps = float('inf')
    else:
        max_steps = len(actions)
    if processor is None: processor = lambda x : x

    #-----------------------------------------------------------------------
    # Determine FPS if not provided
    #-----------------------------------------------------------------------
    if fps is None:
        fps_lu = {'Taxi-v3':2, 'CliffWalking-v0':3, 'FrozenLake-v1':4, 'CartPole-v1':40}
        fps = fps_lu.get(env.spec.id, 20)

            
    #-----------------------------------------------------------------------
    # Set seeds
    #-----------------------------------------------------------------------    
    if vec_env:
        state = env.reset()
    else:
        np_state = set_seed(seed)
        state, info = env.reset(seed=seed)
        env.action_space.seed(seed)
    #else:
    #    state, info = env.reset()
        
    #if seed is not None:
    #    torch.manual_seed(seed)


    frames = []
    frames.append(processor(env.render()) )

    t = 0
    total_reward = 0
    while t < max_steps:
        t += 1
        if actions is None:
            action = agent.select_action(state)
        else: 
            action = actions[t-1]
        
        
        #action = [action] if vec_env else action
        if vec_env:
            state, reward, terminated, info = env.step(action)
        else:          
            state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        frames.append(processor(env.render()))
        if terminated:
            break

    for i in range(2 * fps):
        frames.append(processor(env.render()))

    print(f'{t} steps completed.')
    print(f'Cumulative reward: {total_reward}')
    try:
        print(env.status.title())
    except:
        pass
    os.makedirs(folder, exist_ok=True)
    imageio.mimsave(f'{folder}/{filename}.gif', frames, format='GIF', duration=1000/fps, loop=0)   

    if display_gif:
        with open(f'{folder}/{filename}.gif','rb') as f:
            display(Image(data=f.read(), format='png'))

    #------------------------------------------------------------
    # Unset the seed
    #------------------------------------------------------------
    if vec_env == False:
        unset_seed(np_state)


def create_gif_old(
    env, agent=None, actions=None, episodes=1, max_steps=250, seed=None, fps=None, vec_env=False,
    folder='', filename='', processor=None, display_gif=True
    ):
    
    # Intended to support multiple episodes. 
    # There is some strangeness with how the seed operates, though.
    # Not that env is reset w/o seed at start of each episode
    
    import os
    import torch
    import imageio
    from IPython.display import Image, display, HTML

    if actions is None:
        if max_steps is None: max_steps = float('inf')
    else:
        max_steps = len(actions)
    if processor is None: processor = lambda x : x

    #-----------------------------------------------------------------------
    # Determine FPS if not provided
    #-----------------------------------------------------------------------
    if fps is None:
        if env.spec.id == 'FrozenLake-v1':
            fps = 4
        else:
            fps = 20
            
    #-----------------------------------------------------------------------
    # Set seeds, if needed. Note: Seed not yet passed to agent. 
    #-----------------------------------------------------------------------    
    if seed is None:
        state, info = env.reset()
    else:
        state, info = env.reset(seed=seed)
        env.action_space.seed(seed)
        torch.manual_seed(seed)

    frames = []

    for i in range(episodes):
        obs, info = env.reset()

        frames.append(processor(env.render()) )

        t = 0
        while t < max_steps:
            t += 1
            if actions is None:
                action = agent.select_action(state)
            else: 
                action = actions[t-1]
            action = [action] if vec_env else action
            state, reward, terminated, truncated, info = env.step(action)
            frames.append(processor(env.render()))
            if terminated:
                break

        for i in range(1 * fps):
            frames.append(processor(env.render()))

    print(f'{t} steps completed.')
    try:
        print(env.status.title())
    except:
        pass
    os.makedirs(folder, exist_ok=True)
    imageio.mimsave(f'{folder}/{filename}.gif', frames, format='GIF', fps=fps)   

    if display_gif:
        with open(f'{folder}/{filename}.gif','rb') as f:
            display(Image(data=f.read(), format='png'))



def generate_episode(
    env, agent,  max_steps=None, init_state=None, random_init_action=False, 
    epsilon=0.0, seed=None, verbose=False, n_frames=None):
    
    from stable_baselines3.common.vec_env import VecFrameStack
    from stable_baselines3.common.atari_wrappers import AtariWrapper   
    from stable_baselines3.common.vec_env import DummyVecEnv 
    
    
    # For Atari environments, env will be the BASE environment. 
    # Vector wrapping is performed below. 
    
    # n_frames used only for Atari environments currently
    # could consider using for highway environments
    
    import numpy as np
    import time
    
    #--------------------------------------------------------
    # Set seeds
    #--------------------------------------------------------
    np_state = set_seed(seed)
    
    #--------------------------------------------------------
    # Reset Environment
    #--------------------------------------------------------
    if seed is None:
        state, info = env.reset()
    else:
        state, info = env.reset(seed=int(seed))
        env.action_space.seed(int(seed))

    #--------------------------------------------------------
    # Create Vector Environment for Atari
    #--------------------------------------------------------
    if n_frames is not None:          
        vec_env = AtariWrapper(env, frame_skip=0)
        vec_env = DummyVecEnv([lambda: vec_env])
        vec_env = VecFrameStack(vec_env, n_stack=n_frames)
        state = vec_env.reset()
        
    #--------------------------------------------------------
    # Set initial state (Used for exploring starts)
    #--------------------------------------------------------
    if init_state is not None:
        env.unwrapped.s = init_state
        state = init_state
    
    # In case init state was not specified, store it for later.
    init_state = state

    #--------------------------------------------------------
    # Lists to store information
    #--------------------------------------------------------
    s_list, a_list, r_list, d_list, c_list, i_list =\
        [], [], [], [], [], []
    
    #--------------------------------------------------------
    # Loop for max steps episodes
    #--------------------------------------------------------
    t = 0
    if max_steps is None:
        max_steps = float('inf')
    while t < max_steps:
        t += 1
        
        #--------------------------------------------------------
        # Determine if action should be selected at random. 
        # True if using exp starts and t==1, or if roll < epsilon
        # For sake of efficiiency, don't roll unless needed.
        #--------------------------------------------------------
        random_action = False
        if random_init_action and t == 1:
            random_action = True
        if epsilon > 0:
            roll = np.random.uniform(0,1)
            if roll < epsilon:
                random_action = True
        
        #--------------------------------------------------------
        # Select action.
        #--------------------------------------------------------
        if random_action:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state)
        
        #--------------------------------------------------------
        # Apply action
        #--------------------------------------------------------
        if n_frames is not None:
            state, reward, done, info = vec_env.step([action])
            truncated = False
        else:
            state, reward, done, truncated, info = env.step(action)
        
        t0 = time.time()
        a_list.append(action)
        s_list.append(state)
        r_list.append(reward)
        d_list.append(done)
        c_list.append(truncated)
        i_list.append(info)
        
        if done:
            break

    if verbose:
        ss_list = [state_str(env, s) for s in s_list] # format_states
        i_list = [str(i) for i in i_list]
        d_list = [str(t) for t in d_list]
        
        wn = 6
        wa = max([len(str(a)) for a in a_list]) + 2
        wa = max(wa, 8)
        ws = max([len(s) for s in ss_list]) + 2
        ws = max(ws, 11)
        wr = 8
        wt = 12
        wi = max([len(i) for i in i_list])
        wi = max(wi, 4)
        w = wn + wa + ws + wr + wt + wi   
        
        #--------------------------------------------------------
        # Output Header
        #--------------------------------------------------------   
    
        print(f'{"step":<{wn}}{"action":<{wa}}{"new_state":<{ws}}' +
              f'{"reward":<{wr}}{"terminated":<{wt}}{"info":<{wi}}')
        print('-'*w)
        
        for n, (a, s, r, d, i) in enumerate(
            zip(a_list, ss_list, r_list, d_list, i_list)):
     
            print(f'{n:<{wn}}{a:<{wa}}{s:<{ws}}' + 
                  f'{r:<{wr}}{d:<{wt}}{i:<{wi}}')
        print('-'*w)
        print(t, 'steps completed.')
        
    #--------------------------------------------------------
    # Create history dictionary 
    # Note: States will be one longer than the others.
    #--------------------------------------------------------
    history = {
        'states' : [init_state] + s_list,
        'actions' : a_list,
        'rewards' : r_list 
    } 
    
    unset_seed(np_state)
       
    return history
    
    
def state_str(env, state):
    '''
    Represents states as strings for the sake of displaying them in reports. 
    '''
    import numpy as np
    np.set_printoptions(suppress=True)
    if env.spec.id == 'CartPole-v1':
        ss = str(state.round(3))
    else:
        ss = str(state)
    return ss

def encode_state(state):
    '''
    Encodes states for use as keys in dictionaries. 
    -- Lists --> Tuples
    -- Arrays --> Bytes
    -- Integers/Floats/Strings --> Unaffected
    '''
    import numpy as np
    if type(state) == list: return tuple(state)
    if isinstance(state, np.ndarray): return state.tobytes()
    return state

def decode_state(env, state):
    import numpy as np
    if env.spec.id == 'FrozenLake-v1':
        return int(state)
    

def evaluate(env, agent, gamma, episodes, max_steps=1000, seed=None, 
             check_success=False, show_report=True, vec_env=False):
    import numpy as np
    
    np_state = set_seed(seed)
    
    returns = []
    lengths = []
    success = []
    num_success = 0
    num_failure = 0
    len_success = 0
    len_failure = 0
    
    for n in range(episodes):
        ep_seed = np.random.choice(10**6)
        history = generate_episode(
            env=env, agent=agent, max_steps=max_steps, 
            epsilon=0.0, seed=ep_seed, verbose=False, vec_env=vec_env
        )
        
        #------------------------------------------------------------
        # Calcuate return at initial state
        #------------------------------------------------------------
        num_steps = len(history['actions'])
        G0 = 0
        for t in reversed(range(num_steps)):
            G0 = history['rewards'][t] + gamma * G0
        
        returns.append(G0)
        lengths.append(num_steps)
        
        
        #------------------------------------------------------------
        # Update success rate info, if requested
        #------------------------------------------------------------
        if check_success:
            success.append(True if env.status == 'success' else False)
            if env.status == 'success':
                num_success += 1
                len_success += len(history['actions'])
            else:
                num_failure += 1
                len_failure += len(history['actions'])
    
    #------------------------------------------------------------
    # Build stats report
    #------------------------------------------------------------
    
    stats = {
        'mean_return' : np.mean(returns),
        'stdev_return' : np.std(returns),
        'mean_length' : np.mean(lengths),
        'stdev_length' : np.std(lengths)
    }
    
    if check_success:
        stats.update({
            #'sr' : num_success / episodes,
            #'avg_len_s' : None if num_success == 0 else len_success / num_success,
            #'avg_len_f' : None if num_failure == 0 else len_failure / num_failure
            'sr' : np.mean(success),
            'avg_len_s' : None if num_success == 0 else len_success / num_success,
            'avg_len_f' : None if num_failure == 0 else len_failure / num_failure
         })

    if show_report:
        print(f'Mean Return:    {round(stats["mean_return"], 4)}')
        print(f'StdDev Return:  {round(stats["stdev_return"], 4)}')
        print(f'Mean Length:    {round(stats["mean_length"], 4)}')
        print(f'StdDev Length:  {round(stats["stdev_length"], 4)}')
        
        if check_success:
            print(f'Success Rate:   {round(stats["sr"], 4)}')

    unset_seed(np_state)

    return stats



def sb3_training_curves(eval_env, start=1, fs=[10,2], n=100):
    import matplotlib.pyplot as plt
    import numpy as np
    
    def moving_average(a, n=100):
        padded = np.concatenate([np.zeros(n-1), a])
        conv = np.convolve(padded, np.ones(n), 'valid')
        conv[:n] = conv[:n] / np.arange(1,n+1)
        conv[n:] = conv[n:] / n
        return conv

    rewards = eval_env.get_episode_rewards()
    #timesteps = eval_env.get_total_steps()
    lengths = eval_env.get_episode_lengths()
    num_episodes = len(rewards)
    episodes = range(1, num_episodes+1)
    ma_rewards = moving_average(rewards, n=n)
    ma_lengths = moving_average(lengths, n=n)

    plt.figure(figsize=fs)
    plt.scatter(episodes[start-1:], rewards[start-1:], s=1, alpha=0.9, zorder=2, c='darkgray')
    plt.plot(episodes[start-1:], ma_rewards[start-1:], zorder=3)
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title('Returns for Evaluation Environment During Training')
    plt.grid()
    plt.show()

    plt.figure(figsize=fs)
    plt.scatter(episodes[start-1:], lengths[start-1:], s=1, alpha=0.6, zorder=2, c='darkgray')
    plt.plot(episodes[start-1:], ma_lengths[start-1:], zorder=3)
    plt.xlabel('Episode')
    plt.ylabel('Episode Length')
    plt.title('Episode Lengths for Evaluation Environment During Training')
    plt.grid()
    plt.show()

def sb3_evaluation_curves(path, start=1, fs=[10,2], ylim=None):
    import numpy as np
    import matplotlib.pyplot as plt
    
    data = np.load(path + 'evaluations.npz')
    _ = data['timesteps']

    lengths = data['ep_lengths']
    scores = data['results']

    scores_flat = scores.reshape(-1)
    lengths_flat = lengths.reshape(-1)

    mean_lengths = lengths.mean(axis=1)
    mean_scores = scores.mean(axis=1)
    best = mean_scores.max()
    best_idx = mean_scores.argmax()

    r, c = scores.shape
    repeated_timesteps = np.repeat(np.arange(start, r + 1), c)
    
    plt.figure(figsize=fs)
    plt.scatter(
        repeated_timesteps,
        scores_flat[c*(start-1):], alpha=0.6, s=2, c='darkgray', zorder=2
    )
    plt.scatter(best_idx + 1, best, c='gold', alpha=0.99, zorder=3, s=80, label='Best Mean Return')
    plt.plot(np.arange(start, r+1), mean_scores[start-1:], zorder=4)
    plt.legend()
    if ylim is not None:
        plt.ylim(ylim)
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title('Returns for Evaluation Environment During Training')
    plt.grid()
    plt.show()
    
    plt.figure(figsize=fs)
    plt.scatter(
        repeated_timesteps,
        lengths_flat[c*(start-1):], alpha=0.6, s=2, c='darkgray', zorder=2
    )
    plt.plot(np.arange(start, r+1), mean_lengths[start-1:], zorder=4)
    plt.xlabel('Episode')
    plt.ylabel('Episode Length')
    plt.title('Episode Lengths for Evaluation Environment During Training')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    import os
    import sys
    sys.path.append(os.getcwd())

    import rltools.gym as gym
    env = gym.make('FrozenLake-v1', render_mode='rgb_array')
    random_agent = RandomAgent(env)
    history = generate_episode(env=env, agent=random_agent, max_steps=100, seed=42, verbose=True)
    
    eval_results = evaluate(env, random_agent, gamma=1.0, episodes=10, check_success=True, max_steps=100, seed=1)