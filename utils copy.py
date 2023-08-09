def animate_episode(env, steps, sleep=0.1):
    from IPython.display import clear_output
    import matplotlib.pyplot as plt
    import time

    env.reset()

    plt.imshow(env.render(mode='rgb_array'))
    plt.axis('off')
    plt.show()

    for i in range(steps):
        time.sleep(sleep)
        clear_output(wait=True)
        a = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(a)
        plt.imshow(env.render(mode='rgb_array'))
        plt.axis('off')
        plt.show()

        if terminated:
            print('Terminated')
            break

            
def render_mp4(videopath):
    from base64 import b64encode
    mp4 = open(videopath, 'rb').read()
    base64_encoded_mp4 = b64encode(mp4).decode()
    return f'<video width=400 controls><source src="data:video/mp4;' \
           f'base64,{base64_encoded_mp4}" type="video/mp4"></video>'


class SB3Agent:
    def __init__(self, model): 
        self.model = model
    def select_action(self, state):
        return self.model.predict(state)[0]

class RandomAgent:
    def __init__(self, env):
        self.env = env
    def select_action(self, state):
        return self.env.action_space.sample()


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
    env, agent, episodes=1, max_steps=250, seed=None, fps=20, vec_env=False,
    folder='', filename='', processor=None, display_gif=True
    ):
    
    import os
    import torch
    import imageio
    from IPython.display import Image, display, HTML

    if max_steps is None: max_steps = float('inf')
    if processor is None: processor = lambda x : x

    #--------------------------------------------------------------------------------------
    # Set seeds, if needed. Note: Seed not yet passed to agent. 
    #--------------------------------------------------------------------------------------    
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
            action = agent.select_action(state)
            action = [action] if vec_env else action
            state, reward, terminated, truncated, info = env.step(action)
            frames.append(processor(env.render()))
            if terminated:
                break

    print(f'{t} steps completed.')
    os.makedirs(folder, exist_ok=True)
    imageio.mimsave(f'{folder}/{filename}.gif', frames, format='GIF', fps=fps)   

    if display_gif:
        with open(f'{folder}/{filename}.gif','rb') as f:
            display(Image(data=f.read(), format='png'))


def generate_episode(
    env, agent, max_steps=None, seed=None, verbose=False, vec_env=False):
    
    import numpy as np
    
    if max_steps is None:
        max_steps = float('inf')
    
    #--------------------------------------------------------
    # Set seeds
    #--------------------------------------------------------
    if seed is None:
        state, info = env.reset()
    else:
        state, info = env.reset(seed=seed)
        env.action_space.seed(seed)
        # Set the agent seed?
        
    #--------------------------------------------------------
    # Output Header
    #--------------------------------------------------------   
    if verbose:
        print(f'{"step":<6}{"action":<8}{"new_state":<11}{"reward":<8}{"terminated":<12}{"info":<8}')
        print('-'*58)
            
    #--------------------------------------------------------
    # Create history dictionary 
    # Note: States will be one longer than the others.
    #--------------------------------------------------------
    history = {
        'states' : [state],
        'actions' : [],
        'rewards' : [] 
    } 

    #--------------------------------------------------------
    # Loop for max steps episodes
    #--------------------------------------------------------
    t = 0
    while t < max_steps:
        t += 1
        
        action = agent.select_action(state)
        
        if vec_env:
            state, reward, terminated, truncated, info = env.step([action])
        else:
            state, reward, terminated, truncated, info = env.step(action)
        
        history['states'].append(state)
        history['actions'].append(action)
        history['rewards'].append(reward)
        
        s = str(state)
        if verbose:
            print(f'{t:<6}{action:<8}{s:11}{reward:<8}{str(terminated):<12}{str(info):<8}')

        if terminated:
            break

    if verbose:
        print('-'*58)
        print(t + 1, 'steps completed.')

    return history
    

def success_rate(env, agent, episodes, max_steps=1000, seed=None):
    import numpy as np
    
    goal_count = 0
    fail_count = 0
    total_eps = 0
    total_eps_s = 0
    total_eps_f = 0
    
    if seed is not None:
        np.random.seed(seed)
        seeds = np.random.choice(10*episodes, episodes)
    
    for i in range(episodes):
        sd = None if seed is None else int(seeds[i])
        obs, info = env.reset(seed=sd)
        
        for j in range(max_steps):
            try:
                a = policy[obs]
            except:
                a = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(a)
            if terminated:
                break
        total_eps += j
        if env.unwrapped.s == env.observation_space.n - 1:
            goal_count += 1
            total_eps_s += j
        else:
            fail_count += 1
            total_eps_f += j

    sr = goal_count / n
    info = {
        'sr' : sr,
        'avg_len' : total_eps / n,
        'avg_len_s' : None if goal_count == 0 else total_eps_s / goal_count,
        'avg_len_f' : None if fail_count == 0 else total_eps_f / fail_count
        
    }

    return sr, info