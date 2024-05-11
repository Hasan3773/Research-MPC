from metadrive.metadrive.envs import MetaDriveEnv
from scipy.interpolate import CubicHermiteSpline
import numpy as np
import mpc 

env = MetaDriveEnv(dict(
        traffic_mode="respawn", map="SSS", use_render = True))       
      
obs, info = env.reset()
for i in range(10000):
    action = mpc.find_action(env.vehicle) # calls mpc solver

    obs, reward, terminated, truncated, info = env.step(action)

    env.render(mode="render",
    scaling=None,
    film_size=(500, 500),
    screen_size=(1500, 500),
    # target_vehicle_heading_up=True,
    camera_position=(0,0),
    screen_record=False,
    window=True,
    text={"episode_step": env.engine.episode_step,
            "mode": "Trigger"})
    
    if terminated or truncated:
            obs, info = env.reset()
env.close()

