from metadrive.metadrive.envs import MetaDriveEnv

env = MetaDriveEnv(dict(
        traffic_mode="respawn", map="SSS")) 

obs, info = env.reset()
for i in range(10000):
    action = [0, 0]
    # create action from info from vehicle (env.agent.positon & env.agent.velocity)
    


    obs, reward, terminated, truncated, info = env.step(action)
    env.render(mode="topdown",
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