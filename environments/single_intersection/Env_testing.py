import gymnasium as gym
import numpy as np  # noqa: F401

#from Environment import Traffic_Env
from gymnasium.envs.registration import register

#import inspect
register(
    id="gymnasium_env/TrafficSignalControl-v0",
    max_episode_steps=500,
    entry_point="Environment:Traffic_Env", 
)

env = gym.make('gymnasium_env/TrafficSignalControl-v0')
'''
#print(inspect.getsource(env.reset))
# Test 1 — reset
obs, info = env.reset(seed=42)
print("Reset obs:", obs)
print("Obs shape:", obs.shape)
print("Obs dtype:", obs.dtype)

# Test 2 — observation space check
print("Obs valid:", env.observation_space.contains(obs))

# Test 3 — random steps
for step in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {step+1} | action={action} | reward={reward:.2f} | queues={obs[0:4]} | phase={obs[4]} | amb={obs[5]} | truncated={truncated}")
    print(f"         | delta_Q={info['delta_Q']} | amb_penalty={info['ambulance_penalty']} | starv_penalty={info['starvation_penalty']}")
    if terminated or truncated:
        obs, info = env.reset()
        break

print("\nAll checks passed.")
'''