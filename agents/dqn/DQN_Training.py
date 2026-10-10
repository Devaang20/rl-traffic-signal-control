import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

import Env_testing
from Environment import Traffic_Env

env = gym.make('gymnasium_env/TrafficSignalControl-v0')



model = DQN("MlpPolicy",
            env,
            tensorboard_log="./dqn_traffic_signal_tensorboard/",
            learning_rate=1e-4, 
            buffer_size=100000,
            learning_starts=1500,
            batch_size=256,
            gamma=0.99,
            train_freq=8,
            gradient_steps=1,
            target_update_interval=5000,
            exploration_fraction=0.40,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            policy_kwargs=dict(net_arch=[256,256]),
            verbose=1)  #1 for console logs 



eval_env = Monitor(Traffic_Env())

eval_callback = EvalCallback(
    eval_env,
    eval_freq=50_000,       # evaluate every 50k steps
    n_eval_episodes=20,
    deterministic=True,
    verbose=1
)

model.learn(
    total_timesteps=500_000,
    callback=eval_callback,
    tb_log_name="DQN_traffic_8queue_run1"
)

model.save("dqn_traffic_signal-v1")

#check_env(env,warn=True) -> no errors