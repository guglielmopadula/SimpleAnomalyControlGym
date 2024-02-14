import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
import matplotlib.pyplot as plt
class AnomalyControl(gym.Env):

    def __init__(self):
        #A machine is working normally when working full (machine status 0) and without anomalies (anomaly status 0). At random, there is a big anomaly (status 4). 
        #To revert the anomaly, the machine must go in repair mode(status 1). In this mode, the anomaly will decrease to 1 at every step. In full mode, intermediate anomalies will increase.

        self.observation_space= spaces.Dict({
                            "anomaly_status":spaces.Discrete(4),
                            "machine_status":spaces.Discrete(2),
                            "time":spaces.Discrete(100)
        })


        # We have 2 actions, corresponding to "change mode (1)" "or do nothing (0)"
        self.action_space = spaces.Discrete(2)

    def _get_obs(self):
        return {"anomaly_status": self._anomaly_status, "machine_status": self._machine_status, "time": self._time}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._anomaly_status=0
        self._machine_status=0
        self._time=0
        observation = self._get_obs()
        info = {}

        return observation, info
    
    def step(self, action):

        if action==1:
            reward=0
            if (self._anomaly_status>0)==(self._machine_status>0):
                reward=-1

            self._machine_status=(self._machine_status+1)%2

            observation=self._get_obs()

        if action==0:
            if self._machine_status==0:

                if self._anomaly_status==0:
                    reward=0
                    tmp=np.random.rand()
                    if tmp<0.1:
                        self._anomaly_status=3
                    
                    observation=self._get_obs()

                elif self._anomaly_status==3:
                    reward=-1
                    observation=self._get_obs()

                else:
                    reward=-1
                    self._anomaly_status+=1
                    observation=self._get_obs()


            else:
                if self._anomaly_status==0:
                    reward=-1
                    observation=self._get_obs()

                else:
                    reward=0
                    self._anomaly_status=self._anomaly_status-1
                    observation=self._get_obs()

        done = False
        info = {}
        self._time+=1
        if self._time==99:
            done = True

        return observation, reward, False, done , info
    

env = AnomalyControl()
env.reset()
check_env(env)


model = PPO('MultiInputPolicy', env, verbose=1,learning_rate=0.00005)
model.learn(total_timesteps=200000)
vec_env = model.get_env()
obs = vec_env.reset()
action_list=[]
anomaly_list=[]
machine_list=[]
rev=0

for i in range(100):
    action, _states = model.predict(obs)
    action_list.append(action.item())
    obs, rewards, _ ,dones, info = env.step(action)
    rev=rev+rewards
    anomaly_list.append(obs["anomaly_status"])
    machine_list.append(obs["machine_status"])

print(action_list)
print(anomaly_list)
print(machine_list)


fig, ax = plt.subplots(3,1)
ax[0].plot(action_list)
ax[0].set_title('action')
ax[1].plot(anomaly_list)
ax[1].set_title('anomaly')
ax[2].plot(machine_list)
ax[2].set_title('machine')
fig.savefig('test.png')

print(rev)