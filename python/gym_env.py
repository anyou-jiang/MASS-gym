import gym
from Environment import Environment
import numpy as np
import torch
from Model import MuscleNN
import pyCharacter
import pyMuscle
import random
from torch import Tensor
from gym.spaces import Dict, Box

from torch.distributions import MultivariateNormal
from stable_baselines3.common.env_checker import check_env

class ObservationSpace(Box):
    def __init__(self, num_states):
        self._num_states = num_states
        self._low = np.full(self._num_states, -np.inf)
        self._high = np.full(self._num_states, np.inf)
        self._dtype = np.float64
        self._shape = (self._num_states,)
        super().__init__(self._low, self._high, dtype=self._dtype)


class ActionSpace(Dict):
    def __init__(self, num_action, num_total_related_dofs, num_muscles):
        self._num_action = num_action
        self._num_total_related_dofs = num_total_related_dofs
        self._num_muscles = num_muscles
        self._scale = 1 # scale for covariance matrix of the multivariate normal distribution
        super().__init__(spaces=dict({'motion': Box(-1, 1, (self._num_action,))}))

    def sample(self, mask={}):
        action = {}
        mean = np.zeros(self._num_action)
        covariance_matrix = np.eye(self._num_action) * self._scale
        action['motion'] = np.random.multivariate_normal(mean, covariance_matrix)

        must_use_muscle_model = False
        random_float = np.random.rand()
        if 'must_use_muscle_model' in mask:
            must_use_muscle_model = mask['must_use_muscle_model']
        if must_use_muscle_model:
            action['model'] = MuscleNN(self._num_total_related_dofs, self._num_action, self._num_muscles)
        else:
            action['model'] = None if random_float < 0.1 else MuscleNN(self._num_total_related_dofs, self._num_action, self._num_muscles)
        return action
    
    def seed(self, seed):
        np.random.seed(seed)

class gym_env(gym.Env):
    def __init__(self, meta_file, id=0, seed=0, max_step=np.inf):
        super(gym_env, self).__init__()
        np.random.seed(seed)

        self._mEnv = Environment('id={}'.format(id)) # always single environment
        self._mEnv.Initialize_from_file(meta_file, False)
        self._use_muscle = self._mEnv.GetUseMuscle()     
        self._num_muscles = self._mEnv.GetCharacter().GetNumOfMuscles()      
        if (not self._use_muscle):
            raise ValueError('Only muscle-based simulation is supported')      
        
        self._num_action = self._mEnv.GetNumAction()
        self._num_total_related_dofs = self._mEnv.GetNumTotalRelatedDofs()           
        self._action_space = ActionSpace(self._num_action, self._num_total_related_dofs, self._num_muscles)          
        self._muscle_model = self._action_space.sample(mask={'must_use_muscle_model':True})['model'] if self._use_muscle else None 
        self._observation = None
        self._step = 0       
        self._max_step = max_step
        self._num_simulation_per_control = 30
        self._num_states = self._mEnv.GetNumState()
        self._observation_space = ObservationSpace(self._num_states)    
        self._reward = 0
        self._done = False
        self._info = {}
        self._info['use_muscle'] = self._use_muscle
        self._info['num_of_muscle'] = self._num_muscles
        self._info['num_of_actions'] = self._num_action
        self._info['num_of_total_related_dofs'] = self._num_total_related_dofs       
        self._info['muscle_model'] = self._muscle_model
        self._info['max_step'] = self._max_step
        self._info['num_simulation_per_control'] = self._num_simulation_per_control
        self._info['num_of_states'] = self._num_states
        self._info['reward'] = self._reward         


    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):     
        return self._observation_space

    @property
    def reward_range(self):
        return (0, 1)

    @property
    def spec(self):
        return 'mass-v1.0'
    
    @property
    def metadata(self):
        return {'render.modes': []}

    @property
    def np_random(self):
        return np.random

    def reset(self):
        self._mEnv.Reset(True)
        self._observation = self._mEnv.GetState()
        self._step = 0
        self._reward = 0
        self._muscle_model = self._action_space.sample(mask={'must_use_muscle_model':True})['model'] if self._use_muscle else None

        self._info = {}
              
        return self._observation

    def step(self, action):
        self._step += 1

        motion = action['motion']
        if ('muscle_model' in action):
            self._muscle_model = action['muscle_model']
        
        if (self._use_muscle) and (self._muscle_model is None):
            raise ValueError('Muscle model is required for muscle-based simulation')

        self._mEnv.SetMotion(motion)
       
        tmp = self._mEnv.GetMuscleTorques()
        tmp = np.array(tmp, dtype = np.float64)	# TODO: to remove this line?		
        mt = Tensor(tmp)

        for i in range(self._num_simulation_per_control // 2):
            tmp = self._mEnv.GetDesiredTorques()
            tmp = np.array(tmp, dtype=np.float64) # TODO: to remove this line?
            dt = Tensor(tmp)
            activations = self._muscle_model(mt,dt).cpu().detach().numpy()
            self._mEnv.SetActivationLevels(activations)
            for i in range(2):
                self._mEnv.Step()

        nan_occur = False
        terminated_state = True
        self._reward = 0

        if np.any(np.isnan(motion)):
            nan_occur = True
        
        elif self._mEnv.IsEndOfEpisode() is False:
            terminated_state = False
            self._reward = self._mEnv.GetReward()

        # if terminated_state or nan_occur, then the user should reset the environment
        self._info['terminated'] = terminated_state
        self._info['truncated'] = self._step >= self._max_step or nan_occur
        self._info['max_step_reached'] = self._step >= self._max_step
        self._info['nan_occur'] = nan_occur
        self._observation = self._mEnv.GetState()
        self._done = self._info['terminated'] or self._info['truncated']
        return self._observation, self._reward, self._done, self._info

    def render(self):
        pass

    def close(self):
        self._mEnv.Reset(True)
        self._muscle_model = None 
        self._observation = None
        self._info = {}
        self._reward = 0
        self._action_space = None
        self._max_step = 0
        self._step = 0

import argparse
import os
if __name__=="__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--meta',help='meta file')
    args =parser.parse_args()
    if args.meta is None:
        print('Provide meta file')
        exit()

    env = gym_env(args.meta, id=0, seed=42, max_step=np.inf)
    check_env(env, warn=True, skip_render_check=True)

    observation = env.reset()
    for i in range(1000):
        env.action_space.seed(i)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print('step:{} reward={}, done ={}'.format(i, reward, done))
        if done:
            observation = env.reset()
            break

    env.close()
