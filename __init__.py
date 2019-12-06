import gym, ray
from ray.rllib.agents import sac
from ray.rllib.agents import ppo
from ray.tune.logger import pretty_print
import pystk
import numpy as np
import tensorflow as tf

print(pystk.list_karts())

class TuxEnv(gym.Env):
    def __init__(self, _):
        print("calling __init__")
        gfx_config = pystk.GraphicsConfig.ld()
        gfx_config.screen_width = 160
        gfx_config.screen_height = 120
        gfx_config.render_window = True
        pystk.clean()
        pystk.init(gfx_config)

        # Current action space is only steering left/right
        self.action_space = gym.spaces.Tuple([
                gym.spaces.Box(low=-1.0, high=1.0, shape=(1,)),
                gym.spaces.Discrete(2)])
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(120, 160, 3))

        self.race = None
        self.max_step = 4000
        self.curr_iter = 0
        self.prev_distance = 0

    def reset(self):
        self.curr_iter = 0
        self.prev_distance = 0

        race_config = pystk.RaceConfig(num_kart=6, mode=pystk.RaceConfig.RaceMode.FREE_FOR_ALL)
        race_config.players[0].controller = pystk.PlayerConfig.Controller.PLAYER_CONTROL
        #race_config.players[0].kart = "gnu"
        race_config.track = 'battleisland'
        race_config.step_size = 0.1

        if self.race != None:
            inst = np.asarray(self.race.render_data[0].instance) >> 24 
            img = np.asarray(self.race.render_data[0].image) / 255
            #i = np.concatenate((img,inst[..., np.newaxis]), axis=2)
            i = img
            self.race.stop()
            self.race = None

        self.race = pystk.Race(race_config)
        self.race.start()
        self.race.step()

        # return obs
        inst = np.asarray(self.race.render_data[0].instance) >> 24 
        img = np.asarray(self.race.render_data[0].image) / 255
        #i = np.concatenate((img,inst[..., np.newaxis]), axis=2)
        i = img
        return i 
    def step(self, action):
        self.curr_iter +=1

        steer_dir = action[0]
        rescue = action[1]
        action = pystk.Action(steer=steer_dir, acceleration=0.5) # , rescue=rescue)
        self.race.step(action)
        self.race.step(action)
        self.race.step(action)
        self.race.step(action)
        self.race.step(action)
        self.race.step(action)
        self.race.step(action)

        state = pystk.WorldState()
        state.update()

        inst = np.asarray(self.race.render_data[0].instance) >> 24 
        img = np.asarray(self.race.render_data[0].image) / 255
        #i = np.concatenate((img,inst[..., np.newaxis]), axis=2)
        i = img

        new_distance = state.karts[0].distance_down_track
        delta = new_distance - self.prev_distance

        reward = np.linalg.norm(state.karts[0].velocity) + new_distance / 10.0
        #reward = new_distance - self.prev_distance
        #reward = new_distance / 10000
        reward += (new_distance - state.karts[0].overall_distance)**2 + delta*20
        
        scores = state.ffa.scores
        kart = state.players[0].kart
        rank = sorted(scores, reverse=True).index(scores[kart.id])
        score = {0:10,1:8,2:6}.get(rank, 7-rank)
        reward += score

        is_stuck = (self.curr_iter > 10) and delta < 0.001

        #if is_stuck != rescue:
        #    reward -= 10000

        done = (self.curr_iter == self.max_step) or is_stuck

        # if self.curr_iter % 6 == 0:
        self.prev_distance = new_distance

        # return <obs>, <reward: float>, <done: bool>, <info: dict>
        return i, reward, done, {}

from ray import tune

if __name__ == '__main__':
    ray.init()
    tune.run(
        "PPO",
        checkpoint_freq=1,
        stop={"training_iteration": 50},
        config={
            "env": TuxEnv,
    		"evaluation_interval": 0,
            "num_gpus": 1,
            "num_workers": 0,
            "lambda": 0.95,
            "kl_coeff": 0.2,
    		"train_batch_size": 1000,
            "vf_clip_param": 20,
            "num_sgd_iter": 30,
            "entropy_coeff": 0.01,
            "eager": False,
            #"lr": tune.grid_search([0.001, 0.0001]),
        },
    )
