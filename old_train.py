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
        gfx_config.screen_width = 128
        gfx_config.screen_height = 96
        gfx_config.render_window = True
        pystk.clean()
        pystk.init(gfx_config)

        # Current action space is only steering left/right
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,))
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(96, 128, 3))

        self.race = None
        self.max_step = 200
        self.curr_iter = 0
        self.prev_distance = 0

    def reset(self):
        print("calling reset")
        self.curr_iter = 0
        self.prev_distance = 0

        race_config = pystk.RaceConfig()
        race_config.players[0].controller = pystk.PlayerConfig.Controller.PLAYER_CONTROL
        race_config.players[0].kart = "gnu"
        race_config.track = 'lighthouse'
        race_config.step_size = 0.1

        if self.race != None:
            print("stopping race")
            i = np.asarray(self.race.render_data[0].image) / 255
            self.race.stop()
            self.race = None

        self.race = pystk.Race(race_config)
        self.race.start()
        self.race.step()

        # return obs
        return np.asarray(self.race.render_data[0].image) / 255

    def step(self, action):
        self.curr_iter +=1
        i = np.asarray(self.race.render_data[0].image) / 255

        # Applying predicted action
        steer_dir = action[0]

        # TODO experiment with different accelerations initially, may
        # be easier to train.
        action = pystk.Action(steer=steer_dir, acceleration=1)

        info = dict()

        state = pystk.WorldState()
        state.update()

        velocity_norm = np.linalg.norm(state.karts[0].velocity)

        #
        # Reward based on change in distance along track
        # - this is something we should tune
        #
        new_distance = state.karts[0].distance_down_track
        reward = new_distance - self.prev_distance
        self.prev_distance = new_distance


        self.race.step(action)
        done = (self.curr_iter == self.max_step)

        # return <obs>, <reward: float>, <done: bool>, <info: dict>
        return i, reward, done, info

ray.init()
sac_config = sac.DEFAULT_CONFIG.copy()
sac_config["num_workers"] = 0
sac_config["num_gpus"] = 1

# Would like to remove line, but when evaluation is enabled, a new supertux
# race is started on same process as the trainer, causing crash.
sac_config["evaluation_interval"] = 0 

trainer = sac.SACTrainer(env=TuxEnv, config=sac_config)
#trainer = ppo.PPOTrainer(env=TuxEnv, config=sac_config)

while True:
    print(trainer.train())
