import random
import numpy as np
import requests
import json
import time
from random import shuffle
from requst_url import post_api, get_api


class AgentEnv:
    def __init__(self, config, is_test=False):
        self.get_npc_id_url = config['url'] + config['get_npc_id_url']
        self.reset_url = config['url'] + config['reset_url']
        self.step_url = config['url'] + config['step_url']
        self.action_list = ["accepttask", "startmine", "submittask", "attack"]
        self.reset()

    def step(self, npc_id, action_value):

        action_value = self.action_sorted(npc_id, self.npc_action[npc_id], action_value)
        action_list = []
        for action, v in action_value:
                action_list.append(action)

        data = {"robotid": npc_id, "actionlist": action_list}
        behave = post_api(self.step_url, data)
        print(behave.text)
        execute = behave.json()['data']

        if execute == 'accepttask':
            reward = 0

        if execute == 'startmine':
            reward = 1

        if execute == 'attack':
            reward = 2

        if execute == 'submittask':
            reward = 3

        return reward


    def get_obs(self, npc_id):

        self.npc_state[npc_id] = np.random.randint(10, size=4)

        return np.array(self.npc_state[npc_id], dtype=np.float32)


    def reset(self):
        get_api(self.reset_url)
        time.sleep(3)
        self.npc_id = get_api(self.get_npc_id_url).json()['data']
        self.npc_state = {}
        self.npc_state_size = {}

        self.npc_action = {}
        self.npc_action_size = {}


        for i in self.npc_id:
            self.npc_state[i] = np.random.randint(10, size=4)
            self.npc_state_size[i] = len(self.npc_state[i])

            self.npc_action[i] = self.action_list
            self.npc_action_size[i] = len(self.action_list)

    def action_sorted(self, npc_id, action, value):
        action_value = dict(zip(action, value))
        action_value = sorted(action_value.items(), key=lambda x: x[1], reverse=True)
        return action_value

