import requests
import grpc
import json
import random
import time

from requst_url import post_api, get_api

url = 'http://121.89.205.93:8030'
reset = '/api/train/start'
get_api(url + reset)
time.sleep(3)

auto_mine = '/api/train/robot_id'
data = get_api(url + auto_mine).json()['data']
print(data)


