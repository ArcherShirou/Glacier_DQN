import requests
import json


def post_api(url, data=None):
    if data:
        while True:
            try:
                res = requests.post(url, data=json.dumps(data))
                if res.status_code == 400:
                    print(res)
                    print("========= except trying ===========")
                    continue
            except requests.RequestException as e:
                print(e)
                print("========= except trying ===========")
                continue
            break
    else:
        while True:
            try:
                res = requests.post(url)
                if res.status_code == 400:
                    print(res)
                    print("========= except trying ===========")
                    continue
            except requests.RequestException as e:
                print(e)
                print("========= except trying ===========")
                continue
            break

    return res

def get_api(url, data=None):
    if data:
        while True:
            try:
                res = requests.get(url, data=json.dumps(data))
                if res.status_code == 400:
                    print(res)
                    print("========= except trying ===========")
                    continue
            except requests.RequestException as e:
                print(e)
                print("========= except trying ===========")
                continue
            break
    else:
        while True:
            try:
                res = requests.get(url)
                if res.status_code == 400:
                    print(res)
                    print("========= except trying ===========")
                    continue
            except requests.RequestException as e:
                print(e)
                print("========= except trying ===========")
                continue
            break

    return res