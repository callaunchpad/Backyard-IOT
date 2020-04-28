import requests
import time

URL = "http://0.0.0.0:5000"
while(True):
    try:
        response = requests.get(URL)
        if response:
            print('Success!')
            response_dict = response.json()
            print("Responses:")
            for i in response_dict.keys():
                print("%s: %s", i, response_dict[i])
            print()
            time.sleep(1)
        else:
            print('ERROR:', response.status_code)
    except requests.exceptions.ConnectionError:
        print("Couldn't Connect! Retrying...")
        time.sleep(1)
