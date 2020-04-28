import requests
import time

URL = "http://192.168.1.108:5000/animals"
while(True):
    try:
        response = requests.get(URL)
        if response:
            print('Success!')
            response_dict = response.json()
            print("Inference Time:", response_dict["inference time"], "seconds")
            for i in response_dict["results"].keys():
                print(str(i) + ": " + str(response_dict["results"][i]))
            print()
            time.sleep(1)
        else:
            print('ERROR:', response.status_code)
    except requests.exceptions.ConnectionError:
        print("Couldn't Connect! Retrying...")
        time.sleep(1)
