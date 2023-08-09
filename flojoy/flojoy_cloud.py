from flojoy import utils
from PIL import Image
import json
import os
import requests
import pandas as pd
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """json encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class FlojoyCloud:
    def __init__(self, apikey='default', content='application/json'):
        if apikey == 'default':
            apikey = utils.get_credentials()[0]['value']
        elif apikey == 'env':
            apikey = os.environ.get('FLOJOY_CLOUD_KEY')
        else:
            pass

        self.headers = {
          'api_key': apikey,
          'Content-Type': content
        }

    def create_payload(self, data, dc_type):
        match dc_type:
            case "matrix":
                payload = json.dumps({
                  "data": {
                    "type": "matrix",
                    "m": data
                  }
                })

            case "image":
                RGB_img = np.asarray(data)
                red_channel = RGB_img[:, :, 0]
                green_channel = RGB_img[:, :, 1]
                blue_channel = RGB_img[:, :, 2]

                if RGB_img.shape[2] == 4:
                    alpha_channel = RGB_img[:, :, 3]
                else:
                    alpha_channel = None
                payload = json.dumps({
                  "data": {
                      "type": "image",
                      "r": red_channel,
                      "g": green_channel,
                      "b": blue_channel,
                      "a": alpha_channel,
                  }
                }, cls=NumpyEncoder)
        return payload

    def store_dc(self, data, dc_type):
        url = "https://cloud.flojoy.ai/api/v1/dcs/"
        payload = self.create_payload(data, dc_type)
        response = requests.request("POST", url, headers=self.headers, data=payload)
        return response.text

    def fetch_dc(self, dc_id):
        url = f"https://cloud.flojoy.ai/api/v1/dcs/{dc_id}"
        response = requests.request("GET", url, headers=self.headers)
        return json.loads(response.text)

    def to_python(self, dc):
        dc_type = dc["dataContainer"]["type"]
        match dc_type:
            case "matrix":
                return pd.DataFrame(dc["dataContainer"]["m"])

            case "image":
                image = dc["dataContainer"]["image"]
                r = image["r"]
                g = image["g"]
                b = image["b"]
                a = image["a"]
                if a is None:
                    img_combined = np.stack((r, g, b), axis=2)
                else:
                    img_combined = np.stack((r, g, b, a), axis=2)

                return Image.fromarray(np.uint8(img_combined)).convert('RGB')
