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
    """
    A class that allows pulling and pushing DataContainers from the
    Flojoy cloud client (cloud.flojoy.ai).

    Returns data in a pythonic format (e.g. Pillow for images,
    DataFrames for arrays/matrices).

    Will support the majority of the Flojoy cloud API:
    https://rest.flojoy.ai/api-reference
    """

    def __init__(self, apikey="default", content="application/json"):
        if apikey == "default":
            apikey = utils.get_credentials()[0]["value"]
        elif apikey == "env":
            apikey = os.environ.get("FLOJOY_CLOUD_KEY")
        else:
            pass

        self.headers = {"api_key": apikey, "Content-Type": content}

    def create_payload(self, data, dc_type):
        """
        A method that formats data into a payload that can be handled by
        the Flojoy cloud client.
        """
        match dc_type:
            case "ordered_pair":
                if isinstance(data, dict) and "x" in data:
                    payload = json.dumps(
                        {
                            "data": {
                                "type": "ordered_pair",
                                "x": data["x"],
                                "y": data["y"],
                            }
                        },
                        cls=NumpyEncoder,
                    )
                else:
                    print(
                        "For ordered pair type, data must be in"
                        " dictionary form with keys 'x' and 'y'"
                    )
                    raise TypeError

            case "dataframe":
                data = data.to_json()
                payload = json.dumps({"data": {"type": "dataframe", "m": data}})

            case "matrix":
                payload = json.dumps(
                    {"data": {"type": "matrix", "m": data}}, cls=NumpyEncoder
                )

            case "scalar":
                payload = json.dumps(
                    {"data": {"type": "scalar", "c": data}}, cls=NumpyEncoder
                )

            case "image":
                RGB_img = np.asarray(data)
                red_channel = RGB_img[:, :, 0]
                green_channel = RGB_img[:, :, 1]
                blue_channel = RGB_img[:, :, 2]

                if RGB_img.shape[2] == 4:
                    alpha_channel = RGB_img[:, :, 3]
                else:
                    alpha_channel = None
                payload = json.dumps(
                    {
                        "data": {
                            "type": "image",
                            "r": red_channel,
                            "g": green_channel,
                            "b": blue_channel,
                            "a": alpha_channel,
                        }
                    },
                    cls=NumpyEncoder,
                )

        return payload

    def store_dc(self, data, dc_type):
        """
        A method that stores a formatted data payload onto the Flojoy cloud.
        """
        url = "https://cloud.flojoy.ai/api/v1/dcs/"
        payload = self.create_payload(data, dc_type)
        response = requests.request("POST", url, headers=self.headers, data=payload)
        return json.loads(response.text)

    def fetch_dc(self, dc_id):
        """
        A method that retrieves DataContainers from the Flojoy cloud.
        """
        url = f"https://cloud.flojoy.ai/api/v1/dcs/{dc_id}"
        response = requests.request("GET", url, headers=self.headers)
        return json.loads(response.text)

    def to_python(self, dc):
        """
        A method that converts data from DataContainers into pythonic
        data types like Pillow for images.
        """
        dc_type = dc["dataContainer"]["type"]
        match dc_type:
            case "ordered_pair":
                df = pd.DataFrame(dc["dataContainer"])
                return df.drop(columns=["type"])

            case "dataframe":
                df = pd.DataFrame(dc["dataContainer"]["m"])
                return df

            case "matrix":
                return pd.DataFrame(dc["dataContainer"]["m"])

            case "scalar":
                return float(dc["dataContainer"]["c"])

            case "image":
                image = dc["dataContainer"]
                r = image["r"]
                g = image["g"]
                b = image["b"]
                if "a" in image:
                    a = image["a"]
                    img_combined = np.stack((r, g, b, a), axis=2)
                    return Image.fromarray(np.uint8(img_combined)).convert("RGBA")
                else:
                    img_combined = np.stack((r, g, b), axis=2)
                    return Image.fromarray(np.uint8(img_combined)).convert("RGB")
