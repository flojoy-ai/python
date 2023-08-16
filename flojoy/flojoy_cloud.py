from flojoy import utils
from PIL import Image
import json
import os
import requests
import pandas as pd
import numpy as np
from pydantic import ValidationError, validator
from typing import Optional, Generic, TypeVar
from pydantic.generics import GenericModel


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


def check_deserialize(response):
    DataC = TypeVar("DataC")

    class DefaultModel(GenericModel, Generic[DataC]):
        ref: str
        dataContainer: dict
        metadata: dict
        workspaceId: str
        privacy: str
        location: str
        note: str

    class OrderedPairModel(DefaultModel[DataC], Generic[DataC]):
        data: Optional[DataC]

        @validator("dataContainer")
        def type_must_match(cls, dc):
            assert dc["type"] == "OrderedPair", "dataContainer type does not match."
            return dc

        @validator("dataContainer")
        def keys_must_match(cls, dc):
            assert "x" in dc, 'dataContainer does not contain "x" dataset.'
            assert "y" in dc, 'dataContainer does not contain "y" dataset.'
            assert isinstance(dc["x"], list), '"x" dataset is not a list'
            assert isinstance(dc["y"], list), '"y" dataset is not a list'
            return dc

    class DataFrameModel(DefaultModel[DataC], Generic[DataC]):
        data: Optional[DataC]

        @validator("dataContainer")
        def type_must_match(cls, dc):
            assert dc["type"] == "DataFrame", "dataContainer type does not match."
            return dc

        @validator("dataContainer")
        def keys_must_match(cls, dc):
            assert "m" in dc, 'dataContainer does not contain "m" dataset.'
            assert isinstance(
                dc["m"], dict
            ), f'"m" dataset is not a list, type: {type(dc["m"])}'
            return dc

    class MatrixModel(DefaultModel[DataC], Generic[DataC]):
        data: Optional[DataC]

        @validator("dataContainer")
        def type_must_match(cls, dc):
            assert dc["type"] == "Matrix", "dataContainer type does not match."
            return dc

        @validator("dataContainer")
        def keys_must_match(cls, dc):
            assert "m" in dc, 'dataContainer does not contain "m" dataset.'
            assert isinstance(dc["m"], list), '"m" dataset is not a list'
            return dc

    class ScalarModel(DefaultModel[DataC], Generic[DataC]):
        data: Optional[DataC]

        @validator("dataContainer")
        def type_must_match(cls, dc):
            assert dc["type"] == "Scalar", "dataContainer type does not match."
            return dc

        @validator("dataContainer")
        def keys_must_match(cls, dc):
            assert "c" in dc, 'dataContainer does not contain "m" dataset.'
            # int as well just in case?
            assert isinstance(dc["c"], float), '"c" dataset is not a float'
            return dc

    class ImageModel(DefaultModel[DataC], Generic[DataC]):
        data: Optional[DataC]

        @validator("dataContainer")
        def type_must_match(cls, dc):
            assert dc["type"] == "Image", "dataContainer type does not match."
            return dc

        @validator("dataContainer")
        def keys_must_match(cls, dc):
            assert "r" in dc, 'dataContainer does not contain "r" dataset.'
            assert "g" in dc, 'dataContainer does not contain "r" dataset.'
            assert "b" in dc, 'dataContainer does not contain "r" dataset.'
            assert isinstance(dc["r"], list), '"r" dataset is not a list'
            assert isinstance(dc["g"], list), '"g" dataset is not a list'
            assert isinstance(dc["b"], list), '"b" dataset is not a list'
            return dc

    dc_type = response["dataContainer"]["type"]
    match dc_type:
        case "OrderedPair":
            try:
                OrderedPairModel.parse_obj(response)
            except ValidationError as e:
                print(e)
        case "DataFrame":
            try:
                DataFrameModel.parse_obj(response)
            except ValidationError as e:
                print(e)
        case "Matrix":
            try:
                MatrixModel.parse_obj(response)
            except ValidationError as e:
                print(e)
        case "Scalar":
            try:
                ScalarModel.parse_obj(response)
            except ValidationError as e:
                print(e)
        case "Image":
            try:
                ImageModel.parse_obj(response)
            except ValidationError as e:
                print(e)


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
            case "OrderedPair":
                if isinstance(data, dict) and "x" in data:
                    payload = json.dumps(
                        {
                            "data": {
                                "type": "OrderedPair",
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

            case "DataFrame":
                data = data.to_json()
                payload = json.dumps({"data": {"type": "DataFrame", "m": data}})

            case "Matrix":
                payload = json.dumps(
                    {"data": {"type": "Matrix", "m": data}}, cls=NumpyEncoder
                )

            case "Scalar":
                payload = json.dumps(
                    {"data": {"type": "Scalar", "c": data}}, cls=NumpyEncoder
                )

            case "Image":
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
                            "type": "Image",
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
        response = json.loads(response.text)
        check_deserialize(response)
        return response

    def to_python(self, dc):
        """
        A method that converts data from DataContainers into pythonic
        data types like Pillow for images.
        """
        dc_type = dc["dataContainer"]["type"]
        match dc_type:
            case "OrderedPair":
                df = pd.DataFrame(dc["dataContainer"])
                return df.drop(columns=["type"])

            case "DataFrame":
                df = pd.DataFrame(dc["dataContainer"]["m"])
                return df

            case "Matrix":
                return pd.DataFrame(dc["dataContainer"]["m"])

            case "Scalar":
                return float(dc["dataContainer"]["c"])

            case "Image":
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
