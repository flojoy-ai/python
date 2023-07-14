import decimal
import json as _json

import numpy as np
import pandas as pd
from pathlib import Path
import os
import yaml
from typing import Union, Any
import requests
from dotenv import dotenv_values  # type:ignore
import difflib

__all__ = [
    "send_to_socket",
    "get_frontier_api_key",
    "set_frontier_api_key",
    "set_frontier_s3_key",
]

env_vars = dotenv_values("../.env")
port = env_vars.get("VITE_BACKEND_PORT", "8000")
BACKEND_URL = os.environ.get("BACKEND_URL", f"http://127.0.0.1:{port}")


def send_to_socket(data: str):
    print("posting data to socket:", f"{BACKEND_URL}/worker_response", flush=True)
    requests.post(f"{BACKEND_URL}/worker_response", json=data)


def find_closest_match(given_str: str, available_str: list[str]):
    closest_match = difflib.get_close_matches(given_str, available_str, n=1)
    if closest_match:
        return closest_match[0]
    else:
        return None


class PlotlyJSONEncoder(_json.JSONEncoder):
    """
    Meant to be passed as the `cls` kwarg to json.dumps(obj, cls=..)
    See PlotlyJSONEncoder.default for more implementation information.
    Additionally, this encoder overrides nan functionality so that 'Inf',
    'NaN' and '-Inf' encode to 'null'. Which is stricter JSON than the Python
    version.
    """

    def coerce_to_strict(self, const: Any):
        """
        This is used to ultimately *encode* into strict JSON, see `encode`
        """
        # before python 2.7, 'true', 'false', 'null', were include here.
        if const in ("Infinity", "-Infinity", "NaN"):
            return None
        else:
            return const

    def encode(self, o: Any):
        """
        Load and then dump the result using parse_constant kwarg
        Note that setting invalid separators will cause a failure at this step.
        """
        # this will raise errors in a normal-expected way
        encoded_o = super(PlotlyJSONEncoder, self).encode(o)
        # Brute force guessing whether NaN or Infinity values are in the string
        # We catch false positive cases (e.g. strings such as titles, labels etc.)
        # but this is ok since the intention is to skip the decoding / reencoding
        # step when it's completely safe

        if not ("NaN" in encoded_o or "Infinity" in encoded_o):
            return encoded_o

        # now:
        #    1. `loads` to switch Infinity, -Infinity, NaN to None
        #    2. `dumps` again so you get 'null' instead of extended JSON
        try:
            new_o = _json.loads(encoded_o, parse_constant=self.coerce_to_strict)
        except ValueError:
            # invalid separators will fail here. raise a helpful exception
            raise ValueError(
                "Encoding into strict JSON failed. Did you set the separators "
                "valid JSON separators?"
            )
        else:
            return _json.dumps(
                new_o,
                sort_keys=self.sort_keys,
                indent=self.indent,
                separators=(self.item_separator, self.key_separator),
            )

    def default(self, obj: dict[str, Any]):
        """
        Accept an object (of unknown type) and try to encode with priority:
        1. builtin:     user-defined objects
        2. sage:        sage math cloud
        3. pandas:      dataframes/series
        4. numpy:       ndarrays
        5. datetime:    time/datetime objects
        Each method throws a NotEncoded exception if it fails.
        The default method will only get hit if the object is not a type that
        is naturally encoded by json:
            Normal objects:
                dict                object
                list, tuple         array
                str, unicode        string
                int, long, float    number
                True                true
                False               false
                None                null
            Extended objects:
                float('nan')        'NaN'
                float('infinity')   'Infinity'
                float('-infinity')  '-Infinity'
        Therefore, we only anticipate either unknown iterables or values here.
        """
        # TODO: The ordering if these methods is *very* important. Is this OK?
        encoding_methods = (
            self.encode_as_plotly,
            self.encode_as_numpy,
            self.encode_as_pandas,
            self.encode_as_datetime,
            self.encode_as_date,
            self.encode_as_list,  # because some values have `tolist` do last.
            self.encode_as_decimal,
        )
        for encoding_method in encoding_methods:
            try:
                return encoding_method(obj)
            except NotEncodable:
                pass
        return _json.JSONEncoder.default(self, obj)

    @staticmethod
    def encode_as_plotly(obj: dict[str, Any]):
        """Attempt to use a builtin `to_plotly_json` method."""
        try:
            return obj.to_plotly_json()
        except AttributeError:
            raise NotEncodable

    @staticmethod
    def encode_as_list(obj):
        """Attempt to use `tolist` method to convert to normal Python list."""
        if hasattr(obj, "tolist"):
            return obj.tolist()
        else:
            raise NotEncodable

    @staticmethod
    def encode_as_pandas(obj):
        """Attempt to convert pandas.NaT"""
        if not pd:
            raise NotEncodable
        if obj is pd.NaT:
            return None
        elif isinstance(obj, pd.DataFrame):
            return obj.to_json()
        else:
            raise NotEncodable

    @staticmethod
    def encode_as_numpy(obj):
        """Attempt to convert numpy.ma.core.masked"""
        if not np:
            raise NotEncodable

        if obj is np.ma.masked:
            return float("nan")
        elif isinstance(obj, np.ndarray) and obj.dtype.kind == "M":
            try:
                return np.datetime_as_string(obj).tolist()
            except TypeError:
                pass

        raise NotEncodable

    @staticmethod
    def encode_as_datetime(obj):
        """Convert datetime objects to iso-format strings"""
        try:
            return obj.isoformat()
        except AttributeError:
            raise NotEncodable

    @staticmethod
    def encode_as_date(obj):
        """Attempt to convert to utc-iso time string using date methods."""
        try:
            time_string = obj.isoformat()
        except AttributeError:
            raise NotEncodable
        else:
            return time_string  # iso_to_plotly_time_string(time_string)

    @staticmethod
    def encode_as_decimal(obj):
        """Attempt to encode decimal by converting it to float"""
        if isinstance(obj, decimal.Decimal):
            return float(obj)
        else:
            raise NotEncodable


class NotEncodable(Exception):
    pass


def dump_str(result: Any, limit: int | None = None):
    result_str = str(result)
    return (
        result_str
        if limit is None or len(result_str) <= limit
        else result_str[:limit] + "..."
    )


def get_flojoy_root_dir() -> str:
    home = str(Path.home())
    path = os.path.join(home, ".flojoy/flojoy.yaml")
    stream = open(path, "r")
    yaml_dict = yaml.load(stream, Loader=yaml.FullLoader)
    root_dir = ""
    if isinstance(yaml_dict, str):
        root_dir = yaml_dict.split(":")[1]
    else:
        root_dir = yaml_dict["PATH"]
    return root_dir


def get_frontier_api_key() -> Union[str, None]:
    home = str(Path.home())
    api_key = None
    path = os.path.join(home, ".flojoy/credentials")
    if not os.path.exists(path):
        return api_key

    stream = open(path, "r", encoding="utf-8")
    yaml_dict = yaml.load(stream, Loader=yaml.FullLoader)
    if yaml_dict is None:
        return api_key
    if isinstance(yaml_dict, str) == True:
        split_by_line = yaml_dict.split("\n")
        for line in split_by_line:
            if "FRONTIER_API_KEY" in line:
                api_key = line.split(":")[1]
    else:
        api_key = yaml_dict.get("FRONTIER_API_KEY", None)
    return api_key


def set_frontier_api_key(api_key: str):
    try:
        home = str(Path.home())
        file_path = os.path.join(home, ".flojoy/credentials")

        if not os.path.exists(file_path):
            # Create a new file and write the API_KEY to it
            with open(file_path, "w") as file:
                file.write(f"FRONTIER_API_KEY:{api_key}\n")
        else:
            # Read the contents of the file
            with open(file_path, "r") as file:
                lines = file.readlines()

            # Update the API key if it exists, otherwise append a new line
            updated = False
            for i, line in enumerate(lines):
                if line.startswith("FRONTIER_API_KEY:"):
                    lines[i] = f"FRONTIER_API_KEY:{api_key}\n"
                    updated = True
                    break

            if not updated:
                lines.append(f"FRONTIER_API_KEY:{api_key}\n")
            # Write the updated contents to the file
            with open(file_path, "w") as file:
                file.writelines(lines)

    except Exception as e:
        raise e


def set_frontier_s3_key(s3_name: str, s3_access_key: str, s3_secret_key: str):
    home = str(Path.home())
    file_path = os.path.join(home, os.path.join(".flojoy", "credentials.yaml"))

    data = {
        f"{s3_name}": s3_name,
        f"{s3_name}accessKey": s3_access_key,
        f"{s3_name}secretKey": s3_secret_key,
    }
    if not os.path.exists(file_path):
        # Create a new file and write the ACCSS_KEY to it
        with open(file_path, "w") as file:
            yaml.dump(data, file)
        return

    # Read the contents of the file
    with open(file_path, "r") as file:
        load = yaml.safe_load(file)

    load[f"{s3_name}"] = s3_name
    load[f"{s3_name}accessKey"] = s3_access_key
    load[f"{s3_name}secretKey"] = s3_secret_key

    with open(file_path, "w") as file:
        yaml.dump(load, file)
