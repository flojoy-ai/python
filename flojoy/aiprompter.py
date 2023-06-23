import openai
import os
import numpy as np
from tqdm import tqdm
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import subprocess
import argparse
import yaml


def find_2nd(string, substring):
    return string.find(substring, string.find(substring) + 1)


openai.api_key = os.environ["OPENAI_KEY"]

primary_functions = [
    func_name
    for func_name in dir(np.random)
    if callable(getattr(np.random, func_name)) and not func_name.startswith("_")
]


def generate_wrapper_davinci(function):
    try:
        getattr(np.random, function).__name__
    except AttributeError:
        return "", ""
    message = [
        {
            "role": "user",
            "content": f"Provide to me a wrapper in Python for numpy.random.{function} that includes the following: the input parameters of the wrapper must be 'dc' and 'params', the name of the wrapper must be {getattr(np.random, function).__name__.upper()}, the primary argument of the function must be 'dc[0].y', all optional arguments must be strictly typed and taken as keys of the same name from the dictionary 'params', all internal variables must be strictly typed, and the doc string used in the wrapper must be the original doc string of the function. The functions must enforce strict typing using Python 3.10 syntax.",
        }
    ]
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=message[0]["content"],
        max_tokens=2048,  # Adjust as per your requirements
        # n=1,  # Number of completions to generate
        top_p=1,
        temperature=0.0,  # Controls randomness of the output
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )

    response_retval = openai.Completion.create(
        model="text-davinci-003",
        prompt=f"Change the following Python3.10 function to return a custom class 'DataContainer', instantiated with the arguments 'x=dc[0].y, y=<out>', where <out> is the return value of the function: {response.choices[0]['text']}",
        max_tokens=2048,  # Adjust as per your requirements
        # n=1,  # Number of completions to generate
        top_p=1,
        temperature=0.0,  # Controls randomness of the output
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    return function, response_retval.choices[0]["text"]


def generate_manifest(function):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=f"Return all arguments of the numpy.random.{function} function as keys of a YAML tree of the name 'parameters', giving their default values as a sub-key 'default' and the type of parameter as the a subkey 'type'. If they have no default value, leave the entry blank.",
        max_tokens=2048,  # Adjust as per your requirements
        # n=1,  # Number of completions to generate
        top_p=1,
        temperature=0.0,  # Controls randomness of the output
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    data = yaml.safe_load(response.choices[0]["text"])
    for key in data:
        val = data[key]
        if ":" not in val:
            continue
        data[key] = tmp = {}
        for x in val.split():
            x = x.split(":", 1)
            tmp[x[0]] = x[1]
    return function, data


def write_wrapper_to_file(wkdir, function, string):
    LOCAL_DIR = wkdir / Path(function.upper())
    LOCAL_DIR.mkdir(exist_ok=True)
    with open(LOCAL_DIR / Path(function.upper() + ".py"), "w") as fh:
        # we need to redo the doc string so that it amtches the origianl in the same
        # conventions as the other autogenned nodes
        # First, lets get the OG string
        og_docstring = getattr(np.random, function).__doc__
        og_explanation = og_docstring[og_docstring.find("Parameters") :]
        head = string[: string.find("Parameters")]
        tail = string[find_2nd(string, '"""') :]
        flojoy_disclaimer = (
            "-." * 36
            + "\nThe parameters of the function in this Flojoy wrapper are given below."
            + "\n"
            + "-." * 36
            + "\n"
        )
        string = head + "\n" + flojoy_disclaimer + og_explanation + "\n\n" + tail
        string = string.replace("\t\t", "\t")
        fh.write(
            f"import numpy as np\nfrom flojoy import flojoy, DataContainer\nfrom typing import Optional, Union, Tuple, List\n@flojoy\n{string}"
        )
    subprocess.call(
        ["black", f"{LOCAL_DIR/Path(function.upper()+'.py')}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generate_nodes", "-G", dest="generate_nodes", action="store_true"
    )
    parser.add_argument(
        "--generate_manifests", "-M", dest="generate_manifest", action="store_true"
    )

    args = parser.parse_args()

    DIRECTORY = Path("RANDOM")
    DIRECTORY.mkdir(exist_ok=True)

    MANIFEST_DIRECTORY = DIRECTORY / Path("../../MANIFEST")

    if args.generate_nodes:
        DATA_FNAME = "numpy.random.json"

        WRAPPERS = {}
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_stuff = [
                executor.submit(generate_wrapper_davinci, function)
                for function in primary_functions
            ]
            for future in tqdm(
                as_completed(future_to_stuff), total=len(primary_functions)
            ):
                res = future.result()
                WRAPPERS[res[0]] = res[1]
                if "def" in res[1] and "return" in res[1]:
                    write_wrapper_to_file(DIRECTORY, res[0], res[1])
        with open(DATA_FNAME, "w", encoding="utf-8") as f:
            json.dump(WRAPPERS, f, ensure_ascii=False, indent=4)

    if args.generate_manifest:
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_stuff = [
                executor.submit(generate_manifest, function)
                for function in primary_functions
            ]
            for future in tqdm(
                as_completed(future_to_stuff), total=len(primary_functions)
            ):
                res = future.result()
                func, MANIFEST = res
                MANIFEST["name"] = func
                MANIFEST["key"] = func.upper()
                MANIFEST["type"] = "NUMPY_RANDOM"
                if "size" in MANIFEST["parameters"]:
                    MANIFEST["parameters"]["size"] = {
                        "default": "dc[0].y.shape",
                        "type": "string",
                    }
                with open(
                    MANIFEST_DIRECTORY / Path(func.lower() + ".manifest.yaml"), "w"
                ) as fh:
                    yaml.safe_dump({"COMMAND": MANIFEST}, fh, default_flow_style=False)

    # for function in primary_functions:
    #     node_exists = os.path.exists(DIRECTORY/Path(function.upper())/Path(function.upper()+".py"))
    #     # print(f'{function} included: {node_exists}')
    #     if not node_exists:
    #         os.remove(MANIFEST_DIRECTORY/Path(function.lower()+'.manifest.yaml'))
    #         print('Deleted manifest of', function)
