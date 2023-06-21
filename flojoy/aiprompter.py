import openai
import os
import numpy as np
from tqdm import tqdm
import json 

primary_functions = [
    func_name for func_name in dir(np.random)
    if callable(getattr(np.random, func_name)) and not func_name.startswith("_")
]

print(primary_functions)

openai.api_key = os.environ["OPENAI_KEY"]

WRAPPERS = {}
for function in tqdm(primary_functions):
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

    response_retval= openai.Completion.create(
        model="text-davinci-003",
        prompt=f"Change the following Python3.10 function to return a custom class 'DataContainer', instantiated with the arguments 'x=dc[0].y, y=<out>', where <out> is the return value of the function: {response.choices[0]['text']}",
        max_tokens=2048,  # Adjust as per your requirements
        # n=1,  # Number of completions to generate
        top_p=1,
        temperature=0.0,  # Controls randomness of the output
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    WRAPPERS[function] = response_retval.choices[0]["text"]
    # print(WRAPPERS[function])
    # print(response.choices[0]['text'])
print(WRAPPERS)
with open('numpy.random.json', 'w', encoding='utf-8') as f:
    json.dump(WRAPPERS, f, ensure_ascii=False, indent=4)