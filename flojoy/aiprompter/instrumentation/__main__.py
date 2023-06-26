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
import re
from transformers import pipeline, set_seed
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.pipelines.base import Pipeline
import json
openai.api_key = os.environ["OPENAI_KEY"]
from tenacity import retry, wait_random_exponential, stop_after_attempt

def prompt_pycodegpt(prompt, model_params):
    def load_generation_pipe(model_name_or_path: str, gpu_device: int = 0):
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            use_fast=False,
            device="cpu",  # Have no CUDA installed https://discuss.huggingface.co/t/is-transformers-using-gpu-by-default/8500
        )

        print(
            "load generation pipeline from {} over, vocab size = {}, eos id = {}, gpu device = {}.".format(
                model_name_or_path, len(tokenizer), tokenizer.eos_token_id, gpu_device
            )
        )

        return pipe

    def extract_function_block(string):
        return re.split("\nclass|\ndef|\n#|\n@|\nprint|\nif", string)[0].rstrip()

    def run_code_generation(pipe, prompt, num_completions=1, **gen_kwargs):
        set_seed(123)

        code_gens = pipe(prompt, num_return_sequences=num_completions, **gen_kwargs)

        return [
            extract_function_block(code_gen["generated_text"][len(prompt) :])
            for code_gen in code_gens
        ]

    print("#" * 72 + "\nPyCodeGPT prompting\n" + "#" * 72)
    pipe = load_generation_pipe("Daoguang/PyCodeGPT", 0)
    gen_kwargs = {
        "pad_token_id": pipe.tokenizer.pad_token_id
        if pipe.tokenizer.pad_token_id
        else pipe.tokenizer.eos_token_id,
        "eos_token_id": pipe.tokenizer.eos_token_id,
    }
    gen_kwargs.update(model_params)
    [
        print(code_gen)
        for code_gen in run_code_generation(
            pipe, prompt, num_completions=1, **gen_kwargs
        )
    ]

@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3), after=lambda retry_state:print(f'Attempt: {retry_state.attempt_number}'))
def prompt_gpt35(prompt, model_params, experimental=False):
    print(
        "#" * 72
        + f'\nGPT-3.5-Turbo prompting {"with experimental run" if experimental else ""}\n'
        + "#" * 72
    )
    if not experimental:
        response_retval = openai.Completion.create(
            model="text-davinci-003", prompt=prompt, **model_params
        )
        print(response_retval.choices[0]["text"])

    else:
        messages = [
            {
                "role": "user",
                "content": "Generate an instrument driver for the Agilent 34400A using the QCodes library",
            }
        ]
        functions = [
            {
                "name": "get_driver_information",
                "description": "Prints the driver in a human readable format",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "instrument_name": {
                            "type": "string",
                            "description": "The name of the instrument",
                        },
                        "description": {
                            "type": "string",
                            "description": "A description of the instrument",
                        },
                        "set_methods": {
                            "type": "string",
                            "description": "A comma separated list of driver methods that can set parameter values on the instrument beginning with 'set_'",
                        },
                        "get_methods": {
                            "type": "string",
                            "description": "A comma separated list of driver methods that can get parameter values on the instrument beginning with 'get_",
                        },
                    },
                    "required": [
                        "instrument_name",
                        "description",
                        "set_methods",
                        "get_methods",
                    ],
                },
            }
        ]
        response_retval = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=messages,
            functions=functions,
            function_call="auto",  # auto is default, but we'll be explicit,
        )
        print(
            "Driver info:",
            response_retval["choices"][0]["message"]["function_call"]["arguments"],
        )
        # print(pretty_print_conversation(response_retval['choices'][0]['message']))
        function_args = json.loads(response_retval["choices"][0]["message"]["function_call"]["arguments"])
        # generate the set methods
        COMPLETE_FUNC = []
        for setter in function_args.get('get_methods').split(","):
            head, tail = setter.split('_')[:2]
            messages = [
                {
                    "role": "user",
                    "content": f"Generate a Python3.10 function to {head} the {tail} on the {function_args.get('instrument_name')} using the QCodes library and VisaInstrument class of QCodes",
                }
            ]
            # print(messages[0]['content'])
            response_retval = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0613",
                messages=messages,
                **model_params
            )
            result = response_retval["choices"][0]["message"]['content']
            if 'import VisaInstrument' not in result:
                raise TypeError("Incorrect response returned.")
            # ===============================================================
            # Now we need to process the fragements into one single class ...
            # ===============================================================
            # first, we need to only get the python code delimited in markdown syntax
            fragment = result[result.find('```'):result.find("```", result.find("```") + 1)]
            fragment = fragment[:fragment.find('# Example usage')]
            # now, we remove the leading ```python line
            fragment = "\n".join(fragment.split('\n')[1:])
            COMPLETE_FUNC.append(fragment)
            # print(fragment)
        headers = []
        functionality = []
        for fragment in COMPLETE_FUNC:
            for idl, line in enumerate(fragment.split('\n')):
                if line.startswith('from') or line.startswith('import'):
                    headers.append(line)
                else: 
                    functionality.append(line)
        print("\n".join(set(headers)) + "\n".join(functionality))



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-P", "--pycodegpt", dest="pycodegpt", action="store_true")
    parser.add_argument("-G", "--gpt35", dest="gpt35", action="store_true")
    parser.add_argument(
        "-E", "--experimental", dest="experimental", action="store_true"
    )
    args = parser.parse_args()
    if args.gpt35:
        model_params = {
            "max_tokens": 2048,  # Adjust as per your requirements
            # "n": 1,  # Number of completions to generate
            "top_p": 1,
            "temperature": 0.0,  # Controls randomness of the output
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        }
        prompt_gpt35(
            prompt=f"Write an instrument driver in Python 3.10 for the Agilent 34400A using the QCodes library",
            model_params=model_params,
            experimental=args.experimental,
        )
    if args.pycodegpt:
        model_params = {
            "do_sample": True,
            "temperature": 0.8,
            "max_new_tokens": 500,
            "top_p": 1.0,
            "top_k": 0,
        }
        prompt_pycodegpt(
            prompt="How to make an instrument driver in Python 3.10 for the Agilent 34400A with QCodes?",  # prompt requires question syntax for some reason
            model_params=model_params,
        )
