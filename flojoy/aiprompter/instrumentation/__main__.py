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

openai.api_key = os.environ["OPENAI_KEY"]

response_retval = openai.Completion.create(
    model="text-davinci-003",
    prompt=f"Write an instrument driver in Python 3.10 for the Agilent 34400A using the QCodes library",
    max_tokens=2048,  # Adjust as per your requirements
    # n=1,  # Number of completions to generate
    top_p=1,
    temperature=0.0,  # Controls randomness of the output
    frequency_penalty=0.0,
    presence_penalty=0.0,
)
print(response_retval.choices[0]["text"])



def load_generation_pipe(model_name_or_path: str, gpu_device: int=0):
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)


    pipe = pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer,
        use_fast = False,
        device="cpu" # Have no CUDA installed https://discuss.huggingface.co/t/is-transformers-using-gpu-by-default/8500
    )

    print("load generation pipeline from {} over, vocab size = {}, eos id = {}, gpu device = {}.".format(
        model_name_or_path, len(tokenizer), tokenizer.eos_token_id, gpu_device)
    )

    return pipe

def extract_function_block(string):
    return re.split("\nclass|\ndef|\n#|\n@|\nprint|\nif", string)[0].rstrip()

def run_code_generation(pipe, prompt, num_completions=1, **gen_kwargs):
    set_seed(123)

    code_gens = pipe(prompt,
        num_return_sequences=num_completions,
        **gen_kwargs
    )

    return [extract_function_block(code_gen["generated_text"][len(prompt):]) for code_gen in code_gens]

pipe = load_generation_pipe("Daoguang/PyCodeGPT", 0)


gen_kwargs = {
    "do_sample": True,
    "temperature": 0.8,
    "max_new_tokens": 500,
    "top_p": 1.0,
    "top_k": 0,
    "pad_token_id": pipe.tokenizer.pad_token_id if pipe.tokenizer.pad_token_id else pipe.tokenizer.eos_token_id,
    "eos_token_id": pipe.tokenizer.eos_token_id
}
prompt = "How to make an instrument driver in Python 3.10 for the Agilent 34400A with QCodes?" #prompt requires question syntax for some reason
[print(code_gen) for code_gen in run_code_generation(pipe, prompt, num_completions=1, **gen_kwargs)]
