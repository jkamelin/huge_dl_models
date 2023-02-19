import argparse
import numpy as np
import torch
from time import perf_counter
from transformers import AutoTokenizer

from launchers import create_launcher

MODEL_URL = {
    'ESM-1B': 'facebook/esm1b_t33_650M_UR50S',
    'ESM2': 'facebook/esm2_t48_15B_UR50D',
}

def cli_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', required=True,
                         help=f"Model name")
    parser.add_argument('--model_path', required=True,
                         help=f"Model path")
    parser.add_argument('--launcher', required=False, default='pytorch',
                         help=f"Launcher name")

    args = parser.parse_args()

    return args

def load_model(launcher_name, model_path, model_name):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_URL[model_name])
    launcher = create_launcher(launcher_name, model_path)

    return launcher, tokenizer

def generate(input, launcher, tokenizer):
    t0 = perf_counter()
    inputs = tokenizer(input, return_tensors='pt')
    outputs = launcher.process(inputs)
    mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    predicted_token_id = outputs[0, mask_token_index].argmax(axis=-1)
    predicted_sym = tokenizer.decode(predicted_token_id)
    generation_time = perf_counter() - t0

    return predicted_sym, generation_time

def main():
    args = cli_argument_parser()
    inputs = [
        "MKTVRQERLKSIVRILERSKEPVSGAQ<mask>AEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
        "KALTARQQEVFDLIRDHISQTGMP<mask>TRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE",
        "KALTARQQEVFDLIRD<mask>ISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE",
    ]

    inference_time = []
    for input_ in inputs:
        launcher, tokenizer = load_model(args.launcher, args.model_path, args.model_name)
        result, single_gen_time = generate(input_, launcher, tokenizer)
        print(f'Masked symbol: {result}. Generation time: {single_gen_time} sec.')
        inference_time.append(single_gen_time)

    print(f'Average generation time: {np.average(inference_time)} sec.')

if __name__ == '__main__':
    main()
