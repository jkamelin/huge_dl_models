import argparse
import numpy as np
import torch
from time import perf_counter
from transformers import AutoTokenizer, AutoModelForMaskedLM

MODEL_PATH = {
    'ESM-1B': 'facebook/esm1b_t33_650M_UR50S',
    'ESM2': 'facebook/esm2_t48_15B_UR50D',
}

def cli_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', required=True,
                         help=f"Model name")

    args = parser.parse_args()

    return args

def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForMaskedLM.from_pretrained(model_path, torch_dtype=torch.float32)

    return model, tokenizer

def generate(input, model, tokenizer):
    t0 = perf_counter()
    inputs = tokenizer(input, return_tensors='pt')
    outputs = model(inputs.input_ids).logits.detach().numpy()
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
        model, tokenizer = load_model(MODEL_PATH[args.model])
        result, single_gen_time = generate(input_, model, tokenizer)
        print(f'Masked symbol: {result}. Generation time: {single_gen_time} sec.')
        inference_time.append(single_gen_time)

    print(f'Average generation time: {np.average(inference_time)} sec.')

if __name__ == '__main__':
    main()
