import argparse
import numpy as np
from time import time

from transformers import GPT2Tokenizer, OPTForCausalLM


def cli_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--device', required=False, default="cpu",
                         help=f"Optional. Device for inference")

    args = parser.parse_args()

    return args


def load_model(device):
    tokenizer = GPT2Tokenizer.from_pretrained('facebook/opt-66b')
    model = OPTForCausalLM.from_pretrained('facebook/opt-66b')
    model.to(device)

    return model, tokenizer


def generate(input, model, tokenizer, device):
    t0 = time()
    inputs = tokenizer(input, return_tensors='pt')
    generate_ids = model.generate(inputs.input_ids.to(device), max_length=80)
    result = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    generation_time = time() - t0

    return result, generation_time


def main():
    args = cli_argument_parser()

    model, tokenizer = load_model(args.device)
    inference_time = []
    while True:
        text_input = input('Enter text or "stop" to exit')
        if text_input == 'stop':
            break

        result, time = generate(text_input, model, tokenizer, args.device)
        inference_time.append(time)
        print(result)

    avg_time = np.average(inference_time)
    print(f'Average inference time: {avg_time} seconds')


if __name__ == '__main__':
    main()