import argparse
import onnx
import torch
import torch.onnx

from transformers import GPT2Tokenizer, OPTForCausalLM
from dalle_mini import DalleBart, DalleBartProcessor

MODEL_MAP = {
            'opt': {'model': OPTForCausalLM, 'tokenizer': GPT2Tokenizer},
            'dalle': {'model': DalleBart, 'tokenizer': DalleBartProcessor},
            }


def cli_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', required=True, default='opt',
                         help='Model name')
    parser.add_argument('--model_path', required=True, default='facebook/opt-66b',
                         help='Path to model')
    parser.add_argument('--input_names', required=True, type=str,
                         help='Model input names')
    parser.add_argument('--output_file', required=True, type=str,
                         help='Output file')

    args = parser.parse_args()

    return args



def export_to_onnx(model, tokenizer, output_file, model_path, input_names):
    tokenizer_ = tokenizer.from_pretrained(model_path)
    model_ = model.from_pretrained(model_path)

    inputs = tokenizer_("Hello, my dog is cute", return_tensors="pt")

    inp = (inputs[name] for name in input_names)
    print(inp)

    model_.eval()
    torch.onnx.export(model_, inp, output_file, input_names=input_names)
    model_ = onnx.load(output_file)
    onnx.checker.check_model(model_)


def main():
    args = cli_argument_parser()
    export_to_onnx(**MODEL_MAP[args.model], output_file=args.output_file,
                   model_path=args.model_path, input_names=args.input_names)
