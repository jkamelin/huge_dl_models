import argparse
from transformers.onnx import export, OnnxConfig
from transformers import AutoConfig, AutoTokenizer, AutoModelForMaskedLM
from typing import Mapping, OrderedDict
from pathlib import Path

MODEL_URL = {
    'ESM-1B': 'facebook/esm1b_t33_650M_UR50S',
    'ESM2': 'facebook/esm2_t48_15B_UR50D',
}

class ESMOnnxConfig(OnnxConfig):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                ("input_ids", {0: "batch", 1: "sequence"}),
                ("attention_mask", {0: "batch", 1: "sequence"}),
            ]
        )

def cli_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str, required=True,
                         help=f"Model name")
    parser.add_argument('--output_path', type=Path, required=True,
                         help=f"Path to onnx file")

    args = parser.parse_args()

    return args

def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_URL[model_name])
    model = AutoModelForMaskedLM.from_pretrained(MODEL_URL[model_name])

    return model, tokenizer

def convert(model, tokenizer, model_name,  output_path):
    config = AutoConfig.from_pretrained(MODEL_URL[model_name])
    onnx_config = ESMOnnxConfig(config)

    onnx_inputs, onnx_outputs = export(tokenizer, model, onnx_config, onnx_config.default_onnx_opset, output_path)

    return onnx_inputs, onnx_outputs

def main():
    args = cli_argument_parser()

    model, tokenizer = load_model(args.model_name)
    onnx_inputs, onnx_outputs = convert(model, tokenizer, args.model_name, args.output_path)

    print(f'Input names: {onnx_inputs}\nOutput names: {onnx_outputs}')

if __name__ == '__main__':
    main()
