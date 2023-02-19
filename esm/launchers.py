# Copyright (C) 2023 KNS Group LLC (YADRO)
# SPDX-License-Identifier: Apache-2.0


import logging as log
import onnxruntime as ort
from openvino.runtime import Core, get_version, PartialShape, Dimension
import numpy as np
from typing import Any
import torch
import transformers

from provider import ClassProvider

MODEL_TO_URL = {
    'ESM-1B': 'facebook/esm1b_t33_650M_UR50S',
    'ESM2': 'facebook/esm2_t48_15B_UR50D',
}


class BaseLauncher(ClassProvider):
    __provider_type__ = "launcher"
    def __init__(self, model_path: str) -> None:
        """
        Load model using a model_path

        :param model_path
        """
        pass

    def process(self, input_ids: np.array) -> Any:
        """
        Run launcher with user's input

        :param input_ids
        """
        pass


class PyTorchLauncher(BaseLauncher):
    __provider__ = "pytorch"
    def __init__(self, model_path: str) -> None:
        log.info('PyTorch Runtime')
        self.model = transformers.AutoModelForMaskedLM.from_pretrained(model_path, torch_dtype=torch.float32)

    def process(self, inputs) -> Any:
        generated_ids = self.model(**inputs)
        return generated_ids.logits.detach().numpy()


class ONNXLauncher(BaseLauncher):
    __provider__ = "onnx"
    def __init__(self, model_path: str) -> None:
        log.info('ONNX Runtime')
        self.session = ort.InferenceSession(model_path)

    def process(self, inputs) -> Any:
        ort_inputs = {}
        for name in inputs.keys():
            ort_inputs[name] = inputs[name].numpy()
        outputs = self.session.run(["last_hidden_state"], ort_inputs)
        return outputs[0]


class OpenVINOLaucnher(BaseLauncher):
    __provider__ = "openvino"
    def __init__(self, model_path: str) -> None:
        log.info('OpenVINO Runtime')
        super().__init__(model_path)
        core = Core()
        self.model = core.read_model(model_path)
        self.input_tensor = self.model.inputs[0].any_name
        self.model.reshape({self.input_tensor: PartialShape([Dimension(1), Dimension(0, 1024)])})

        # load model to the device
        self.compiled_model = core.compile_model(self.model, "CPU")
        self.output_tensor = self.compiled_model.outputs[0]
        self.infer_request = self.compiled_model.create_infer_request()

    def process(self, inputsy) -> Any:
        # infer by OpenVINO runtime
        outputs = self.infer_request.infer(inputs)[self.output_tensor]
        return outputs


def create_launcher(laucnher_name: str, model_path: str):
    return BaseLauncher.provide(laucnher_name, model_path)
