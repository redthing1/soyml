import numpy as np
import onnx
import onnxruntime as ort
from minlog import logger


def select_best_providers():
    all_providers = ort.get_available_providers()
    logger.debug(f"[session_ort] all providers: {all_providers}")

    accel_providers = [
        "CUDAExecutionProvider",
        "ROCMExecutionProvider",
        "CoreMLExecutionProvider",
        "DirectMLExecutionProvider",
        "CPUExecutionProvider",
    ]

    # go through providers in order of preference, and if one is available, use it
    for provider in accel_providers:
        if provider in all_providers:
            return [provider]


def session_ort_init(self):
    try:
        execution_providers = select_best_providers()
        logger.debug(f"[session_ort] execution providers: {execution_providers}")
        self.ort_session = ort.InferenceSession(
            self.ort_model_file, providers=execution_providers
        )
        self.inputs = self.ort_session.get_inputs()
        self.outputs = self.ort_session.get_outputs()
        self.input_shapes = {}
        self.output_shapes = {}
        for input in self.inputs:
            self.input_shapes[input.name] = input.shape
        for output in self.outputs:
            self.output_shapes[output.name] = output.shape
    except Exception as e:
        raise Exception(f"failed to load ort model: {e}")


def session_ort_execute(self, inputs, output_names):
    logger.debug(f"[session_ort] execute: {inputs.keys()} -> {output_names}")
    try:
        outputs = self.ort_session.run(output_names, inputs)
    except Exception as e:
        raise Exception(f"failed to execute ort: {e}")

    return outputs
