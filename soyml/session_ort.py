from typing import List, Dict, Any, Optional
import numpy as np
import onnx
import onnxruntime as ort


def select_best_providers(blacklist: List[str] = []):
    all_providers = ort.get_available_providers()
    # if log:
    #     log.debug(f"all providers: {all_providers}")

    accel_providers = [
        "CUDAExecutionProvider",
        "ROCMExecutionProvider",
        "CoreMLExecutionProvider",
        "DirectMLExecutionProvider",
        "CPUExecutionProvider",
    ]

    # remove blacklisted providers
    accel_providers = [p for p in accel_providers if p not in blacklist]

    # go through providers in order of preference, and if one is available, use it
    for provider in accel_providers:
        if provider in all_providers:
            return [provider]


def session_ort_init(
    self, use_cpu_only: bool = False, provider_blacklist: List[str] = []
):
    log = self.log.logger_for("session_ort")
    best_execution_providers = select_best_providers(blacklist=provider_blacklist)
    try:
        log.debug(
            f"creating ort inference session with providers: {best_execution_providers}"
        )
        # if use cpu only, then only use CPUExecutionProvider
        if use_cpu_only:
            self.ort_session = ort.InferenceSession(
                self.ort_model_file,
                providers=["CPUExecutionProvider"],
            )
        # otherwise, we can use the best available compute provider
        else:
            self.ort_session = ort.InferenceSession(
                self.ort_model_file, providers=best_execution_providers
            )
        self.inputs = self.ort_session.get_inputs()
        self.outputs = self.ort_session.get_outputs()
        self.input_shapes = {}
        self.output_shapes = {}
        for input in self.inputs:
            self.input_shapes[input.name] = input.shape
        for output in self.outputs:
            self.output_shapes[output.name] = output.shape

        inputs_str = [
            f"{input.name}@{input.type}{input.shape}" for input in self.inputs
        ]
        outputs_str = [
            f"{output.name}@{output.type}{output.shape}" for output in self.outputs
        ]
        log.debug(f"  ort session inputs: {inputs_str}")
        log.debug(f"  ort session outputs: {outputs_str}")
    except Exception as e:
        raise Exception(f"failed to load ort model: {e}")


def session_ort_execute(self, inputs, output_names):
    log = self.log.logger_for("session_ort")
    try:
        outputs = self.ort_session.run(output_names, inputs)
    except Exception as e:
        raise Exception(f"failed to execute ort: {e}")

    return outputs
