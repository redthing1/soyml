import os

from .backends import SoyMLBackend
from .session import SoyMLSession
from minlog import logger


class SoyMLLoader(object):
    def __init__(self, model_dir: str, log=None, use_cpu_only=False):
        self.model_dir = model_dir
        self.log = log.logger_for("soyml_loader") if log else logger
        self.use_cpu_only = use_cpu_only

    """
    Load a model session from model file in the model directory.

    :param model_basename: the base name of the model file (without extension/suffix)
    :param backend: the inference backend to use for the model
    :param device_blacklist: a list of devices to blacklist for inference of this model
    """

    def load_session(self, model_basename, backend, device_blacklist=[]):
        self.ensure_model_files_exist(model_basename, backend)
        model_fnames = self.get_model_file_names(model_basename, backend)

        self.log.debug(f"loading model files: {model_fnames}")

        if backend == SoyMLBackend.ONNXRUNTIME:
            return SoyMLSession(
                log=self.log,
                use_ort=True,
                ort_use_cpu_only=self.use_cpu_only,
                ort_provider_blacklist=device_blacklist,
                ort_model_file=model_fnames[0],
            )
        if backend == SoyMLBackend.NCNN:
            return SoyMLSession(
                log=self.log,
                use_ncnn=True,
                ncnn_param_file=model_fnames[0],
                ncnn_model_file=model_fnames[1],
            )
        if backend == SoyMLBackend.WONNX:
            return SoyMLSession(
                log=self.log,
                use_wonnx=True,
                wonnx_model_file=model_fnames[0],
            )
        if backend == SoyMLBackend.TORCH:
            return SoyMLSession(
                log=self.log,
                use_torch=True,
                torch_model_file=model_fnames[0],
            )

    def get_model_file_names(self, model_basename, backend):
        model_basename = os.path.join(self.model_dir, model_basename)
        if backend == SoyMLBackend.ONNXRUNTIME:
            return [f"{model_basename}.sim.onnx"]
        if backend == SoyMLBackend.NCNN:
            return [f"{model_basename}.ncnn.param", f"{model_basename}.ncnn.bin"]
        if backend == SoyMLBackend.WONNX:
            return [f"{model_basename}.web.onnx"]
        if backend == SoyMLBackend.TORCH:
            return [f"{model_basename}.pt"]
        return []

    def ensure_model_files_exist(self, model_basename, backend):
        for model_file_name in self.get_model_file_names(model_basename, backend):
            if not os.path.exists(model_file_name):
                raise FileNotFoundError(f"model file {model_file_name} does not exist")
