import os

from .backends import SoyMLBackend
from .session import SoyMLSession


class SoyMLLoader(object):
    def __init__(self, model_dir: str):
        self.model_dir = model_dir

    def load_session(self, model_basename, backend):
        self.ensure_model_files_exist(model_basename, backend)
        model_fnames = self.get_model_file_names(model_basename, backend)
        if backend == SoyMLBackend.ONNXRUNTIME:
            return SoyMLSession(
                use_ort=True,
                ort_model_file=model_fnames[0],
            )
        if backend == SoyMLBackend.NCNN:
            return SoyMLSession(
                use_ncnn=True,
                ncnn_param_file=model_fnames[0],
                ncnn_bin_file=model_fnames[1],
            )
        if backend == SoyMLBackend.WONNX:
            return SoyMLSession(
                use_wonnx=True,
                wonnx_model_file=model_fnames[0],
            )
        if backend == SoyMLBackend.TORCH:
            return SoyMLSession(
                use_torch=True,
                torch_model_file=model_fnames[0],
            )

    def get_model_file_names(self, model_basename, backend):
        model_basename = os.path.join(self.model_dir, model_basename)
        if backend == SoyMLBackend.ONNXRUNTIME:
            return [f"{model_basename}.sim.onnx"]
        if backend == SoyMLBackend.NCNN:
            return [f"{model_basename}.ncnn.param", f"{model_basename}.bin"]
        if backend == SoyMLBackend.WONNX:
            return [f"{model_basename}.web.onnx"]
        if backend == SoyMLBackend.TORCH:
            return [f"{model_basename}.pt"]
        return []

    def ensure_model_files_exist(self, model_basename, backend):
        for model_file_name in self.get_model_file_names(model_basename, backend):
            if not os.path.exists(model_file_name):
                raise FileNotFoundError(f"model file {model_file_name} does not exist")
