from .backends import SoyMLBackend


class SoyMLSession(object):
    def __init__(
        self,
        log=None,
        use_ort=False,
        ort_model_file=None,
        use_ncnn=False,
        ncnn_param_file=None,
        ncnn_model_file=None,
        use_wonnx=False,
        wonnx_model_file=None,
        use_torch=False,
        torch_model_file=None,
    ):
        self.log = log.logger_for("soyml_session") if log else None

        self.backend = SoyMLBackend.UNKNOWN
        if use_ort:
            self.backend = SoyMLBackend.ONNXRUNTIME
            self.ort_model_file = ort_model_file
            # load the model
            from .session_ort import session_ort_init

            session_ort_init(self)
        if use_ncnn:
            self.backend = SoyMLBackend.NCNN
            self.ncnn_param_file = ncnn_param_file
            self.ncnn_model_file = ncnn_model_file
            # load the model
            from .session_ncnn import session_ncnn_init

            session_ncnn_init(self)

        if use_wonnx:
            self.backend = SoyMLBackend.WONNX
            self.wonnx_model_file = wonnx_model_file
            # load the model
            from .session_wonnx import session_wonnx_init

            session_wonnx_init(self)

        if use_torch:
            self.backend = SoyMLBackend.TORCH
            self.torch_model_file = torch_model_file
            # load the model
            from .session_torch import session_torch_init

            session_torch_init(self)

        if self.log:
            self.log.trace(f"initialized session with backend {self.backend}")

    def execute(self, inputs, output_names):
        # # ensure there's exactly one output (for now)
        # if len(output_names) != 1:
        #     raise Exception(f"expected exactly one output, but got {len(output_names)}")

        # execute the model

        if self.backend == SoyMLBackend.ONNXRUNTIME:
            from .session_ort import session_ort_execute

            return session_ort_execute(self, inputs, output_names)
        if self.backend == SoyMLBackend.NCNN:
            from .session_ncnn import session_ncnn_execute

            return session_ncnn_execute(self, inputs, output_names)
        if self.backend == SoyMLBackend.WONNX:
            from .session_wonnx import session_wonnx_execute

            return session_wonnx_execute(self, inputs, output_names)
        if self.backend == SoyMLBackend.TORCH:
            from .session_torch import session_torch_execute

            return session_torch_execute(self, inputs, output_names)
