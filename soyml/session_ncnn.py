import numpy as np

from ncnn_vulkan import ncnn
from minlog import logger


def session_ncnn_init(self):
    log = self.log.logger_for("session_ncnn")
    try:
        self.ncnn_net = ncnn.Net()
        self.ncnn_net.opt.use_vulkan_compute = True
        log.debug(f"loading ncnn model: {self.ncnn_model_file}")
        self.ncnn_net.load_param(self.ncnn_param_file)
        self.ncnn_net.load_model(self.ncnn_model_file)
    except Exception as e:
        ncnn.destroy_gpu_instance()  # cleanup
        raise Exception(f"failed to load ncnn model: {e}")


def session_ncnn_execute(self, inputs, output_names):
    log = self.log.logger_for("session_ncnn")
    log.debug(f"execute: {inputs.keys()} -> {output_names}")
    try:
        extractor = self.ncnn_net.create_extractor()

        # for each input, we need to call ex.input
        for input_key, input_value in inputs.items():
            # print("input shape:", input_value.shape)
            input_mat = ncnn.Mat(input_value)
            # print("input_mat shape:", input_mat.shape)
            extractor.input(input_key, input_mat)

        # # let's for now, expect only one output
        # output0_key = output_names[0]
        # ret, output0 = extractor.extract(output0_key)
        # outputs = [output0]

        # collect outputs into a list
        outputs_list = []
        for output_key in output_names:
            ret, output = extractor.extract(output_key)
            outputs_list.append(output)

        if ret != 0:
            raise Exception(f"failed to execute ncnn: {ret}")
    except Exception as e:
        ncnn.destroy_gpu_instance()  # cleanup
        raise Exception(f"failed to execute ncnn: {e}")

    return np.array(outputs_list)
