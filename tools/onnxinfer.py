from typing import List, Optional, Union
from dataclasses import dataclass, field

import numpy as np
import onnx
import onnxruntime
from loguru import logger


# from .uitls import timer

@dataclass
class ONNXIO:  # noqa
    name: str = ''
    type: str = ''
    shape: List[int] = field(default_factory=list)  # [n,c,h,w]

    def __repr__(self):
        return f'Name: {self.name}\nType: {self.type}\nShape: {self.shape}'


class ONNXInfer:
    def __init__(
        self,
        model_path: Optional[str] = '',
        device: Optional[str] = 'GPU',
        auto_load: bool = False,
        auto_warmup: bool = False
    ) -> None:
        self._device = device
        self._gpu_is_available = False
        self._model_path = model_path
        self._model = None
        self._onnx_session = None
        self._inputs = []
        self._outputs = []

        if auto_load:
            self.load()

        if auto_warmup:
            self.warmup()

    @property
    def gpu_is_available(self) -> bool:
        return self._gpu_is_available

    @property
    def model_path(self) -> str:
        return self._model_path

    @model_path.setter
    def model_path(self, value: str) -> None:
        self._model_path = value

    def load(self, path: Optional[str] = None) -> None:

        if path is not None:
            self._model_path = path

        curr_device = onnxruntime.get_device()
        if curr_device == 'GPU':
            self._gpu_is_available = True

        logger.info(f'Curr Support ONNXRunTime Device: {curr_device}')
        logger.info(f'Available Providers: {onnxruntime.get_available_providers()}')

        # Load ONNX Model
        self._model = onnx.load(self._model_path)
        onnx.checker.check_model(self._model)
        logger.info(f'Loading ONNX Model: {self._model_path}.')

        # Build ONNXRunTime Session
        providers = ['CPUExecutionProvider']
        if self._device.lower() == 'gpu':
            if self._gpu_is_available:
                # providers.append('TensorrtExecutionProvider')
                providers.insert(0, 'CUDAExecutionProvider')
            else:
                logger.warning('Can`t use GPU ONNXRuntime.')

        logger.info(f'Curr Providers: {providers}')
        self._onnx_session = onnxruntime.InferenceSession(self._model_path, providers=providers)

        self._decode_input()
        self._decode_output()

    def _decode_input(self) -> None:
        inputs = self._onnx_session.get_inputs()
        self._inputs = []
        for item in inputs:
            self._inputs.append(ONNXIO(item.name, item.type, item.shape))
            if item.shape[0] is None:
                self._is_dynamic = True
        logger.info('Decode Input')

    def _decode_output(self) -> None:
        outputs = self._onnx_session.get_outputs()
        self._outputs = []
        for item in outputs:
            self._outputs.append(ONNXIO(item.name, item.type, item.shape))
        logger.info('Decode Output')

    @property
    def inputs(self) -> List[ONNXIO]:
        return self._inputs

    @property
    def outputs(self) -> List[ONNXIO]:
        return self._outputs

    def get_input(self, idx: int) -> ONNXIO:
        return self._inputs[idx]

    def get_output(self, idx: int) -> ONNXIO:
        return self._outputs[idx]

    # @timer
    def warmup(self, times: Optional[int] = 10) -> None:
        logger.info(f'Begin Warmup Model x{times}.')
        input_shape = self.get_input(0).shape
        data = np.random.rand(*input_shape).astype(np.float32)  # noqa
        for _ in range(times):
            self._run_impl(data)
        logger.info('Warmup Model Done.')

    def _run_impl(self, data: np.ndarray) -> Union[list, None]:  # noqa
        input_name = self.get_input(0).name
        onnx_input = {input_name: data}
        outputs = self._onnx_session.run(None, onnx_input)
        return outputs

    # @timer
    def run(self, data: np.ndarray) -> Union[list, None]:  # noqa
        if isinstance(type(data), np.ndarray):
            logger.error('Input data type must be np.ndarray.')
            return None

        if list(data.shape) != self.get_input(0).shape:
            logger.error(f'{data.shape}!={self.get_input(0).shape}')
            return None

        return self._run_impl(data)


if __name__ == '__main__':
    root = r'D:\llf\code\amat\temp\models\20240516_160741_danyang_F_1_3_256_256_CLS0_SEG6_Static.onnx'
    onnxinfer = ONNXInfer(root, auto_load=True, auto_warmup=True)

    data = np.random.rand(1, 3, 256, 256).astype(np.float32)
    ret = onnxinfer.run(data)
    print(f'Output: {ret}')
