import PIL
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflitert_interpreter
import python_speech_features
import sounddevice as sd
import soundfile as sf
import scipy.signal


# np.set_printoptions(suppress=True)


class TestAudio():
    def __init__(self, wav_path, model_path):
        # self.mean = [128., 128., 128.]
        # self.scale = [0.0078125, 0.0078125, 0.0078125]
        self.wav_path = wav_path
        self.model_path = model_path
        self.delegate_options = {'platform': 'J7',
                                 'version': '8.2',
                                 'tidl_tools_path': '/home/zifeng/reclone/r8.2/edgeai-benchmark/tidl_tools',
                                 'artifacts_folder': '/home/zifeng/reclone/r8.2/edgeai-benchmark/work_dirs/modelartifacts/8bits/cl-1998_tflitert_imagenet1k_mlperf_wake_word_stop_model_complexV6_tflite/artifacts',
                                 'tensor_bits': 32, 'import': 'yes', 'accuracy_level': 1, 'debug_level': 0,
                                 'priority': 0,
                                 'advanced_options:high_resolution_optimization': 0,
                                 'advanced_options:pre_batchnorm_fold': 1,
                                 'advanced_options:calibration_frames': 50,
                                 'advanced_options:calibration_iterations': 5,
                                 'advanced_options:quantization_scale_type': 0,
                                 'advanced_options:activation_clipping': 0,
                                 'advanced_options:weight_clipping': 0, 'advanced_options:bias_calibration': 0,
                                 'advanced_options:channel_wise_quantization': 0,
                                 'advanced_options:output_feature_16bit_names_list': '',
                                 'advanced_options:params_16bit_names_list': '',
                                 'advanced_options:add_data_convert_ops': 3,
                                 'deny_list': '22,9,25'}

        self.tidl_delegate = [tflitert_interpreter.load_delegate('/home/zifeng/reclone/r8.2/edgeai-benchmark/tidl_tools/libtidl_tfl_delegate.so', self.delegate_options)]
        self.interpreter = tflitert_interpreter.Interpreter(self.model_path, \
                                                            experimental_delegates=self.tidl_delegate)
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def decimate(self, signal, sample_rate, resample_rate):

        # Check to make sure we're downsampling
        if resample_rate > sample_rate:
            print("Error: target sample rate higher than original")
            return signal, sample_rate

        dec_factor = sample_rate / resample_rate
        if not dec_factor.is_integer():
            print("Error: can only decimate by integer factor")
            return signal, sample_rate

        # Do decimation
        resampled_signal = scipy.signal.decimate(signal, int(dec_factor))

        print('\033[93m' + 'resample_rate' + '\033[0m')
        print(resampled_signal)
        return resampled_signal, resample_rate

    def read_wav(self, wav_path):
        wav_data, fs = sf.read(wav_path)
        # print(wav_data.dtype)
        # print(wav_data.shape)
        # wav_data = np.round(wav_data, decimals=7)
        print('\033[93m' + 'wav_data' + '\033[0m')
        print(wav_data)
        print(type(wav_data))

        return wav_data, fs

    def wav_mfcc(self, wav_data, sample_rate):
        # resample_wav_data, fs = self.decimate(wav_data, sample_rate, resample_rate=8000)
        # wav_data = np.float32(wav_data)
        # print(wav_data.dtype)

        mfccs = python_speech_features.base.mfcc(wav_data,
                                                 samplerate=sample_rate,
                                                 winlen=0.256,
                                                 winstep=0.050,
                                                 numcep=16,
                                                 nfilt=26,
                                                 nfft=2048,
                                                 preemph=0.0,
                                                 ceplifter=0,
                                                 appendEnergy=False,
                                                 winfunc=np.hanning)
        # print('\033[93m' + 'mfccs not transpose' + '\033[0m')
        #
        # print(mfccs)
        mfccs = mfccs.transpose()  # 转置

        # mfccs_zero = np.zeros((16, 16), dtype=np.float32)
        # print(mfccs_zero)
        # # print(type(mfccs))
        #
        # mfccs1 = mfccs[:8, :]
        # mfccs1 = np.round(mfccs1, decimals=2)
        # mfccs1 = np.float32(mfccs1)

        # mfccs2 = mfccs[8:, :]
        # mfccs2 = np.round(mfccs2, decimals=2)
        # mfccs2 = np.float32(mfccs2)

        # print(mfccs1)
        # print(mfccs1.dtype)
        # print("========================================")
        # print(mfccs2)
        # print(mfccs2.dtype)
        # mfccs = mfccs_zero + mfccs

        # mfccs = np.concatenate((mfccs1, mfccs2), axis=0)
        mfccs = np.pad(mfccs, ((8, 8), (8, 8)), "constant", constant_values=0)
        mfccs = np.round(mfccs, decimals=6)
        print('\033[93m' + 'mfccs' + '\033[0m')
        # print(mfccs)

        mfccs = np.float32(mfccs.reshape((1, mfccs.shape[0], mfccs.shape[1], 1)))
        # mfccs = np.repeat(mfccs, 3, axis=3)
        # print(mfccs)
        # print(mfccs.shape)
        # print(mfccs.dtype)

        return mfccs

    def nor(self, wav_data):
        # mean = [[[[0.]]]]
        # scale = [[[[1.]]]]
        # mean = [[[[-2.9545634, -2.9545634, -2.9545634]]]]
        # scale = [[[[0.0825400802, 0.0825400802, 0.0825400802]]]]
        # mean = [[[[-0.73864084]]]]
        # scale = [[[[0.16151726587153076]]]]
        # mean = np.array(mean)
        # scale = np.array(scale)
        mean = [[[[-2.9545634]]]]
        scale = [[[[0.0825400802]]]]

        wav_data = (wav_data - mean) * scale
        wav_data = np.array(wav_data, dtype=np.float32)
        # print(wav_data)
        print(wav_data.dtype)
        return wav_data

    def tensor_pad(self, mfccs):
        mfccs = np.pad(mfccs, ((0, 0), (8, 8), (8, 8), (0, 0)), "constant", constant_values=0)
        return mfccs

    def invoke(self, wav_data):
        self.interpreter.allocate_tensors()
        self.interpreter.set_tensor(self.input_details[0]['index'], wav_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        max = np.argmax(output_data)
        print(max)
        print(output_data[0][max])
        print(output_data[0])
        print(output_data.dtype)

    def infer(self):
        wav_data, fs = self.read_wav(self.wav_path)
        resample_wav_data, fs = self.decimate(wav_data, fs, 8000)

        # print(resample_wav_data)
        wav_data = self.wav_mfcc(resample_wav_data, fs)
        wav_data = self.nor(wav_data)

        # wav_data = np.round(wav_data, decimals=1)
        # print(wav_data)
        print(wav_data.shape)
        print(wav_data.dtype)
        print(type(wav_data))
        # print(wav_data)
        print(wav_data[0][0][0])
        self.invoke(wav_data)


model_path = '/home/zifeng/reclone/r8.2/edgeai-modelzoo/modelartifacts/8bits/cl-1998_tflitert_imagenet1k_mlperf_wake_word_stop_model_complexV6_tflite/model/wake_word_stop_model_complexV6.tflite'
wav_path = '/home/zifeng/reclone/r8.2/edgeai-benchmark/dependencies/datasets/wavdataset/val/34/backward010.wav'
a = TestAudio(wav_path=wav_path, model_path=model_path)
a.infer()
