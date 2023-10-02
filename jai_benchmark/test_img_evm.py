import PIL
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflitert_interpreter



class TestImg():
    def __init__(self, img_path, model_path):
        self.mean = [128., 128., 128.]
        self.scale = [0.0078125, 0.0078125, 0.0078125]
        self.img_path = img_path
        self.model_path = model_path
        self.delegate_options = {
            "tidl_tools_path": "null",
            "artifacts_folder": '/home/zifeng/reclone/r8.2/edgeai-modelzoo/modelartifacts/8bits/temp/cl-0000_tflitert_imagenet1k_mlperf_mobilenet_v1_1.0_224_tflite/artifacts',
            "import": 'no'
        }
        self.tidl_delegate = [tflitert_interpreter.load_delegate('/home/zifeng/reclone/r8.2/edgeai-benchmark/tidl_tools/libtidl_tfl_delegate.so', self.delegate_options)]
        self.interpreter = tflitert_interpreter.Interpreter(self.model_path,\
                                           experimental_delegates=self.tidl_delegate)
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()



    def read_img(self, img_path):
        img = PIL.Image.open(img_path)
        img = img.convert('RGB')
        img = img.resize((224, 224))
        # img.show()
        img_data = np.array(img, dtype=np.float64)
        print(img_data)

        img_data = np.reshape(img_data, (1, img_data.shape[0], img_data.shape[1], img_data.shape[2]))
        print(img_data.dtype)
        print(img_data.shape)
        return img_data

    def nor(self, img_data):
        mean = [[[[128., 128., 128.]]]]
        scale = [[[[0.0078125, 0.0078125, 0.0078125]]]]
        # mean = np.array(mean, dtype=np.float64)
        # scale = np.array(scale, dtype=np.float64)
        print(img_data)

        img_data = (img_data - mean) * scale
        img_data = np.array(img_data, dtype=np.float32)
        print(img_data)
        print(img_data.dtype)
        print("+++++++++++++++")
        print(img_data[0][0][0][0])
        print("+++++++++++++++")

        return img_data


    def invoke(self, img_data):
        self.interpreter.allocate_tensors()
        self.interpreter.set_tensor(self.input_details[0]['index'], img_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        max = np.argmax(output_data)
        print(max)
        print(output_data[0][max])
        print(output_data[0][:5])
        print(output_data.dtype)


    def infer(self):
        image_data = self.read_img(self.img_path)
        image_data = self.nor(image_data)
        print(type(image_data))
        self.invoke(image_data)

model_path = '/home/zifeng/reclone/r8.2/edgeai-modelzoo/modelartifacts/8bits/temp/cl-0000_tflitert_imagenet1k_mlperf_mobilenet_v1_1.0_224_tflite/model/mobilenet_v1_1.0_224.tflite'
img_path = '/home/zifeng/reclone/r8.2/edgeai-benchmark/dependencies/datasets/imagenetv2c/val/870/e3a2b04a266380018920fdd2e5f7fc4564860334.jpeg'
a = TestImg(img_path=img_path, model_path=model_path)
# a.infer()