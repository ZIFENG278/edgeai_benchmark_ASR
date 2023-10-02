import os
import random
import shutil

folder_path = './dependencies/datasets/wav_dataset/val/'

tags = ['cat', 'tree', 'three', 'zero', 'go', 'on', 'follow', 'seven', 'forward', 'yes', 'nine', 'visual', 'off',
        'left', 'eight', 'stop', 'no', 'bird', 'up', 'wow', 'two', 'marvin', 'six', 'four', 'five', 'right', 'one',
        'dog', 'house', 'happy', 'bed', 'learn', 'sheila', 'down', 'backward']

origin_dataset_path = '/home/zifeng/tflite-speech-recognition-224/data_speech_commands_v0.02/'

# for i in range(35):
#     os.makedirs(folder_path+str(i))
# random.seed(100)
# for i, v in enumerate(tags):
#     sum_dataset_path = origin_dataset_path + v
#     sum_file_name = os.listdir(sum_dataset_path)
#     random.shuffle(sum_file_name)
#     file_name = sum_file_name[:15]
#     for j in file_name:
#      shutil.copy(origin_dataset_path+v + '/' + j, folder_path+str(i))




vals = os.listdir(folder_path)

# for i in vals:
#     filenames = os.listdir(folder_path + i)
#     print(filenames)
#     for index, filename in enumerate(filenames):
#         oldname = folder_path + i + '/' + filename
#         newname = folder_path + i + '/' + tags[int(i)] + str(index).rjust(3, '0') + '.wav'
#         os.rename(oldname, newname)


# for i in vals:
#     filenames = os.listdir(folder_path + i)
#     print(len(filenames))
#     if len(filenames) > 10:
#         more_filenames = len(filenames) - 10
#         for delet_file in filenames[-more_filenames:]:
#             os.remove(folder_path + i + '/' + delet_file)

with open('./dependencies/datasets/wav_dataset/val.txt', 'w') as f:
    strr = ""
    for i in range(35):
        filenames = os.listdir(folder_path + str(i))
        for j in range(10):
            strr += str(i) + '/' + filenames[j] + ' ' + str(i) + '\n'

    print(strr)
    f.write(strr)


