from deoldify import device
from deoldify.device_id import DeviceId
from time import sleep
import matplotlib.pyplot as plt
import json
import torch
import os
#choices:  CPU, GPU0...GPU7
device.set(device=DeviceId.GPU0)

import argparse
parser = argparse.ArgumentParser(description='Inference code to colorize and fix old photos')
parser.add_argument('--artistic', type=bool, help='use artistic mode', default=False)
parser.add_argument('--render_factor', type=int, help='render scale', default=35)
parser.add_argument('--input', type=str, help='input image name', default='image.png')
parser.add_argument('--output', type=str, help='output image name', default='output.png')

args = parser.parse_args()


message_json = "/workspace/go_proj/src/Ai_WebServer/algorithm_utils/colorImage/message.json"
user_img_dir = "/workspace/go_proj/src/Ai_WebServer/static/algorithm/colorImage/user_imgs"
res_img_dir = "/workspace/go_proj/src/Ai_WebServer/static/algorithm/colorImage/res_imgs"
# message_json = "./message.json"
# user_img_dir = "./test_images"  # 老照片存放的文件夹
# res_img_dir = "./result_images"  # 输出文件夹


def set_args(msg):
    args.input = msg['user_img']  # 输入的老照片
    args.render_factor = msg["render_factor"] # 上色比例，越大越鲜艳（更占显存）
    args.artistic = msg["artistic"]   # 是否采用artistic模式（相当于换模型，用0或1）
    args.output = "deoldify_"+str(msg["render_factor"])+str(1 if msg["artistic"] else 0)+"_"+msg['user_img']


if __name__ == '__main__':
    last_msg = {}
    from deoldify.visualize import *

    plt.style.use('dark_background')
    torch.backends.cudnn.benchmark = True

    colorizer = get_image_colorizer(artistic=False)

    while True:
        try:
            with open(message_json, "r", encoding="utf-8") as f:
                message = json.load(f)
        except Exception as e:
            print(e)
            sleep(1)
            message = {}
            continue
        if message == last_msg:
            print("deoldify wait...")
            sleep(1)
            continue
        else:
            message_art = message['artistic']
            change = message_art != args.artistic
            if change:
                colorizer = get_image_colorizer(artistic=args.artistic)
        set_args(message)
        image_file = os.path.join(user_img_dir, args.input)
        output_file = os.path.join(res_img_dir, args.output)
        if os.path.exists(output_file):
            print("deoldify exist...")
            continue
        print('input:', image_file, ', output:', output_file)
        try:
            result = colorizer.plot_transformed_image(path=image_file, results_dir=Path(output_file), render_factor=args.render_factor,
                                                  compare=False)
            print('output success in:', output_file)
        except Exception as e:
            print(e)
            print('colorized error!')
        # print(type(result))
        # plt.imshow(result)
        # plt.show()
        last_msg = message
