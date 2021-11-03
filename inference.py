#NOTE:  This must be the first call in order to work properly!

from deoldify import device
from deoldify.device_id import DeviceId
#choices:  CPU, GPU0...GPU7
device.set(device=DeviceId.GPU0)

import argparse
parser = argparse.ArgumentParser(description='Inference code to colorize and fix old photos')
parser.add_argument('--artistic', type=bool, help='use artistic mode', default=False)
parser.add_argument('--render_factor', type=int, help='render scale', default=35)
parser.add_argument('--input', type=str, help='input image name', default='test_images/image.png')
parser.add_argument('--output', type=str, help='output image name', default='result_images/output.png')

args = parser.parse_args()

from deoldify.visualize import *
plt.style.use('dark_background')
torch.backends.cudnn.benchmark=True
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*?Your .*? set is empty.*?")

colorizer = get_image_colorizer(artistic=args.artistic)

#NOTE:  Max is 45 with 11GB video cards. 35 is a good default
# render_factor=args.render_factor
#NOTE:  Make source_url None to just read from file at ./video/source/[file_name] directly without modification
# source_path = args.source_path

result = colorizer.plot_transformed_image(path=args.input, results_dir=None, render_factor=args.render_factor, compare=False)
# 输出在result_imgae/image.png
# print(result)
# show_image_in_notebook(result_path)
# from PIL import Image
# Image.save(result, 'result_images/output.png')
