import argparse
import numpy as np
import random
from PIL import Image

action_list = [[0, 1], [0, -1], [1, 0], [-1, 0], [1, 1], [1, -1], [-1, 1], [-1, -1]]
scale = 2


def random_walk(canvas, ini_x, ini_y, length):
    x = ini_x
    y = ini_y
    img_size = canvas.shape[-1]
    x_list = []
    y_list = []
    for i in range(length):
        r = random.randint(0, len(action_list) - 1)
        x = np.clip(x + action_list[r][0], a_min=0, a_max=int(img_size/scale) - 1)
        y = np.clip(y + action_list[r][1], a_min=0, a_max=int(img_size/scale) - 1)
        x_list.append(x)
        y_list.append(y)
    ##x_list += x_list
    x_li = [xx*scale for xx in x_list]
    y_li = [yy*scale for yy in y_list]
    x_li_1 = [i+1 for i in x_li]
    y_li_1 = [j+1 for j in y_li]
    print(max(x_li_1))
    print(max(y_li_1))
    print("lwn x",len(x_li))

    canvas[np.array(x_li), np.array(y_li)] = 0
    canvas[np.array(x_li_1), np.array(y_li)] = 0
    canvas[np.array(x_li), np.array(y_li_1)] = 0
    canvas[np.array(x_li_1), np.array(y_li_1)] = 0
    return canvas


if __name__ == '__main__':
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--N', type=int, default=1000)
    parser.add_argument('--save_dir', type=str, default='mask')
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    for i in range(args.N):
        canvas = np.ones((args.image_size, args.image_size)).astype("i")
        ini_x = random.randint(0, args.image_size/scale - 1)
        ini_y = random.randint(0, args.image_size/scale - 1)
        mask = random_walk(canvas, ini_x, ini_y, int((args.image_size/scale)) ** 2)
        print("save:", i, np.sum(mask))

        img = Image.fromarray(mask * 255).convert('1')
        img.save('{:s}/{:06d}.jpg'.format(args.save_dir, i))
