import math
import os
import random

import cv2 as cv
import numpy as np
import torch
from torchvision import transforms

from config import device, im_size, fg_path_test, a_path_test, bg_path_test
from data_gen import data_transforms, gen_trimap, fg_test_files, bg_test_files
from utils import compute_mse, compute_sad, ensure_folder, safe_crop, draw_str
from test import gen_test_names, process_test


def composite4(fg, bg, a, w, h):
    fg = np.array(fg, np.float32)
    bg_h, bg_w = bg.shape[:2]
    x = 0
    if bg_w > w:
        x = np.random.randint(0, bg_w - w)
    y = 0
    if bg_h > h:
        y = np.random.randint(0, bg_h - h)
    bg = np.array(bg[y:y + h, x:x + w], np.float32)
    alpha = np.zeros((h, w, 1), np.float32)
    alpha[:, :, 0] = a
    im = alpha * fg + (1 - alpha) * bg
    im = im.astype(np.uint8)
    return im, bg


if __name__ == '__main__':
    checkpoint = 'BEST_checkpoint.tar'
    checkpoint = torch.load(checkpoint)
    model = checkpoint['model']
    model = model.to(device)
    model.eval()

    transformer = data_transforms['valid']

    ensure_folder('images')

    names = gen_test_names()
    names = random.sample(names, 10)

    for i, name in enumerate(names):
        fcount = int(name.split('.')[0].split('_')[0])
        bcount = int(name.split('.')[0].split('_')[1])
        im_name = fg_test_files[fcount]
        bg_name = bg_test_files[bcount]
        img, alpha, fg, bg = process_test(im_name, bg_name)

        print('\nStart processing image: {}'.format(name))

        h, w = img.shape[:2]

        trimap = gen_trimap(alpha)

        x = torch.zeros((1, 4, h, w), dtype=torch.float)
        img = img[..., ::-1]  # RGB
        img = transforms.ToPILImage()(img)  # [3, 320, 320]
        img = transformer(img)  # [3, 320, 320]
        x[0:, 0:3, :, :] = img
        x[0:, 3, :, :] = torch.from_numpy(trimap.copy()) / 255.

        # Move to GPU, if available
        x = x.type(torch.FloatTensor).to(device)  # [1, 4, 320, 320]
        alpha = alpha / 255.

        with torch.no_grad():
            pred = model(x)  # [1, 4, 320, 320]

        pred = pred.cpu().numpy()
        pred = pred.reshape((h, w))  # [320, 320]

        pred[trimap == 0] = 0.0
        pred[trimap == 255] = 1.0

        # Calculate loss
        # loss = criterion(alpha_out, alpha_label)
        mse_loss = compute_mse(pred, alpha, trimap)
        sad_loss = compute_sad(pred, alpha)
        str_msg = 'sad: %.4f, mse: %.4f' % (sad_loss, mse_loss)
        print(str_msg)

        out = (pred.copy() * 255).astype(np.uint8)
        draw_str(out, (10, 20), str_msg)
        cv.imwrite('images/{}_out.png'.format(i), out)

        # sample_bg = sample_bgs[i]
        # bg = cv.imread(os.path.join(bg_test, sample_bg))
        # bh, bw = bg.shape[:2]
        # wratio = im_size / bw
        # hratio = im_size / bh
        # ratio = wratio if wratio > hratio else hratio
        # if ratio > 1:
        #     bg = cv.resize(src=bg, dsize=(math.ceil(bw * ratio), math.ceil(bh * ratio)), interpolation=cv.INTER_CUBIC)
        # im, bg = composite4(bgr_img, bg, y_pred, im_size, im_size)
        # cv.imwrite('images/{}_compose.png'.format(i), im)
        # cv.imwrite('images/{}_new_bg.png'.format(i), bg)
