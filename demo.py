import math
import os
import random

import cv2 as cv
import numpy as np
import torch

from config import device, im_size
from data_gen import data_transforms, generate_trimap, random_choice, get_alpha_test
from utils import compute_mse_loss, compute_sad_loss, ensure_folder, get_final_output, safe_crop, draw_str


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
    alpha[:, :, 0] = a / 255.
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

    out_test_path = 'data/merged_test/'
    test_images = [f for f in os.listdir(out_test_path) if
                   os.path.isfile(os.path.join(out_test_path, f)) and f.endswith('.png')]
    samples = random.sample(test_images, 10)

    bg_test = 'data/bg_test/'
    test_bgs = [f for f in os.listdir(bg_test) if
                os.path.isfile(os.path.join(bg_test, f)) and f.endswith('.jpg')]
    sample_bgs = random.sample(test_bgs, 10)

    total_loss = 0.0
    for i in range(len(samples)):
        filename = samples[i]
        image_name = filename.split('.')[0]

        print('\nStart processing image: {}'.format(filename))

        bgr_img = cv.imread(os.path.join(out_test_path, filename))
        bg_h, bg_w = bgr_img.shape[:2]
        print('bg_h, bg_w: ' + str((bg_h, bg_w)))

        a = get_alpha_test(image_name)
        a_h, a_w = a.shape[:2]
        print('a_h, a_w: ' + str((a_h, a_w)))

        alpha = np.zeros((bg_h, bg_w), np.float32)
        alpha[0:a_h, 0:a_w] = a
        trimap = generate_trimap(alpha)
        different_sizes = [(320, 320), (320, 320), (320, 320), (480, 480), (640, 640)]
        crop_size = random.choice(different_sizes)
        x, y = random_choice(trimap, crop_size)
        print('x, y: ' + str((x, y)))

        bgr_img = safe_crop(bgr_img, x, y, crop_size)
        alpha = safe_crop(alpha, x, y, crop_size)
        trimap = safe_crop(trimap, x, y, crop_size)
        cv.imwrite('images/{}_image.png'.format(i), np.array(bgr_img).astype(np.uint8))
        cv.imwrite('images/{}_trimap.png'.format(i), np.array(trimap).astype(np.uint8))
        cv.imwrite('images/{}_alpha.png'.format(i), np.array(alpha).astype(np.uint8))

        img = bgr_img[..., ::-1]  # RGB
        img = np.transpose(img, (2, 0, 1))

        x_test = np.empty((1, 4, im_size, im_size), dtype=np.float32)
        x_test[0, 0:3, :, :] = img / 255.
        x_test[0, 3, :, :] = trimap / 255.

        x_test = torch.from_numpy(x_test).type(torch.FloatTensor).to(device)  # [1, 4, 320, 320]

        with torch.no_grad():
            y_pred = model(x_test)

        y_pred = y_pred.cpu().numpy()
        print('y_pred.shape: ' + str(y_pred.shape))
        y_pred = np.reshape(y_pred, (im_size, im_size))
        print(y_pred.shape)
        y_pred = y_pred * 255.0
        y_pred = get_final_output(y_pred, trimap)
        y_pred = y_pred.astype(np.uint8)

        sad_loss = compute_sad_loss(y_pred, alpha, trimap)
        mse_loss = compute_mse_loss(y_pred, alpha, trimap)
        str_msg = 'sad_loss: %.4f, mse_loss: %.4f, crop_size: %s' % (sad_loss, mse_loss, str(crop_size))
        print(str_msg)

        out = y_pred.copy()
        draw_str(out, (10, 20), str_msg)
        cv.imwrite('images/{}_out.png'.format(i), out)

        sample_bg = sample_bgs[i]
        bg = cv.imread(os.path.join(bg_test, sample_bg))
        bh, bw = bg.shape[:2]
        wratio = im_size / bw
        hratio = im_size / bh
        ratio = wratio if wratio > hratio else hratio
        if ratio > 1:
            bg = cv.resize(src=bg, dsize=(math.ceil(bw * ratio), math.ceil(bh * ratio)), interpolation=cv.INTER_CUBIC)
        im, bg = composite4(bgr_img, bg, y_pred, im_size, im_size)
        cv.imwrite('images/{}_compose.png'.format(i), im)
        cv.imwrite('images/{}_new_bg.png'.format(i), bg)
