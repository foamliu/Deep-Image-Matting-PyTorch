import os
import random

import cv2 as cv
import torch
from torchvision import transforms

from config import device, im_size, print_freq
from data_gen import data_transforms, process, generate_trimap, random_choice
from utils import compute_mse_loss, compute_sad_loss, AverageMeter, get_logger
from utils import safe_crop

with open('data/Combined_Dataset/Test_set/test_fg_names.txt') as f:
    fg_test_files = f.read().splitlines()
with open('data/Combined_Dataset/Test_set/test_bg_names.txt') as f:
    bg_test_files = f.read().splitlines()


def gen_names():
    num_fgs = 50
    num_bgs = 1000
    num_bgs_per_fg = 20

    names = []
    bcount = 0
    for fcount in range(num_fgs):
        for i in range(num_bgs_per_fg):
            names.append(str(fcount) + '_' + str(bcount) + '.png')
            bcount += 1

    return names


def get_alpha_test(name):
    fg_i = int(name.split("_")[0])
    name = fg_test_files[fg_i]
    filename = os.path.join('data/mask_test', name)
    alpha = cv.imread(filename, 0)
    return alpha


if __name__ == '__main__':
    checkpoint = 'BEST_checkpoint.tar'
    checkpoint = torch.load(checkpoint)
    model = checkpoint['model']
    model = model.to(device)
    model.eval()

    transformer = data_transforms['valid']

    names = gen_names()

    mse_losses = AverageMeter()
    sad_losses = AverageMeter()

    logger = get_logger()

    for i, name in enumerate(names):
        fcount = int(name.split('.')[0].split('_')[0])
        bcount = int(name.split('.')[0].split('_')[1])
        im_name = fg_test_files[fcount]
        bg_name = bg_test_files[bcount]
        img, alpha, fg, bg = process(im_name, bg_name)

        # crop size 320:640:480 = 1:1:1
        different_sizes = [(320, 320), (480, 480), (640, 640)]
        crop_size = random.choice(different_sizes)

        trimap = generate_trimap(alpha)
        x, y = random_choice(trimap, crop_size)
        img = safe_crop(img, x, y, crop_size)
        alpha = safe_crop(alpha, x, y, crop_size)

        trimap = generate_trimap(alpha)

        x = torch.zeros((4, im_size, im_size), dtype=torch.float)
        img = transforms.ToPILImage()(img)
        img = transformer(img)
        x[0:3, :, :] = img
        x[3, :, :] = torch.from_numpy(trimap.copy()) / 255.

        img = torch.from_numpy(x)

        # Move to GPU, if available
        img = img.type(torch.FloatTensor).to(device)  # [N, 3, 320, 320]
        alpha_label = alpha_label.type(torch.FloatTensor).to(device)  # [N, 320, 320]
        alpha_label = alpha_label.reshape((-1, 2, im_size * im_size))  # [N, 320*320]

        # Forward prop.
        alpha_out = model(img)  # [N, 320, 320]
        print(alpha_out.size())
        alpha_out = alpha_out.reshape((-1, 1, im_size * im_size))  # [N, 320*320]

        # Calculate loss
        # loss = criterion(alpha_out, alpha_label)
        mse_loss = compute_mse_loss(alpha_out, alpha_label)
        sad_loss = compute_sad_loss(alpha_out, alpha_label)

        # Keep track of metrics
        mse_losses.update(mse_loss.item())
        sad_losses.update(sad_loss.item())

        if i % print_freq == 0:
            status = '[{0}/{1}]\t' \
                     'MSE Loss {mse_loss.val:.4f} ({mse_loss.avg:.4f})\t' \
                     'SAD Loss {sad_loss.val:.4f} ({sad_loss.avg:.4f})\t'.format(i, len(valid_loader),
                                                                                 mse_loss=mse_losses,
                                                                                 sad_loss=sad_losses)
            logger.info(status)

    print("sad:{} mse:{}".format(sad_losses.avg, mse_losses.avg))
