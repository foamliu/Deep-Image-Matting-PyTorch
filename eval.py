import os

import cv2 as cv
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

from config import device, im_size
from data_gen import data_transforms
from utils import ensure_folder, get_final_output

IMG_FOLDER = 'data/alphamatting/'
TRIMAP_FOLDERS = ['data/alphamatting/Trimap1', 'data/alphamatting/Trimap2',
                  'data/alphamatting/Trimap3']
OUTPUT_FOLDERS = ['images/alphamatting/Trimap1', 'images/alphamatting/Trimap2', 'images/alphamatting/Trimap3', ]

if __name__ == '__main__':
    checkpoint = 'BEST_checkpoint.tar'
    checkpoint = torch.load(checkpoint)
    model = checkpoint['model']
    model = model.to(device)
    model.eval()

    transformer = data_transforms['valid']

    ensure_folder('images')
    ensure_folder('images/alphamatting')
    ensure_folder(OUTPUT_FOLDERS[0])
    ensure_folder(OUTPUT_FOLDERS[1])
    ensure_folder(OUTPUT_FOLDERS[2])

    files = [f for f in os.listdir(IMG_FOLDER) if f.endswith('.png')]

    for file in tqdm(files):
        filename = os.path.join(IMG_FOLDER, file)
        img = cv.imread(filename)
        img = cv.resize(img, (im_size, im_size))

        for i in range(3):
            trimap = cv.imread(os.path.join(TRIMAP_FOLDERS[i], file), 0)
            trimap = cv.resize(trimap, (im_size, im_size), cv.INTER_NEAREST)

            x_test = torch.zeros((1, 4, im_size, im_size), dtype=torch.float)
            img = transforms.ToPILImage()(img)
            img = transformer(img)
            x_test[0:, 0:3, :, :] = img
            x_test[0:, 3, :, :] = torch.from_numpy(trimap.copy()) / 255.

            print(x_test.size())

            with torch.no_grad():
                y_pred = model(x_test)

            y_pred = y_pred.cpu().numpy()
            y_pred = y_pred * 255.
            y_pred = np.reshape(y_pred, (im_size, im_size))
            y_pred = get_final_output(y_pred, trimap)
            y_pred = y_pred.astype(np.uint8)

            filename = os.path.join(OUTPUT_FOLDERS[i], file)
            cv.imwrite('images/alphamatting/{}.png'.format(file), y_pred)
