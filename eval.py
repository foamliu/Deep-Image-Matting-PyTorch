import os

import cv2 as cv
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

from config import device
from data_gen import data_transforms
from utils import ensure_folder

IMG_FOLDER = 'data/alphamatting/input_lowres'
TRIMAP_FOLDERS = ['data/alphamatting/trimap_lowres/Trimap1', 'data/alphamatting/trimap_lowres/Trimap2',
                  'data/alphamatting/trimap_lowres/Trimap3']
OUTPUT_FOLDERS = ['images/alphamatting/output_lowres/Trimap1', 'images/alphamatting/output_lowres/Trimap2', 'images/alphamatting/output_lowres/Trimap3', ]

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
        h, w = img.shape[:2]

        out_file = os.path.join('images/alphamatting', file)
        cv.imwrite(out_file, img)

        x = torch.zeros((1, 4, h, w), dtype=torch.float)
        image = img[..., ::-1]  # RGB
        image = transforms.ToPILImage()(image)
        image = transformer(image)
        x[0:, 0:3, :, :] = image

        for i in range(3):
            trimap = cv.imread(os.path.join(TRIMAP_FOLDERS[i], file), 0)
            x[0:, 3, :, :] = torch.from_numpy(trimap.copy()) / 255.

            # Move to GPU, if available
            x = x.type(torch.FloatTensor).to(device)

            with torch.no_grad():
                pred = model(x)

            pred = pred.cpu().numpy()
            pred = pred.reshape((h, w))

            pred[trimap == 0] = 0.0
            pred[trimap == 255] = 1.0

            out = (pred.copy() * 255).astype(np.uint8)

            filename = os.path.join(OUTPUT_FOLDERS[i], file)
            cv.imwrite(filename, out)
