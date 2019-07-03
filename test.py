import torch

from config import device, im_size, print_freq
from data_gen import DIMDataset
from data_gen import data_transforms
from utils import compute_mse_loss, compute_sad_loss, AverageMeter, get_logger

if __name__ == '__main__':
    checkpoint = 'BEST_checkpoint.tar'
    checkpoint = torch.load(checkpoint)
    model = checkpoint['model']
    model = model.to(device)
    model.eval()

    transformer = data_transforms['valid']

    valid_dataset = DIMDataset('valid')
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=2, shuffle=False, num_workers=8)

    mse_losses = AverageMeter()
    sad_losses = AverageMeter()

    logger = get_logger()

    for i, (img, alpha_label) in enumerate(valid_loader):
        # Move to GPU, if available
        img = img.type(torch.FloatTensor).to(device)  # [N, 3, 320, 320]
        alpha_label = alpha_label.type(torch.FloatTensor).to(device)  # [N, 320, 320]
        alpha_label = alpha_label.reshape((-1, 2, im_size * im_size))  # [N, 320*320]

        # Forward prop.
        alpha_out = model(img)  # [N, 320, 320]
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
