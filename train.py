import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn

from config import device, grad_clip, print_freq
from data_gen import DIMDataset
from models import DIMModel
from utils import parse_args, save_checkpoint, AverageMeter, clip_gradient, get_logger, accuracy, adjust_learning_rate


def train_net(args):
    torch.manual_seed(7)
    np.random.seed(7)
    checkpoint = args.checkpoint
    start_epoch = 0
    best_loss = float('inf')
    writer = SummaryWriter()
    epochs_since_improvement = 0

    # Initialize / load checkpoint
    if checkpoint is None:
        model = DIMModel()
        model = nn.DataParallel(model)

        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD([{'params': model.parameters()}],
                                        lr=args.lr, momentum=args.mom, weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.Adam([{'params': model.parameters()}],
                                         lr=args.lr, weight_decay=args.weight_decay)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    logger = get_logger()

    # Move to GPU, if available
    model = model.to(device)

    # Loss function
    L1Loss = nn.L1Loss().to(device)
    CrossEntropyLoss = nn.CrossEntropyLoss().to(device)

    # Custom dataloaders
    train_dataset = DIMDataset('train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    valid_dataset = DIMDataset('valid')
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=0.1)

    # Epochs
    for epoch in range(start_epoch, args.end_epoch):
        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(optimizer, 0.8)

        # One epoch's training
        train_loss = train(train_loader=train_loader,
                           model=model,
                           criterions=(L1Loss, CrossEntropyLoss),
                           optimizer=optimizer,
                           epoch=epoch,
                           logger=logger)

        writer.add_scalar('Train_Loss', train_loss, epoch)

        # One epoch's validation
        valid_loss = valid(valid_loader=valid_loader,
                           model=model,
                           criterions=(L1Loss, CrossEntropyLoss),
                           logger=logger)

        writer.add_scalar('Valid_Loss', valid_loss, epoch)

        # Check if there was an improvement
        is_best = valid_loss < best_loss
        best_loss = min(valid_loss, best_loss)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(epoch, epochs_since_improvement, model, optimizer, best_loss, is_best)


def train(train_loader, model, criterions, optimizer, epoch, logger):
    model.train()  # train mode (dropout and batchnorm is used)

    losses = AverageMeter()

    reg_losses = AverageMeter()
    expression_accs = AverageMeter()
    gender_accs = AverageMeter()
    glasses_accs = AverageMeter()
    race_accs = AverageMeter()

    L1Loss, CrossEntropyLoss = criterions

    # Batches
    for i, (img, reg, expression, gender, glasses, race) in enumerate(train_loader):
        # Move to GPU, if available
        img = img.to(device)
        reg_label = reg.type(torch.FloatTensor).to(device)  # [N, 5]
        expression_label = expression.type(torch.LongTensor).to(device)  # [N, 3]
        gender_label = gender.type(torch.LongTensor).to(device)  # [N, 2]
        glasses_label = glasses.type(torch.LongTensor).to(device)  # [N, 3]
        race_label = race.type(torch.LongTensor).to(device)  # [N, 4]

        # Forward prop.
        reg_out, expression_out, gender_out, glasses_out, race_out = model(img)  # embedding => [N, 17]

        # Calculate loss
        reg_loss = L1Loss(reg_out, reg_label)
        expression_loss = CrossEntropyLoss(expression_out, expression_label)
        gender_loss = CrossEntropyLoss(gender_out, gender_label)
        glasses_loss = CrossEntropyLoss(glasses_out, glasses_label)
        race_loss = CrossEntropyLoss(race_out, race_label)

        loss = reg_loss + expression_loss + gender_loss + glasses_loss + race_loss

        # Back prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        clip_gradient(optimizer, grad_clip)

        # Update weights
        optimizer.step()

        # Keep track of metrics
        losses.update(loss.item())

        reg_losses.update(reg_loss.item())
        expression_accuracy = accuracy(expression_out, expression_label)
        expression_accs.update(expression_accuracy)
        gender_accuracy = accuracy(gender_out, gender_label)
        gender_accs.update(gender_accuracy)
        glasses_accuracy = accuracy(glasses_out, glasses_label)
        glasses_accs.update(glasses_accuracy)
        race_accuracy = accuracy(race_out, race_label)
        race_accs.update(race_accuracy)

        # Print status

        if i % print_freq == 0:
            status = 'Epoch: [{0}][{1}/{2}]\t' \
                     'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                     'Reg Loss {reg_loss.val:.4f} ({reg_loss.avg:.4f})\t' \
                     'Expression Accuracy {expression_acc.val:.4f} ({expression_acc.avg:.4f})\t' \
                     'Gender Accuracy {gender_acc.val:.4f} ({gender_acc.avg:.4f})\t' \
                     'Glasses Accuracy {expression_acc.val:.4f} ({expression_acc.avg:.4f})\t' \
                     'Race Accuracy {expression_acc.val:.4f} ({expression_acc.avg:.4f})\t'.format(epoch, i,
                                                                                                  len(train_loader),
                                                                                                  loss=losses,
                                                                                                  reg_loss=reg_losses,
                                                                                                  expression_acc=expression_accs,
                                                                                                  gender_acc=gender_accs,
                                                                                                  glasses_acc=glasses_accs,
                                                                                                  race_acc=race_accs)
            logger.info(status)

    return losses.avg


def valid(valid_loader, model, criterions, logger):
    model.eval()  # eval mode (dropout and batchnorm is NOT used)

    losses = AverageMeter()

    reg_losses = AverageMeter()
    expression_accs = AverageMeter()
    gender_accs = AverageMeter()
    glasses_accs = AverageMeter()
    race_accs = AverageMeter()

    L1Loss, CrossEntropyLoss = criterions

    # Batches
    for i, (img, reg, expression, gender, glasses, race) in enumerate(valid_loader):
        # Move to GPU, if available
        img = img.to(device)
        reg_label = reg.type(torch.FloatTensor).to(device)  # [N, 5]
        expression_label = expression.type(torch.LongTensor).to(device)  # [N, 3]
        gender_label = gender.type(torch.LongTensor).to(device)  # [N, 2]
        glasses_label = glasses.type(torch.LongTensor).to(device)  # [N, 3]
        race_label = race.type(torch.LongTensor).to(device)  # [N, 4]

        # Forward prop.
        reg_out, expression_out, gender_out, glasses_out, race_out = model(img)

        # Calculate loss
        reg_loss = L1Loss(reg_out, reg_label)
        expression_loss = CrossEntropyLoss(expression_out, expression_label)
        gender_loss = CrossEntropyLoss(gender_out, gender_label)
        glasses_loss = CrossEntropyLoss(glasses_out, glasses_label)
        race_loss = CrossEntropyLoss(race_out, race_label)

        loss = reg_loss + expression_loss + gender_loss + glasses_loss + race_loss

        # Keep track of metrics
        losses.update(loss.item())

        reg_losses.update(reg_loss.item())
        expression_accuracy = accuracy(expression_out, expression_label)
        expression_accs.update(expression_accuracy)
        gender_accuracy = accuracy(gender_out, gender_label)
        gender_accs.update(gender_accuracy)
        glasses_accuracy = accuracy(glasses_out, glasses_label)
        glasses_accs.update(glasses_accuracy)
        race_accuracy = accuracy(race_out, race_label)
        race_accs.update(race_accuracy)

    # Print status
    status = 'Validation: Loss {loss.avg:.4f}\t' \
             'Reg Loss {reg_loss.val:.4f} ({reg_loss.avg:.4f})\t' \
             'Expression Accuracy {expression_acc.val:.4f} ({expression_acc.avg:.4f})\t' \
             'Gender Accuracy {gender_acc.val:.4f} ({gender_acc.avg:.4f})\t' \
             'Glasses Accuracy {expression_acc.val:.4f} ({expression_acc.avg:.4f})\t' \
             'Race Accuracy {expression_acc.val:.4f} ({expression_acc.avg:.4f})\n'.format(loss=losses,
                                                                                          reg_loss=reg_losses,
                                                                                          expression_acc=expression_accs,
                                                                                          gender_acc=gender_accs,
                                                                                          glasses_acc=glasses_accs,
                                                                                          race_acc=race_accs)

    logger.info(status)

    return losses.avg


def main():
    global args
    args = parse_args()
    train_net(args)


if __name__ == '__main__':
    main()
