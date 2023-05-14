import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.models as models

from config import generate_opt
from data.dataset import GalaxyDataset
from tensorboardX import SummaryWriter
from data.data_augmentation import preprocess


def setup_seed(seed):
    """
    :param seed: setup random seed to ensure the reproducibility of the model
    :return: None
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def generate_loader(args):
    """
    :param args: the arguments
    :return:
    """
    train_data = GalaxyDataset(path=args.path + '/train/', transform=preprocess())
    val_data = GalaxyDataset(path=args.path + '/val/', transform=preprocess())
    test_data = GalaxyDataset(path=args.path + '/test/', transform=preprocess())

    train_loader = data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = data.DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = data.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


def val(model, args, loader, epoch, device, mode='val'):
    model.to(device)
    total = 0
    corrects = 0
    model.eval()
    val_loss = 0.0
    loss_func = nn.CrossEntropyLoss()
    with torch.no_grad():
        # val_bar = tqdm(loader, file=sys.stdout)
        for i, (images, labels) in enumerate(loader):
            # to cuda
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            # put the outputs to the loss function directly, the labels will broadcast automatically
            loss = loss_func(outputs, labels)
            # outputs is a matrix of [batch_size, 10], the max value's index of every row is the predicted label
            _, predictions = torch.max(outputs, dim=1)
            corrects += (predictions == labels).cpu().sum().item()
            total += labels.size(0)
            val_loss += loss.item() * images.size(0)
            # val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
            #                                            args.epoch)
    # calculate loss and val acc
    val_loss = val_loss / len(loader.dataset)
    accuracy = corrects * 1.0 / total * 100.0
    if mode == 'val':
        return val_loss, accuracy
    return accuracy


def load_model(save_path, optimizer, model):
    """
    # 断点训练
    :param save_path: the model save path
    :param optimizer: optimizer setup
    :param model: model setup
    :return:
    """
    assert os.path.exists(save_path), "file {} does not exist.".format(save_path)
    model_data = torch.load(save_path)
    start_epoch = model_data['epoch']
    model.load_state_dict(model_data['model_dict'])
    optimizer.load_state_dict(model_data['optimizer_dict'])
    print('model load success')

    return start_epoch, model, optimizer


def train(args, model, optimizer, scheduler, loss_func, train_loader, val_loader, test_loader, device, writer):
    """
    :param model: example of training model
    :param train_loader: train loader
    :param val_loader: val loader
    :param device: cuda or cpu
    :return: None
    """

    model.train()
    model.to(device)
    # iter for every epoch
    best_acc = 0.0
    for epoch in range(args.epoch - args.start_epoch):
        # one epoch
        epoch = epoch + args.start_epoch
        per_epoch_loss = 0.0
        per_epoch_correct = 0
        per_epoch_total = 0
        # train_bar = tqdm(train_loader, total=len(train_loader))

        for i, (images, labels) in enumerate(train_loader):
            # one iteration
            # to cuda
            images = images.to(device)
            labels = labels.to(device)
            # forward and backward
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()
            # calculate accuracy
            _, predictions = torch.max(outputs, dim=1)
            correct_nums = (predictions == labels).cpu().sum().item()
            per_epoch_correct += correct_nums
            # calculate loss, the loss belongs to every sample
            per_epoch_loss += loss.item() * images.size(0)
            per_epoch_total += labels.size(0)
            # tqdm train_bar

        # adjust the scheduler
        if scheduler is not None:
            scheduler.step()
        # calculate the accuracy of three different datasets
        train_acc = per_epoch_correct * 1.0 / per_epoch_total * 100.0
        val_loss, val_acc = val(model, args, val_loader, epoch, device)
        test_acc = val(model, args, test_loader, epoch, device, mode='test')
        # calculate the loss of every sample loader.dataset to get the dataset
        per_epoch_loss = per_epoch_loss / len(train_loader.dataset)
        # train_bar.set_description(f'Epoch [{epoch}/{args.epoch}]')
        # train_bar.set_postfix(loss=per_epoch_loss, train_acc=train_acc, val_acc=val_acc)
        # write the loss and the acc in tensorboard
        writer.add_scalars('Loss',
                           {'train': per_epoch_loss,
                            'val': val_loss},
                           global_step=epoch + 1)
        writer.add_scalars('Accuracy',
                           {'train': train_acc,
                            'val': val_acc},
                           global_step=epoch + 1)
        # save model pre_weights
        os.makedirs(args.weights_save_path, exist_ok=True)
        # print training message
        print(
            f'Epoch:{epoch}, per epoch train loss is: {per_epoch_loss:.4f}, val loss:{val_loss:.4f},',
            f' train acc:{train_acc:.2f}, val_acc: {val_acc:.2f},test_acc: {test_acc:.2f}.')
        if val_acc > best_acc or epoch % args.save_frequency == 0:
            best_acc = val_acc
            final_save_path = os.path.join(args.weights_save_path, str(epoch).zfill(4) + '.pth')
            torch.save({'epoch': epoch,
                        'optimizer_dict': optimizer.state_dict(),
                        'model_dict': model.state_dict()},
                       final_save_path)


if __name__ == "__main__":
    # get args
    args = generate_opt()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # setup random seed
    setup_seed(2023)
    # resnet50 model
    model = models.resnet50(pretrained=False)
    model.load_state_dict(torch.load('./pre_weights/resnet50-11ad3fa6.pth'))
    model.fc = nn.Sequential(
        nn.Linear(2048, 1000, bias=True),
        nn.Dropout(0.5),
        nn.ReLU(inplace=True),
        nn.Linear(1000, 256, bias=True),
        nn.Dropout(0.5),
        nn.ReLU(inplace=True),
        nn.Linear(256, 10, bias=True),
    )
    # kaiming_init
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
            nn.init.constant_(m.bias.data, 0.0)
    # select the loss function
    loss_func = nn.CrossEntropyLoss(reduction='mean')
    # optimizer and the scheduler
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)  # update lr for every epoch
    # dataloader
    train_loader, val_loader, test_loader = generate_loader(args=args)
    # log
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(logdir=args.log_dir)
    # 断点训练
    if args.resume:
        start_epoch, model, optimizer = load_model(args.last_model_path, model, optimizer)

    train(args=args,
          model=model,
          optimizer=optimizer,
          scheduler=scheduler,
          loss_func=loss_func,
          train_loader=train_loader,
          val_loader=val_loader,
          test_loader=test_loader,
          device=device,
          writer=writer
          )
