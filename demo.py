import os
import torch.utils.data as data
from data.dataset import GalaxyDataset
from config import generate_opt
from data.data_augmentation import preprocess
from torchvision import datasets


def generate_loader(args):
    """
    :param args: the arguments
    :return:
    """
    train_data = GalaxyDataset(os.path.join(args.path, 'train'), transform=preprocess())
    val_data = GalaxyDataset(os.path.join(args.path, 'val'), transform=preprocess())
    print(len(train_data))
    # test_data = GalaxyDataset(path=args.path + '/test/')

    train_loader = data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = data.DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=False, num_workers=0)
    # test_loader = data.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader


if __name__ == "__main__":
    args = generate_opt()
    train_loader, val_loader = generate_loader(args)
    train_dataset = datasets.ImageFolder(os.path.join(args.path, 'train'), transform=preprocess())
    print(len(train_dataset))
    # print(train)
