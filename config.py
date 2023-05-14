import argparse


def generate_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=10000, type=int)
    parser.add_argument('--path', default='../../datasets/sdss_10/', type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--save_frequency', default=10, type=int, help='the epoch frequency of the training process')
    parser.add_argument('--weights_save_path', default='./weights.', type=str)
    parser.add_argument('--log_dir', default='./logs/50/', type=str)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--last_model_path', default='./pre_weights/', type=str)

    args = parser.parse_args()

    return args