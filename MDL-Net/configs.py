import argparse


def load_config():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_g', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--resize', type=bool, default=True)
    parser.add_argument('--print_intervals', type=int, default=20)
    parser.add_argument('--evaluation', type=bool, default=None)
    parser.add_argument('--checkpoints', type=str, default='...', help='model checkpoints path')
    parser.add_argument('--checkpoints_3', type=str, default='...', help='model checkpoints path')
    parser.add_argument('--device_num', type=int, default=1)
    parser.add_argument('--gradient_clip', type=float, default=2.)
    parser.add_argument('--metrix', type=bool, default=True)
    parser.add_argument('--mode', type=int, default=4)

    return parser.parse_args()
