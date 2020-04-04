import numpy as np
import argparse
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from generation.data.data_simulation import Nakagami
from generation.data.dataset_pytorch import SignalsDataset
import os
from generation.config import (
    CHECKPOINTS_DIR
)


class AutoEncoder(nn.Module):
    def __init__(self, sample_size):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(sample_size, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 12)
        )
        self.decoder = nn.Sequential(
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True), nn.Linear(128, sample_size), nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def run_train(model, dataloader, device='cpu', **kwargs):
    def save_checkpoint():
        if not kwargs['no_save'] and epoch % kwargs['save_interval'] == 0:
            if kwargs['save_dir']:
                save_dir = os.path.join(kwargs['save_dir'], kwargs['model_name'])
            else:
                save_dir = os.path.join(CHECKPOINTS_DIR, kwargs['model_name'])

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            checkpoint_name = os.path.join(save_dir, 'checkpoint_{}.pth'.format(epoch // kwargs['save_interval']))
            torch.save(model.state_dict(), checkpoint_name)

    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=kwargs['learning_rate'], weight_decay=1e-5)

    for epoch in range(kwargs['num_epochs']):
        for signal in dataloader:
            signal = Variable(signal).to(device)
            output = model(signal)
            loss = criterion(output, signal)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if kwargs['verbose'] and epoch % kwargs['print_each'] == 0:
            print('epoch [{}/{}], loss:{:.4f}'
                  .format(epoch + 1, kwargs['num_epochs'], loss.data.item()))

        save_checkpoint()


def generate_new_signal(model, dataset, device='cpu', samples_num=10):
    idxs = np.random.choice(range(len(dataset)), samples_num)
    sample_signals = dataset[idxs]
    sample_signals = Variable(torch.from_numpy(sample_signals)).to(device)
    output = model.encoder(sample_signals)
    result = model.decoder(torch.mean(output, 0)).cpu().data.numpy()

    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--learning_rate', type=float, required=True)
    parser.add_argument('--print_each', type=int, default=20, help='number of epochs between print messages')
    parser.add_argument('--verbose', action="store_false", help='increase output verbosity')
    parser.add_argument('--cpu', action='store_true', help='use CPU instead of CUDA')
    parser.add_argument('--save_dir', type=str, help='dir to save model checkpoints', default='')
    parser.add_argument('--save_interval', type=int, default=10, help='save a checkpoint every N epochs')
    parser.add_argument('--no_save', action='store_true', help='don’t save models or checkpoints')
    parser.add_argument('--model_name', type=str, default='autoencoder', help='model name while saving')
    return parser.parse_args()


def main():
    args = parse_args()

    # Data params
    SAMPLE_SIZE = 1000
    Q_LOWER = 0.001
    Q_UPPER = 0.999
    NU_MIN = 0.9
    NU_MAX = 1.2
    NU_STEP = 0.005

    nakagami = Nakagami(SAMPLE_SIZE, Q_LOWER, Q_UPPER)
    nu_values = np.arange(NU_MIN, NU_MAX, NU_STEP)
    data = nakagami.get_nakagami_data(nu_values)
    dataset = SignalsDataset(data)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = AutoEncoder(SAMPLE_SIZE)
    device = torch.device("cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu")
    run_train(model, dataloader, device,
              learning_rate=args.learning_rate,
              num_epochs=args.num_epochs,
              batch_size=args.batch_size,
              print_each=args.print_each,
              verbose=args.verbose,
              cpu=args.cpu,
              save_dir=args.save_dir,
              save_interval=args.save_interval,
              no_save=args.no_save,
              model_name=args.model_name)


if __name__ == '__main__':
    main()
