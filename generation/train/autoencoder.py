import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from generation.data.data_simulation import Nakagami


class SignalsDataset(Dataset):
    def __init__(self, signals_data):
        self.signals_data = signals_data

    def __len__(self):
        return len(self.signals_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.signals_data[idx].astype("float32")


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


def run_train(model, dataloader, **kwargs):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=kwargs['learning_rate'], weight_decay=1e-5)

    for epoch in range(kwargs['num_epochs']):
        for signal in dataloader:
            signal = Variable(signal)
            output = model(signal)
            loss = criterion(output, signal)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if kwargs['verbose'] and epoch % kwargs['print_each'] == 0:
            print('epoch [{}/{}], loss:{:.4f}'
                  .format(epoch + 1, kwargs['num_epochs'], loss.data.item()))


def generate_new_signal(model, dataset, samples_num=10):
    idxs = np.random.choice(range(len(dataset)), samples_num)
    sample_signals = dataset[idxs]
    sample_signals = Variable(torch.from_numpy(sample_signals))
    output = model.encoder(sample_signals)
    result = model.decoder(torch.mean(output, 0)).cpu().data.numpy()

    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--learning_rate', type=int, required=True)
    parser.add_argument('--print_each', type=int, default=20, help='number of epochs between print messages')
    parser.add_argument("--verbose", help="increase output verbosity",
                        action="store_true", default=True)
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
    run_train(model, dataloader,
              learning_rate=args.learning_rate,
              num_epochs=args.num_epochs,
              batch_size=args.batch_size,
              print_each=args.print_each,
              verbose=args.verboe)


if __name__ == '__main__':
    main()
