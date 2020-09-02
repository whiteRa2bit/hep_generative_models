import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from generation.dataset.dataset_pytorch import SignalsDataset
from generation.train.utils import save_checkpoint, parse_args


class AutoEncoder(nn.Module):
    def __init__(self, sample_size):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(sample_size, 32),
            nn.ReLU(True),
            nn.Linear(32, 16),
            nn.ReLU(True),
            nn.Linear(16, 4)
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(True),
            nn.Linear(16, 32),
            nn.ReLU(True), nn.Linear(32, sample_size), nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def run_train(dataset, device='cpu', **kwargs):
    dataloader = DataLoader(dataset, batch_size=kwargs['batch_size'], shuffle=True)
    model = AutoEncoder(kwargs['sample_size'])
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

            # rows_num = 3
            # f, ax = plt.subplots(rows_num, rows_num, figsize=(rows_num**2, rows_num**2))
            # gs = gridspec.GridSpec(rows_num, rows_num)
            # gs.update(wspace=0.05, hspace=0.05)

            # for i in range(rows_num**2):
            #     ax[i // rows_num][i % rows_num].plot(generate_new_signal(model, dataset, device))
            # plt.show()

        # save_checkpoint(model, epoch, **kwargs)

    return model


def generate_new_signal(model, dataset, device='cpu', samples_num=10):
    idxs = np.random.choice(range(len(dataset)), samples_num)
    sample_signals = dataset[idxs]
    sample_signals = Variable(torch.from_numpy(sample_signals)).to(device)
    output = model.encoder(sample_signals)
    result = model.decoder(torch.mean(output, 0)).cpu().data.numpy()

    return result
