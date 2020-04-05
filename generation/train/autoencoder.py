import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from generation.data.data_simulation import Nakagami
from generation.data.dataset_pytorch import SignalsDataset
from generation.train.utils import save_checkpoint, parse_args


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

            rows_num = 3
            f, ax = plt.subplots(rows_num, rows_num, figsize=(rows_num**2, rows_num**2))
            gs = gridspec.GridSpec(rows_num, rows_num)
            gs.update(wspace=0.05, hspace=0.05)

            for i, sample in enumerate(rows_num**2):
                ax[i // rows_num][i % rows_num].plot(generate_new_signal(model, dataset, device))
            plt.show()

        save_checkpoint(model, epoch, **kwargs)

    return model


def generate_new_signal(model, dataset, device='cpu', samples_num=10):
    idxs = np.random.choice(range(len(dataset)), samples_num)
    sample_signals = dataset[idxs]
    sample_signals = Variable(torch.from_numpy(sample_signals)).to(device)
    output = model.encoder(sample_signals)
    result = model.decoder(torch.mean(output, 0)).cpu().data.numpy()

    return result


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

    device = torch.device("cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu")
    run_train(dataset, device,
              sample_size=args.sample_size,
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
