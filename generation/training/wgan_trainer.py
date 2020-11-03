import tqdm
import torch
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
import wandb

from generation.training.schedulers import GradualWarmupScheduler
from generation.training.abstract_trainer import AbstractTrainer


class WganTrainer(AbstractTrainer):
    def run_train(self, dataset):
        g_cosine_scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(
            self.g_optimizer, self.config["epochs_num"], eta_min=0, last_epoch=-1)
        g_scheduler = GradualWarmupScheduler(
            self.g_optimizer, multiplier=10, total_epoch=200, after_scheduler=g_cosine_scheduler)
        d_cosine_scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(
            self.d_optimizer, self.config["epochs_num"], eta_min=0, last_epoch=-1)
        d_scheduler = GradualWarmupScheduler(
            self.d_optimizer, multiplier=10, total_epoch=200, after_scheduler=d_cosine_scheduler)
        
        dataloader = DataLoader(dataset, batch_size=self.config["batch_size"], shuffle=True)
        self._initialize_wandb()

        for epoch in tqdm.tqdm(range(self.config['epochs_num'])):
            for it, data in enumerate(dataloader):
                if it == dataloader.dataset.__len__() // self.config['batch_size']:
                    break

                # Dicriminator forward-loss-backward-update
                X = Variable(data)
                X = X.to(self.config['device'])
                z = Variable(torch.randn(self.config['batch_size'], self.config['z_dim']))
                z = z.to(self.config['device'])

                g_sample = self.generator(z)
                d_real = self.discriminator(X)
                d_fake = self.discriminator(g_sample)

                gradient_penalty = self._compute_gp(X, g_sample)

                d_loss = torch.mean(d_fake) - torch.mean(d_real)
                d_loss_gp = d_loss + gradient_penalty
                d_loss_gp.backward()
                self.d_optimizer.step()

                # Housekeeping - reset gradient
                self._reset_grad()

                if it % self.config['d_coef'] == 0:
                    # Generator forward-loss-backward-update
                    X = Variable(data)
                    X = X.to(self.config['device'])
                    z = Variable(torch.randn(self.config['batch_size'], self.config['z_dim']))
                    z = z.to(self.config['device'])

                    g_sample = self.generator(z)
                    d_fake = self.discriminator(g_sample)

                    g_loss = -torch.mean(d_fake)
                    g_loss.backward()
                    self.g_optimizer.step()

                    # Housekeeping - reset gradient
                    self._reset_grad()

            g_scheduler.step(epoch + 1)
            d_scheduler.step(epoch + 1)

            if epoch % self.config['log_each'] == 0:
                wandb.log(
                    {
                        "D loss": d_loss.cpu(),
                        "Gradient penalty": gradient_penalty.cpu(),
                        "G loss": g_loss.cpu(),
                        "G lr": self.g_optimizer.param_groups[0]['lr'],
                        "D lr": self.d_optimizer.param_groups[0]['lr']
                    },
                    step=epoch)
                self.generator.visualize(g_sample, X, epoch)
            if epoch % self.config['save_each'] == 0:
                self._save_checkpoint(self.generator, f"generator_{epoch}")
                self._save_checkpoint(self.discriminator, f"discriminator_{epoch}")

    def _compute_gp(self, X, g_sample):
        alpha = torch.rand((self.config["batch_size"], 1,
                            1)).to(self.config['device'])  # TODO: (@whiteRa2bit, 2020-09-25) Fix shape
        x_hat = alpha * X.data + (1 - alpha) * g_sample.data
        x_hat.requires_grad = True
        pred_hat = self.discriminator(x_hat)
        gradients = grad(
            outputs=pred_hat,
            inputs=x_hat,
            grad_outputs=torch.ones(pred_hat.size()).to(self.config['device']),
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        gradient_penalty = self.config['lambda'] * ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_penalty
