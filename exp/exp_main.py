from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from models import AHuber, PatchTST, iTransformer

import os
import time
import sys

import warnings

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'PatchTST': PatchTST,
            'iTransformer': iTransformer,
            'AHuber': AHuber

        }
        model = model_dict[self.args.model].Model(self.args).float()

        return model

    def _get_data(self, flag):

        # Call data_provider
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                if self.args.model == 'iTransformer':
                    
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                    
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[..., f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                loss = criterion(outputs, batch_y)
                total_loss.append(loss.item())

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        _, vali_loader = self._get_data(flag='val')
        # Also get test data loader early
        _, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            adjust_learning_rate(model_optim, epoch, self.args)

            train_loss = []
            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                # Move time features to device
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                if self.args.model == 'iTransformer':
                    # Prepare decoder input for iTransformer
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x)
                
                outputs = outputs[..., f_dim:]
                main_loss = criterion(outputs, batch_y)
                loss = main_loss

                train_loss.append(loss.item())

                loss.backward()
                model_optim.step()

                # Custom progress bar: update every 10 steps or on last step
                if (i + 1) % 10 == 0 or (i + 1) == train_steps:
                    progress = (i + 1) / train_steps
                    bar_length = 30
                    filled_len = int(round(bar_length * progress))
                    
                    # Progress bar body
                    bar = '█' * filled_len + ' ' * (bar_length - filled_len)

                    # ETA calculation
                    elapsed_time = time.time() - epoch_time
                    eta_seconds = (elapsed_time / (i + 1)) * (train_steps - (i + 1))

                    # Format output
                    progress_str = (
                        f"\rEpoch [{epoch + 1}/{self.args.train_epochs}] "
                        f"[{bar}] {progress:.1%} | "
                        f"Loss: {loss.item():.3f} | "
                        f"ETA: {int(eta_seconds)}s"
                    )
                    
                    sys.stdout.write(progress_str)
                    sys.stdout.flush()

            # Newline after epoch loop
            print()

            print(f"Epoch: {epoch + 1} training cost time: {time.time() - epoch_time:.3f}")
            
            # Notify validation step
            print("Running validation...")
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_loader, criterion)

            print("Epoch: {0}/{1}, Steps: {2} | Train Loss: {3:.3f} Vali Loss: {4:.3f}".format(
                epoch + 1, self.args.train_epochs, train_steps, train_loss, vali_loss))
            
            # During last 5 epochs compute and print test loss
            if epoch >= self.args.train_epochs - 5:
                print("Running test loss calculation...")
                test_loss = self.vali(test_loader, criterion)
                print("Epoch: {0}/{1} | Test Loss: {2:.3f}".format(
                    epoch + 1, self.args.train_epochs, test_loss))

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        
        self.model.eval()
        with torch.no_grad():
            # Unpack and evaluate batches
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # Move time features to device
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # Call model appropriately
                if self.args.model == 'iTransformer':
                    # Prepare decoder input for iTransformer
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[..., f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                preds.append(outputs.detach().cpu().numpy())
                trues.append(batch_y.detach().cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        with open("result.txt", 'a') as f:
            f.write(setting + "  \n")
            f.write('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
            f.write('\n\n')

        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)

        return mse, mae

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, batch_x in enumerate(pred_loader):
                batch_x = batch_x[0].float().to(self.device)

                outputs = self.model(batch_x)

                preds.append(outputs.detach().cpu().numpy())

        preds = np.concatenate(preds, axis=0)

        # 结果保存
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
