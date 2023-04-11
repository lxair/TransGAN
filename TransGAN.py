from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import copy
import math
from ...utils import get_or_create_path
from ...log import get_module_logger

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ...model.base import Model
from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import random
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from qlib.contrib.model.pytorch_transformer_ts import Transformer
from qlib.contrib.model.pytorch_transformer_ts import TransformerModel
import numpy as np


class TransGANModel(Model):
    def __init__(
        self,
        d_feat: int = 4,
        d_model: int = 4,
        nhead: int = 2,
        num_layers: int = 2,
        dropout: float = 0,
        activation: str = "gelu",
        batch_size: int = 256,
        early_stop=5,
        num_epochs: int = 5,
        learning_rate: float = 0.002,
        lr : float =0.0002,
        weight_decay: float = 1e-3,
        evaluation_epoch_num: int = 10,
        n_jobs: int = 10,
        hidden_size:int = 5,
        loss="mse",
        metric="",
        optimizer_betas :float= (0.9,0.999),
        optimizer="adam",
        seed=499,
        GPU=3,
    ):
        super().__init__()
        self.d_model = d_model
        self.hidden_size = hidden_size
        self.nhead = nhead
        self.lr = lr
        self.metric = metric
        self.d_feat = d_feat
        self.num_layers = num_layers
        self.optimizer_betas = optimizer_betas
        self.dropout = dropout
        self.loss = loss
        self.activation = activation
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.seed = seed
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.evaluation_epoch_num = evaluation_epoch_num
        self.n_jobs = n_jobs
        self.optimizer = optimizer.lower()
        self.device = torch.device("cuda:%d" % GPU if torch.cuda.is_available() and GPU >= 0 else "cpu")
        self.logger = get_module_logger("TransGANModel")
        self.logger.info("Naive TransGAN:" "\nbatch_size : {}" "\ndevice : {}".format(self.batch_size, self.device))

        # Create model
        self.model = TransGAN(d_feat,hidden_size)

        # Optimizer
        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay = self.weight_decay)
        elif optimizer.lower() == "gd":
            self.train_optimizer = optim.SGD(self.model.parameters(), lr=self.lr, weight_decay = self.weight_decay)
        else:
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))


    def mse(self, pred, label):
        loss = (pred.float() - label.float()) ** 2 
        return torch.mean(loss)

    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")

    def mse(self, pred, label):
        loss = (pred.float() - label.float()) ** 2 
        return torch.mean(loss)  

    def loss_fn(self, pred, label):
        mask = ~torch.isnan(label)
        if self.loss == "mse":
            return self.mse(pred[mask], label[mask])

        raise ValueError("unknown loss `%s`" % self.loss)
    
    def metric_fn(self, pred, label):

        mask = torch.isfinite(label)

        if self.metric in ("", "loss"):
            return -self.loss_fn(pred[mask], label[mask])

        raise ValueError("unknown metric `%s`" % self.metric)

    def train_epoch_gan(self, data_loader):
        
        self.model.train()
        
        generator = Generator(d_feat = 6,hidden_size=4).to(self.device) # 实例化Generator的类，调用里面的属性。
        discriminator = Discriminator(hidden_size=4).to(self.device)
        print("Generator and discriminator are initialized")

        criterion = nn.BCELoss() # 二分类任务的损失函数 ，输出层激活函数为sigmoid，输出代表正例的概率，与真实标签比较。
        optimizer_generator = optim.Adam(generator.parameters(), lr=self.lr, betas=self.optimizer_betas) # 优化器，adam优化generator的参数，学习率为lr,动量参数beats
        optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=self.lr, betas=self.optimizer_betas)

        real_label = 1.
        fake_label = 0.
        col_set=["KMID","KLOW","OPEN0","Ref($close, -2) / Ref($close, -1) - 1"]
        # col_set=["feature", "label"]

        for sequence_batch in data_loader:
    
            discriminator.zero_grad() # 更新梯度，清零。
            # Format batch
            real_sequence = sequence_batch[:, :, :3].to(self.device) # 取前三个时间特征，因为第四个是要预测的。
            batch_size = real_sequence.size(0) # 读取batch_size,后面比较的时候需要用到生成了batch_size的数据。
            real_labels = torch.full((batch_size,), real_label, dtype=torch.float, device=self.device) #111111.....
            # Forward pass real batch through D
            discriminator_output_real = discriminator(real_sequence) # 将真实的值传到鉴别器中，先学习真实的数据。
            # Calculate loss on all-real batch
            discriminator_output_real_scaled = torch.sigmoid(discriminator_output_real) # 这里的值好像不在0-1之间，用sigmod函数，然后与真实的值进行比较。
            discriminator_error_real = criterion(discriminator_output_real_scaled, real_labels) # 进行比较，就是loss，交叉滴。
            discriminator_error_real.backward(retain_graph=True) # 回传，更新模型的参数。
            
            ## Training with fake batch
            # Assign first t values
            noise = torch.randn((batch_size, 29, 3)).to(self.device)
            generator_input_sequence = sequence_batch[:, :, :3].to(self.device) # 取前三个时间特征，因为第四个是要预测的。
            generator_input_sequence = generator_input_sequence[:,:-1].to(self.device) # 留最后一个时间步的数据，然后用生成器的生成数据与其进行比较，计算renerator的loss.
            generator_input_sequence_with_noise = torch.cat([generator_input_sequence, noise], dim=2)
            #  Generate (t+1)th value from first t values
            generated_values = generator(generator_input_sequence_with_noise) # 将真实的值传入到生成器中学习。
            
            fake_labels = torch.full((batch_size,), fake_label, dtype=torch.float, device=self.device)

            generated_values = generated_values[:, :, :3].to(self.device)
            generator_result_concat = torch.cat((generator_input_sequence, generated_values), dim =1) #将生成的数据和真实的数据进行拼接。

            # discriminator_output_fake = discriminator(generated_values.detach())
            discriminator_output_fake = discriminator(generator_result_concat).view(-1) #将生成的和虚假的拼接放到discriminator，进行训练。

            discriminator_output_fake = torch.sigmoid(discriminator_output_fake)
            discriminator_error_fake = criterion(discriminator_output_fake, fake_labels) # 计算鉴别器辨别的能力的loss.
            
            # Calculate the gradients for this batch
            discriminator_error_fake.backward(retain_graph=True)  # 进行回传，更新损失函数。
            # Add the gradients from the all-real and all-fake batches
            discriminator_error = discriminator_error_real + discriminator_error_fake # 将真实样本和生成样本的判别器损失相加得到总的判别器损失
            # Update D
            optimizer_discriminator.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            generator.zero_grad() # 清零
            real_labels = torch.full((batch_size,), real_label, dtype=torch.float, device=self.device)
            # Since we just updated D, perform another forward pass of all-fake batch through D
            generator_result_concat_grad = torch.cat((generator_input_sequence, generated_values), 1) # 拼接
            discriminator_output_fake = discriminator(generator_result_concat_grad).view(-1) # 作为输入到鉴别器
            discriminator_output_fake = torch.sigmoid(discriminator_output_fake)
            # Calculate G's loss based on this output
            generator_error = criterion(discriminator_output_fake, real_labels) # 生成器的误差，即生成器生成的样本被鉴别器判定为真实样本的概率
            generator_error.backward()
            optimizer_generator.step()

        
        print('Generator Loss: {:.4f}, Discriminator Loss: {:.4f}'.format(generator_error.item(), discriminator_error.item()))
        for col_name, real, generated in zip(col_set, sequence_batch[0][-1], generated_values[0][0]):
                print(f"{col_name} | Real:{real:.4f} / Generated:{generated:.4f}")


    def test_epoch(self, data_loader):
        
        generator = Generator(d_feat = 3,hidden_size=4).to(self.device)
        torch.save(generator.generator_transformer.state_dict(), 'saved_models/generator_transformer.pth')

    #   加载
        # generator_transformer = Transformer(d_model=4, nhead=2, num_layers=2, dropout=0, device=None)
        generator.generator_transformer.load_state_dict(torch.load('saved_models/generator_transformer.pth'))
        
        generator.generator_transformer.eval()

        scores = []
        losses = []

        for data in data_loader:

            feature = data[:, :, 0:-1].to(self.device).to(self.device)

            label = data[:, -1, -1].to(self.device)

            feature = feature.to("cuda:3")
            label = label.to("cuda:3")

            

            with torch.no_grad():
                pred = generator.generator_transformer(feature.float())  # .float(),这里跳进去的函数是”mse“那段/
                loss = self.loss_fn(pred, label)
                losses.append(loss.item())

                score = self.metric_fn(pred, label)
                scores.append(score.item())

        return np.mean(losses), np.mean(scores)
    
    def train_epoch(self,data_loader):
        
        # model.train()的作用是启用 Batch Normalization 和 Dropout。
        generator = Generator(d_feat = 3,hidden_size=4).to(self.device) # 实例化了class Generator
        torch.save(generator.generator_transformer.state_dict(), 'saved_models/generator_transformer.pth')

        # 加载
  
        generator.generator_transformer.load_state_dict(torch.load('saved_models/generator_transformer.pth'),)
        
        generator.generator_transformer.train() 

        for data in data_loader: 
            
            data = data.to("cuda:3")

            feature = data[:, :, 0:-1].to(self.device) # 三维数组除了最后一列都选。
            label = data[:, -1, -1].to(self.device) ## 选最后一列最后一个数。

            feature = feature.to("cuda:3")
            label = label.to("cuda:3")
            pred = generator.generator_transformer(feature.float())  # .float()

            loss = self.loss_fn(pred, label)

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(generator.generator_transformer.parameters(), 3.0)
            self.train_optimizer.step()
    
    def predict(self, dataset):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        
        generator = Generator(d_feat = 3,hidden_size=4).to(self.device) # 实例化了class Generator
        torch.save(generator.generator_transformer.state_dict(), 'saved_models/generator_transformer.pth')

    #     # 加载
        generator.generator_transformer.load_state_dict(torch.load('saved_models/generator_transformer.pth'))
        # 加载
        
        dl_test = dataset.prepare("test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
        dl_test.config(fillna_type="ffill+bfill")
        test_loader = DataLoader(dl_test, batch_size=self.batch_size, num_workers=self.n_jobs)
        
        generator.generator_transformer.eval()
        
        preds = []

        for data in test_loader:

            feature = data[:, :, 0:-1].to(self.device)
           

            with torch.no_grad():
                pred = generator.generator_transformer(feature.float()).detach().cpu().numpy()

            preds.append(pred)
        return pd.Series(np.concatenate(preds), index=dl_test.get_index())
    

    def fit(
        self,
        dataset: DatasetH,
        evals_result=dict(),
        save_path=None,
    ):
        
        dl_train = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        dl_valid = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        
        if dl_train.empty or dl_valid.empty:
            raise ValueError("Empty data from dataset, please check your dataset config.")

        dl_train.config(fillna_type="ffill+bfill")  # process nan brought by dataloader
        dl_valid.config(fillna_type="ffill+bfill")  # process nan brought by dataloader

        # Dataloader是pytorch的数据处理，并用batch_sampler将指定batch分配给指定worker，worker将它负责的batch加载进RAM。
        # num_worker设置得大，好处是寻batch速度快，因为下一轮迭代的batch很可能在上一轮/上上一轮...迭代时已经加载好了。坏处是内存开销大，也加重了CPU负担

        train_loader = DataLoader(
            dl_train, batch_size=self.batch_size, shuffle=True, num_workers=self.n_jobs, drop_last=True
        )
        valid_loader = DataLoader(
            dl_valid, batch_size=self.batch_size,shuffle=False, num_workers=self.n_jobs, drop_last=True
        )
                                            
      ##  transformer进行GAN网络训练之前，要先拟合。
        transformer_model = TransformerModel(d_model=4, nhead=2, num_layers=2, dropout=0, device=None)
        transformer_model.fit(dataset)


      ##  训练GAN网络的transformer模型
        for epoch in range(1, self.num_epochs + 1):
            self.logger.info("Epoch [%d/%d]", epoch, self.num_epochs)
            self.logger.info("Training...")
            train_result = self.train_epoch_gan(train_loader)

        generator = Generator(d_feat = 3,hidden_size=4).to(self.device) # 实例化了class Generator
        torch.save(generator.generator_transformer.state_dict(), 'saved_models/generator_transformer.pth')
        generator.generator_transformer.load_state_dict(torch.load('saved_models/generator_transformer.pth'))

    #   ## 训练好的transformer进行训练，方法是不变的
        save_path = get_or_create_path(save_path)
        stop_steps = 0
        train_loss = 0
        best_score = -np.inf
        best_epoch = 0
        evals_result["train"] = []
        evals_result["valid"] = []

        # train
        self.logger.info("training...")
        self.fitted = True

        for step in range(self.num_epochs):
            self.logger.info("Epoch%d:", step)
            self.logger.info("training...")
            self.train_epoch(train_loader)
            
            self.logger.info("evaluating...")
            
            train_loss, train_score = self.test_epoch(train_loader)
            val_loss, val_score = self.test_epoch(valid_loader)
            self.logger.info("train %.6f, valid %.6f" % (train_score, val_score))
            evals_result["train"].append(train_score)
            evals_result["valid"].append(val_score)

            if val_score > best_score:
                best_score = val_score
                stop_steps = 0
                best_epoch = step
                best_param = copy.deepcopy(generator.generator_transformer.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    self.logger.info("early stop")
                    break

        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        generator.generator_transformer.load_state_dict(best_param)
        torch.save(best_param, save_path)

        if self.use_gpu:
            torch.cuda.empty_cache()
        

class Generator(nn.Module):
    def __init__(self, d_feat,hidden_size):
        super(Generator, self).__init__()
        
        # self.transformer_encoder = nn.TransformerEncoder(
        #     nn.TransformerEncoderLayer(d_model=hidden_size, nhead = 1),
        #     num_layers=1)

        # self.conv = nn.Sequential(
        #     nn.Conv1d(in_channels=3, out_channels=32, kernel_size=3),
        #     nn.BatchNorm1d(32),
        #     nn.ReLU(),
        #     nn.Conv1d(in_channels=32, out_channels=3, kernel_size=3),
        #     nn.BatchNorm1d(3),
        #     nn.ReLU(),
        # )
        
        # self.fc2 = nn.Linear(1, 1)
        self.generator_transformer = Transformer(d_feat,d_model = 4, nhead=2, num_layers=2, dropout=0, device=None)

        # self.fc2 = nn.Linear(1, 1)

        self.fc = nn.Linear(1, 6)

    def forward(self, input_sequences):
        input_sequences = input_sequences.float()
        # x: [batch_size, seq_len, hidden_size]
        
        input_sequences = input_sequences.to("cuda:3")
        
        # 将噪音向量与输入序列拼接起来，形成新的输入  输入是（256，29，6）
        # input_sequences = self.conv(input_sequences.transpose(1,2)) # [batch_size, hidden_size, seq_len]
    
        # input_sequences = input_sequences.transpose(1,2) # [batch_size, seq_len, hidden_size]
        
        input_sequences = self.generator_transformer(input_sequences)     # transformer取的是最后一个时间步的batch_size
        
        input_sequences = input_sequences.unsqueeze(1) # [256,1]
         
        # input_sequences = F.relu(self.fc2(input_sequences))  # 线性层，1.增加模型复杂度，可以学习更复杂的特征 2. 线性层将输入数据进行线性变换，将其映射到更高维度的特征空间中。
        
        input_sequences = self.fc(input_sequences) # [256,3]   # 维度变换
        
        input_sequences = input_sequences.view(256,-1,6)
        
        return input_sequences
    
class Discriminator(nn.Module):
    def __init__(self, hidden_size):
        super(Discriminator, self).__init__()
        # self.transformer_encoder = nn.TransformerEncoder(
        #     nn.TransformerEncoderLayer(d_model=hidden_size, nhead = 1),
        #     num_layers=1)

        self.discriminator_transformer = Transformer(d_feat=3,d_model=4, nhead=2, num_layers=2, dropout=0, device=None)
        
    def forward(self, x):
        x = x.float()
        # x: [batch_size, seq_len, hidden_size]
        x = self.discriminator_transformer(x)   
        return x
    
class TransGAN(nn.Module):
    def __init__(self, d_feat,hidden_size):
        super(TransGAN, self).__init__()
        
        # 定义生成器
        self.generator = Generator(d_feat,hidden_size)
        
        # 定义辨别器
        self.discriminator = Discriminator(hidden_size)

    def forward(self, input_sequences):
        # 生成器生成样本

        input_sequences = input_sequences.to("cuda:3")
        generated_samples = self.generator(input_sequences)
        
        # 将生成的样本和真实样本拼接在一起
        combined_samples = torch.cat([input_sequences, generated_samples], dim=0)
        
        # 辨别器判断样本真伪
        outputs = self.discriminator(combined_samples)
        
        # 将输出分为真实样本和生成样本的预测结果
        real_outputs, generated_outputs = torch.split(outputs, input_sequences.shape[0], dim=0)
        
        return generated_samples, real_outputs, generated_outputs
    
    