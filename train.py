import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import argparse

import numpy as np
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn.functional as F

from torchsummary import summary
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.tensorboard import SummaryWriter

from models.vgg import VGG16
from models.resnet import ResNet18, ResNet50, ResNet101
from models.regnet import RegNetX_200MF, RegNetY_400MF
from models.mobilenetv2 import MobileNetV2
from models.resnext import ResNeXt29_32x4d, ResNeXt29_2x64d
from models.dla_simple import SimpleDLA
from models.densenet import DenseNet121
from models.preact_resnet import PreActResNet18
from models.dpn import DPN92
from models.dla import DLA

modle_dic = {'VGG16':VGG16, 
             'ResNet18':ResNet18,
             'ResNet50':ResNet50,
             'ResNet101':ResNet101,
             'RegNetX_200MF':RegNetX_200MF,
             'RegNetY_400MF':RegNetY_400MF,
             'MobileNetV2':MobileNetV2,
             'ResNeXt29(32x4d)':ResNeXt29_32x4d,
             'ResNeXt29(2x64d)':ResNeXt29_2x64d,
             'SimpleDLA':SimpleDLA,
             'DenseNet121':DenseNet121,
             'PreActResNet18':PreActResNet18,
             'DPN92':DPN92,
             'DLA':DLA}

class warmup_lambda:
    def __init__(self, batchs:int, warmup_epochs:int=1,):
        self.batchs = batchs
        self.warmup_epochs = warmup_epochs

    def __call__(self, epoch):
        return epoch / self.batchs / self.warmup_epochs

def top_k_acc(predict, y, k=5):
    topk_pred = np.argsort(predict, axis=1)[:, -k:][:, ::-1]
    correct = np.any(topk_pred == y.reshape(-1, 1), axis=1)
    top_k_acc = np.sum(correct) / y.size
    return top_k_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CIFAR10 Training')
    parser.add_argument('--mode_name', type=str, default='MobileNetV2', help='Name of the model')
    parser.add_argument('--learn_rate', type=float, default=0.1, help='Learning rate for the optimizer')
    parser.add_argument('--train_epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--warmup_epochs', type=int, default=1, help='Number of warmup epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training and validation')
    parser.add_argument('--log_dir', type=str, default='./runs', help='Directory for saving logs')

    args = parser.parse_args()
    print("Net Name:", args.mode_name)
    print("Training Epochs:", args.train_epochs)
    print("Warmup Epochs:", args.warmup_epochs)
    print("Learning Rate:", args.learn_rate)
    print("Batch Size:", args.batch_size)
    log_dir = f"{args.log_dir}/{args.mode_name}/{datetime.now().strftime('%Y%m%d%H%M%S')}"
    print("Log directory:", log_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_train = datasets.CIFAR10(root='./data', train=True, download=True)
    x_train = torch.tensor(dataset_train.data, dtype=torch.int8).to(device)
    x_train = x_train.permute(0, 3, 1, 2)
    y_train = torch.tensor(dataset_train.targets).to(device)

    train_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.RandomCrop(32, 4),
        v2.RandomHorizontalFlip(p=0.5),
        # v2.RandomVerticalFlip(p=0.5),
        # v2.RandomRotation(30),
        v2.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    ])

    dataset_val = datasets.CIFAR10(root='./data', train=False, download=True)
    x_val = torch.tensor(dataset_val.data, dtype=torch.int8).to(device)
    x_val = x_val.permute(0, 3, 1, 2)
    y_val = torch.tensor(dataset_val.targets).to(device)
    
    val_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    ])

    model = MobileNetV2(num_classes=len(dataset_train.classes))
    model.to(device)
    summary(model, (3, 32, 32))

    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        torch.backends.cudnn.benchmark = True

    loss_fun = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learn_rate, momentum=0.9, weight_decay=5e-4)
    train_epochs = (len(x_train) + args.batch_size - 1) // args.batch_size
    warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda(train_epochs, args.warmup_epochs))
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epochs-args.warmup_epochs)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
    with SummaryWriter(log_dir=log_dir, comment='', flush_secs=5) as writer:
        input_tensor = torch.Tensor(args.batch_size, 3, 32, 32).to(device)
        writer.add_graph(model, input_tensor)

        val_top1_best, val_top5_best = 0.0, 0.0
        for epoch in range(args.train_epochs):
            epoch += 1
            # Train
            model.train()
            loss_list, y_list, p_list = [], [], []
            idx = torch.randperm(len(y_train), device=device)
            for i in tqdm(range(0, len(idx), args.batch_size), desc=f'Epoch {epoch}', unit=' Batch', ncols=0):
                i = idx[i:i + args.batch_size]
                x, y = x_train[i], y_train[i]
                x = train_transform(x)
                p = model(x)
                
                optimizer.zero_grad()
                loss = loss_fun(p, y)
                loss.backward()
                optimizer.step()

                if epoch <= args.warmup_epochs:
                    LR = optimizer.param_groups[0]["lr"]
                    print(f' LR: {LR:.6f}', end='\r\n', flush=True)
                    time.sleep(0.1)
                    warmup.step()

                loss_list.append(loss.cpu().detach().numpy())
                y_list.append(y.cpu().detach().numpy())
                p_list.append(p.cpu().detach().numpy())
            
            train_loss = np.array(loss_list).mean()
            p = np.concatenate(p_list)
            y = np.concatenate(y_list)
            train_top1 = top_k_acc(p, y, 1)
            # val
            model.eval()
            loss_list, y_list, p_list = [], [], []
            idx = torch.arange(len(y_val), device=device)
            for i in tqdm(range(0, len(idx), args.batch_size), desc=f'Epoch {epoch}', unit=' Batch', ncols=0):
                i = idx[i:i + args.batch_size]
                x, y = x_val[i], y_val[i]
                x = val_transform(x)
                p = model(x)

                loss = loss_fun(p, y)

                loss_list.append(loss.cpu().detach().numpy())
                y_list.append(y.cpu().detach().numpy())
                p_list.append(p.cpu().detach().numpy())
            
            val_loss = np.array(loss_list).mean()
            p = np.concatenate(p_list)
            y = np.concatenate(y_list)
            val_top1 = top_k_acc(p, y, 1)
            val_top5 = top_k_acc(p, y, 5)
            # print & log
            LR = optimizer.param_groups[0]["lr"]
            print(f'Epoch {epoch}/{args.train_epochs}, Train: {{LR: {LR:.6f}, Loss: {train_loss:.6f}, Top1: {train_top1:.4f}}}, Val: {{Loss: {val_loss:.6f}, Top1: {val_top1:.4f}, Top5: {val_top5:.4f}}}')
            writer.add_scalar('Train/LR', LR, epoch)
            writer.add_scalar('Train/Loss', train_loss, epoch)
            writer.add_scalar('Train/Top1', train_top1, epoch)
            writer.add_scalar('Test/Loss', val_loss, epoch)
            writer.add_scalar('Test/Top1', val_top1, epoch)
            writer.add_scalar('Test/Top5', val_top5, epoch)
            # save
            if val_top1 > val_top1_best or val_top5 > val_top5_best:
                val_top1_best = val_top1
                val_top5_best = val_top5
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'train_top1': train_top1,
                    'val_loss': val_loss,
                    'val_top1': val_top1,
                    'val_top5': val_top5,
                }, f'{log_dir}/acc_best.pt')
            # scheduler
            if epoch >= args.warmup_epochs:
                scheduler.step()