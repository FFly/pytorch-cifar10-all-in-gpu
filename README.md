# pytorch-cifar10-all-in-gpu

## 运行
* python train.py --log_dir=base_logs --mode_name=VGG16
* python train.py --log_dir=base_logs --mode_name=ResNet18
* python train.py --log_dir=base_logs --mode_name=ResNet50
* python train.py --log_dir=base_logs --mode_name=ResNet101
* python train.py --log_dir=base_logs --mode_name=RegNetX_200MF
* python train.py --log_dir=base_logs --mode_name=RegNetY_400MF
* python train.py --log_dir=base_logs --mode_name=MobileNetV2
* python train.py --log_dir=base_logs --mode_name=ResNeXt29(32x4d)
* python train.py --log_dir=base_logs --mode_name=ResNeXt29(2x64d)
* python train.py --log_dir=base_logs --mode_name=SimpleDLA
* python train.py --log_dir=base_logs --mode_name=DenseNet121
* python train.py --log_dir=base_logs --mode_name=PreActResNet18
* python train.py --log_dir=base_logs --mode_name=DPN92
* python train.py --log_dir=base_logs --mode_name=DLA

## 查看
* tensorboard.exe --logdir=base_logs

## 成绩
| 模型           | Top1  |
|----------------|-------|
| ResNet18       | 93.75 |
| MobileNetV2    | 91.69 |