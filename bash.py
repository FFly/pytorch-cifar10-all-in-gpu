import os

if __name__ == "__main__":
    model_list = [
        'VGG16', 
        'ResNet18', 
        'ResNet50', 
        'ResNet101', 
        'RegNetX_200MF', 
        'RegNetY_400MF', 
        'MobileNetV2', 
        'ResNeXt29(32x4d)', 
        'ResNeXt29(2x64d)', 
        'SimpleDLA', 
        'DenseNet121', 
        'PreActResNet18', 
        'DPN92', 
        'DLA'
    ]
    
    for model_name in model_list:
        os.system(f'python.exe train.py --log_dir=logs --mode_name={model_name}')
