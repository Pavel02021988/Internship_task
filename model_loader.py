# пакеты для загрузки и преобразования даных
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms
from torchvision import transforms
from json import dumps
import sys
from PIL import Image
from os import listdir

# Создаем класс основанный на классе Dataset pytorch, который будет преобразовывать входные данные
class Dataset(Dataset):

    def __init__(self, files_path, transforms=None):
        self.files_path = files_path
        self.img_list = listdir(files_path)
        self.data_len = len(self.img_list)
        self.transforms = transforms

    def __getitem__(self, index):
        image_path = self.img_list[index]
        img = Image.open(self.files_path + '/' + image_path)
        if self.transforms is not None:
            img_transform = self.transforms(img)
        return (img_transform, image_path)

    def __len__(self):
        return (self.data_len)

# функция для загрузки модели
def load_model(model_path = 'model.pth'):
    model = torch.load(model_path, map_location=device)
    model.eval()
    return model

def dataset_loader(data_path):
    transformations = transforms.Compose([transforms.Resize((224,224)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                        ])
    custom_dataset = Dataset(data_path, transformations)
    dataset_loader = torch.utils.data.DataLoader(dataset=custom_dataset, batch_size=1, shuffle=False)
    return dataset_loader

# применяем модель и сохраняем результат в джейсон файл
def process_results(data_path):
    to_json = {}
    model = load_model()
    for i in dataset_loader(data_path):
        pred = model(i[0].to(device)).data.cpu().numpy()
        to_json[i[1][0]] = int(1-pred.argmax())*'fe' + 'male'
    with open('process_results.json', 'w') as f:
        f.write(dumps(to_json))

if __name__ == "__main__":
    if len(sys.argv) > 1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        process_results(sys.argv[1])
        print('Готово')
    else:
        print('Проверьте правильность пути')