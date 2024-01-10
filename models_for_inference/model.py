########### АРХИТЕКТУРА СЕТИ ###############
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class Tnet(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.k=k
        self.conv1 = nn.Conv1d(k,64,1)
        self.conv2 = nn.Conv1d(64,128,1)
        self.conv3 = nn.Conv1d(128,1024,1)
        self.fc1 = nn.Linear(1024,512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,k*k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, input):
        # input.shape == (bs,n,3)
        bs = input.size(0)
        xb = F.relu(self.bn1(self.conv1(input)))
        xb = F.relu(self.bn2(self.conv2(xb)))
        xb = F.relu(self.bn3(self.conv3(xb)))
        pool = nn.MaxPool1d(xb.size(-1))(xb)
        flat = nn.Flatten(1)(pool)
        xb = F.relu(self.bn4(self.fc1(flat)))
        xb = F.relu(self.bn5(self.fc2(xb)))

        #initialize as identity
        init = torch.eye(self.k, requires_grad=True).repeat(bs,1,1)
        if xb.is_cuda:
            init=init.cuda()
        matrix = self.fc3(xb).view(-1,self.k,self.k) + init
        return matrix


class Transform(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_transform = Tnet(k=3)
        self.feature_transform = Tnet(k=64)
        self.conv1 = nn.Conv1d(3,64,1)

        self.conv2 = nn.Conv1d(64,128,1)
        self.conv3 = nn.Conv1d(128,1024,1)


        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, input):
        matrix3x3 = self.input_transform(input)
        # batch matrix multiplication
        xb = torch.bmm(torch.transpose(input,1,2), matrix3x3).transpose(1,2)

        xb = F.relu(self.bn1(self.conv1(xb)))

        matrix64x64 = self.feature_transform(xb)
        xb = torch.bmm(torch.transpose(xb,1,2), matrix64x64).transpose(1,2)

        xb = F.relu(self.bn2(self.conv2(xb)))
        xb = self.bn3(self.conv3(xb))
        xb = nn.MaxPool1d(xb.size(-1))(xb)
        output = nn.Flatten(1)(xb)
        return output, matrix3x3, matrix64x64

class PointNet(nn.Module):
    def __init__(self, classes = 2):
        super().__init__()
        self.transform = Transform()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, classes)
        

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        xb, matrix3x3, matrix64x64 = self.transform(input)
        xb = F.relu(self.bn1(self.fc1(xb)))
        xb = F.relu(self.bn2(self.dropout(self.fc2(xb))))
        output = self.fc3(xb)
        return self.logsoftmax(output), matrix3x3, matrix64x64
    





#####################################
# Реализация нормального шума
class RandomNoise(object):
    def __init__(self, std=0.002):
        self.std = std
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2

        noise = np.random.normal(0, self.std, (pointcloud.shape))
    
        noisy_pointcloud = pointcloud + noise
        return  noisy_pointcloud


class ToTensor(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2

        return torch.from_numpy(pointcloud)

# Реализация нормализации чтобы максильное значение равнялось 1 (схоже с методом в изображениях)
class Normalize(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2
        
        norm_pointcloud = pointcloud - np.mean(pointcloud, axis=0) 
        norm_pointcloud /= np.max(np.linalg.norm(norm_pointcloud, axis=1))

        return  norm_pointcloud


# Создание рандомного семплирования (если точек меньше то просто добавлю дубликаты чтобы добить число)
class PointSampler:
    def __init__(self, num_points=512): 
        self.num_points = num_points

    def __call__(self, pointcloud):
        num_total_points = pointcloud.shape[0]
        
        if num_total_points < self.num_points:
              
            # Семплирование оставшихся точек из уже существующих точек
            sampled_indices = np.random.choice(num_total_points, self.num_points - num_total_points, replace=True)
            duplicated_points = pointcloud[sampled_indices, :]
            
            # Собираем все точки в итоговую выборку
            sampled_points = np.vstack((pointcloud, duplicated_points))
            #print("Всего точек меньше чем надо семплировать - будут дубликаты")
        else:
            # Случайно берем точки:
            sampled_indices = np.random.choice(num_total_points, self.num_points, replace=False)
            sampled_points = pointcloud[sampled_indices, :]
        
        return sampled_points

# Создание рандомного семплирования  с попыткой брать больше точек что дальше лежат от (0,0) 
# (если точек меньше то просто добавлю дубликаты чтобы добить число)
class PointSampler_weighted:
    def __init__(self, num_points=512): 
        self.num_points = num_points

    def __call__(self, pointcloud):
        num_total_points = pointcloud.shape[0]
        
        if num_total_points < self.num_points:
              
            # Семплирование оставшихся точек из уже существующих точек
            sampled_indices = np.random.choice(num_total_points, self.num_points - num_total_points, replace=True)
            duplicated_points = pointcloud[sampled_indices, :]
            
            # Собираем все точки в итоговую выборку
            sampled_points = np.vstack((pointcloud, duplicated_points))
            #print("Всего точек меньше чем надо семплировать - будут дубликаты")
        else:
            # Выше вероятность чем дальше от 0,0
            distances = np.linalg.norm(pointcloud, axis=1)
            weights = distances / np.sum(distances)

            sampled_indices = np.random.choice(num_total_points, self.num_points, replace=False, p=weights)
            sampled_points = pointcloud[sampled_indices, :]
        
        return sampled_points