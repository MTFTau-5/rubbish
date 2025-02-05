import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans

# 定义 CNN 模型
class CNN(nn.Module):
    def __init__(self, channels, num_classes,input_shape=(9, 1071)):
        super(CNN, self).__init__()
        self.input_shape = input_shape
        self.fc1 = nn.Linear(1071, channels)
        self.conv1 = nn.Conv1d(in_channels=9, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, num_classes)



    def forward(self, x):
        
        
        x = self.fc1(x)
        x = x.permute(0, 2, 1)  
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = self.global_avg_pool(x)
        x = x.squeeze(-1).squeeze(-1) 
        features = x  
        x = self.fc(x)
        return x, features



def kmean_jia_cnn(model, train_loader, device, num_clusters):
    model.eval()
    all_features = []
    with torch.no_grad():
        for inputs, _ in train_loader:
            
            inputs = inputs.to(device).float()
            _, features = model(inputs)
            all_features.extend(features.cpu().numpy())
    all_features = np.array(all_features)
    
    kmeans = KMeans(n_clusters=num_clusters, n_init=20)
    kmeans.fit(all_features)
    cluster_centers = kmeans.cluster_centers_
    cluster_centers = torch.tensor(cluster_centers, dtype=torch.float32).to(device)
    
    return cluster_centers


def target_mubiao(q):
    weight = q ** 2 / q.sum(0)
    p = (weight.t() / weight.sum(1)).t()
    return p
