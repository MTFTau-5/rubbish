import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from feeder import Feeder
import time
import numpy as np
import os
import torch.nn.functional as F
from util import yaml_parser
from sklearn.cluster import KMeans
from net import CNN, BYOL
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    (  
        train_data_path, 
        test_data_path, 
        valid_data_path, 
        valid_label_path,
        train_label_path,
        batch_size,
        epochs,
        cnn_channels,
        num_classes,
        num_clusters,
        update_interval,
        num_epochs,
        lr

    ) = yaml_parser()
    
    train_data_path = '/home/mtftau-5/workplace/shl-code/data/fft_data/train/data.npy'
    train_label_path = '/home/mtftau-5/workplace/shl-code/data/fft_data/train/label.npy'
    test_data_path = '/home/mtftau-5/workplace/shl-code/data/fft_data/test/data.npy'
    valid_label_path = '/home/mtftau-5/workplace/shl-code/data/fft_data/valid/label.npy'
    valid_data_path = '/home/mtftau-5/workplace/shl-code/data/fft_data/valid/data.npy'

    # 创建数据集实例
    train_dataset = Feeder(train_data_path, train_label_path)
    test_dataset = Feeder(test_data_path)
    valid_dataset = Feeder(valid_data_path, valid_label_path)

    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size , shuffle=False)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    for inputs, labels in train_loader:
        print(f"输入数据的形状: {inputs.shape}")  # 检查输入数据的形状
        break

    # 定义模型参数
    cnn_channels = cnn_channels
    num_classes = num_classes
    num_clusters = num_classes  # 聚类的类别数

    # 初始化 BYOL 模型
    byol_model = BYOL(cnn_channels, num_classes)

    # 定义设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    byol_model = byol_model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(byol_model.parameters(), lr=lr)

    # 训练参数
    num_epochs = epochs
    best_valid_loss = float('inf')
    update_interval = update_interval

    # 初始化聚类中心
    def initialize_cluster_centers(byol_model, train_loader, device, num_clusters):
        byol_model.eval()
        all_features = []
        with torch.no_grad():
            for inputs, _ in train_loader:
                inputs = inputs.to(device).float()
                # 使用新添加的 extract_features 方法获取特征
                features = byol_model.extract_features(inputs)
                all_features.extend(features.cpu().numpy())
        all_features = np.array(all_features)
        kmeans = KMeans(n_clusters=num_clusters, n_init=20)
        kmeans.fit(all_features)
        cluster_centers = kmeans.cluster_centers_
        return torch.tensor(cluster_centers, dtype=torch.float32).to(device)

    cluster_centers = initialize_cluster_centers(byol_model, train_loader, device, num_clusters)

    # 计算目标分布
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        p = (weight.t() / weight.sum(1)).t()
        return p

    # 训练循环
    for epoch in range(num_epochs):
        byol_model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        if epoch % update_interval == 0:
            # 更新聚类中心
            cluster_centers = initialize_cluster_centers(byol_model,
                                                         train_loader,
                                                         device,
                                                         num_clusters)

        for inputs, labels in train_loader:
            inputs = inputs.to(device).float()
            labels = labels.to(device)

            optimizer.zero_grad()
            features = byol_model.extract_features(inputs)
            outputs = byol_model.fc(features)

            # 计算软分配
            q = 1.0 / (1.0 + torch.sum((features.unsqueeze(1) - cluster_centers) ** 2, dim=2))
            q = q ** ((1 + 1) / 2)
            q = (q.t() / torch.sum(q, dim=1)).t()

            # 计算目标分布
            p = target_distribution(q)

            # 计算聚类损失
            clustering_loss = F.kl_div(q.log(), p, reduction='batchmean')

            classification_loss = criterion(outputs, labels)

            # 计算 BYOL 损失
            byol_loss = byol_model(inputs, inputs).mean()

            loss = classification_loss + clustering_loss + byol_loss

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_loss /= len(train_loader)
        train_accuracy = 100. * train_correct / train_total


        byol_model.eval()
        valid_loss = 0
        valid_correct = 0
        valid_total = 0

        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs = inputs.to(device).float()
                labels = labels.to(device)
                # 只接收一个返回值
                features = byol_model.extract_features(inputs)
                # 手动计算分类输出
                outputs = byol_model.fc(features)
                loss = criterion(outputs, labels)

                valid_loss += loss.item()
                _, predicted = outputs.max(1)
                valid_total += labels.size(0)
                valid_correct += predicted.eq(labels).sum().item()

        valid_loss /= len(valid_loader)
        valid_accuracy = 100. * valid_correct / valid_total

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_accuracy:.2f}%')
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            # 获取当前时间戳
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            # 构建带有时间戳的保存路径
            save_path = f'/home/mtftau-5/workplace/shl-code/output/output_{timestamp}.pth'
            torch.save(byol_model.state_dict(), save_path)
            print(f'Best model saved at {save_path}')


    byol_model.eval()
    model_dir = '/home/mtftau-5/workplace/shl-code/output'
    model_files = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.startswith('output_') and f.endswith('.pth')]
    if not model_files:
        raise ValueError("No saved model files found.")
    latest_model_file = max(model_files, key=os.path.getctime)

    byol_model.load_state_dict(torch.load(latest_model_file))
    byol_model.eval()

    all_predictions = []

    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device).float()
            # 只接收一个返回值
            features = byol_model.extract_features(inputs)
            # 手动计算分类输出
            outputs = byol_model.fc(features)
            _, predicted = outputs.max(1)
            all_predictions.extend(predicted.cpu().tolist())

    np.save('test_predictions.npy', all_predictions)
if __name__ == '__main__':
    main()