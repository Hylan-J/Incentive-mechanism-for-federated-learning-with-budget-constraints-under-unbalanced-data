import torch
from torch.utils.data import DataLoader


class Server:
    def __init__(self, testset, device, net, R):
        """
        服务器端测试相关参数
        """
        # 测试集
        self.testset = testset
        # 设备
        self.device = device
        # 网络模型
        self.net = net.to(self.device)
        # 批处理大小
        self.batch_size = 10
        # 服务器端预算
        self.R = R

    def evaluate(self):
        # 加载训练数据集
        test_dataloader = DataLoader(dataset=self.testset, batch_size=self.batch_size, shuffle=False)
        # 模型评估
        self.net.eval()

        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        return accuracy
