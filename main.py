import yaml
import random
from copy import deepcopy
import matplotlib.pyplot as plt

import torch.utils.data
import torchvision.datasets
from torchvision import transforms

from DeepLearning.models.CIFAR_10 import CIFAR_10
from DeepLearning.models.MNIST import MNIST

from utils import *
from FederatedLearning.objects.Client import Client
from FederatedLearning.objects.Server import Server
from FederatedLearning.aggregation_algorithm import *
from FederatedLearning.incentive_mechanism import *

########################################################################################################################
# 画图设置
# -------------------------------------------------------------------------------------------------------------------- #
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题

########################################################################################################################
# 导入配置参数
# -------------------------------------------------------------------------------------------------------------------- #
with open("config/config.yaml", 'r') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
    file.close()


class Experiment:
    def __init__(self, dataset, device, divide):

        self.dataset = dataset
        self.device = device
        self.divide = divide

        if self.dataset == config["Options_Dataset"][0]:  # 如果选择了MNIST数据集
            self.trainset = torchvision.datasets.MNIST(root='./DeepLearning/datasets',
                                                       train=True,
                                                       transform=transforms.ToTensor())
            self.testset = torchvision.datasets.MNIST(root='./DeepLearning/datasets',
                                                      train=False,
                                                      transform=transforms.ToTensor())
            self.model = MNIST()

        elif self.dataset == config["Options_Dataset"][1]:  # 如果选择了CIFAR—10数据集
            self.trainset = torchvision.datasets.CIFAR10(root='./DeepLearning/datasets',
                                                         train=True,
                                                         transform=transforms.ToTensor())
            self.testset = torchvision.datasets.CIFAR10(root='./DeepLearning/datasets',
                                                        train=False,
                                                        transform=transforms.ToTensor())
            self.model = CIFAR_10()

        self.hyperparameter = {
            "learning_rate": config["Dataset_Parameter"][self.dataset]["learning_rate"],
            "local_epochs": config["Dataset_Parameter"][self.dataset]["local_epochs"],
            "batch_size": config["Dataset_Parameter"][self.dataset]["batch_size"]
        }

        ################################################################################################################
        # 初始化客户端和服务器端
        # ------------------------------------------------------------------------------------------------------------ #
        # 初始化服务器端
        test_index = []
        for i in range(config["Num_Classes"]):
            test_indices = torch.where(torch.Tensor(self.testset.targets) == i)[0]
            test_index.extend(test_indices[:10])
        test_data = torch.utils.data.Subset(self.testset, test_index)
        """self.server = Server(testset=test_data,
                             device=self.device,
                             net=globals()[self.dataset](),
                             R=0)"""
        self.server = Server(testset=self.testset,
                             device=self.device,
                             net=globals()[self.dataset](),
                             R=0)
        # 获取服务器端初始化参数
        divided = False
        EMDs = [0.0] * config["Num_Clients"]
        distributions = [[]] * config["Num_Clients"]
        # 当所有的客户端参数没有获取成功时
        while not divided:
            # 针对每个客户端循环进行
            for i in range(config["Num_Clients"]):
                # distributed: 标志位，说明数据是否根据分配规则分配好
                # distribution: 列表，共10个元素，每个元素的索引代表类别，元素值代表个数
                distributed, distribution = False, []
                # 当没分配好参数时
                while not distributed:
                    distributed, distribution = generate_data(EMD_desired=config["EMD_Desired_Group"][self.divide][i],
                                                              data_num=config["Num_Data"],
                                                              classes_num=config["Num_Classes"])

                # 获取约束优化情况下的数据分配情况
                distribution = [int(distribution_data_) for distribution_data_ in distribution]
                distributions[i] = distribution
                EMDs[i] = cal_EMD(distribution / np.sum(distribution) * 1.0)
            # 检验分配出的EMD是否符合划分规则
            divided = check_data(self.divide, EMDs, distributions)

        print("\r-----------------------------------------------------------------")
        print("|\033[93m{:^63s}\033[0m|".format("clients init"))
        print("-----------------------------------------------------------------")

        # 初始化客户端
        self.clients = []
        num_index_used = [0] * config["Num_Classes"]
        for i in range(config["Num_Clients"]):
            train_index = []
            for num_data in distributions[i]:
                # 如果某个类的分配数量大于0
                if num_data > 0:
                    # 获取当前数据对应的类别
                    target_class = distributions[i].index(num_data)
                    # 获取训练集对应类的索引
                    train_indices = torch.where(torch.Tensor(self.trainset.targets) == target_class)[0]
                    range_left = num_index_used[target_class]
                    range_right = num_index_used[target_class] + num_data
                    train_index.extend(train_indices[range_left:range_right])
                    num_index_used[target_class] += num_data

            train_data = torch.utils.data.Subset(self.trainset, train_index)
            delta = 1 / (e ** EMDs[i])
            C = random.uniform(4 * delta, 6 * delta)
            client = Client(id=i,
                            dataset=train_data,
                            EMD=EMDs[i],
                            device=self.device,
                            net=globals()[self.dataset](),
                            hyperparameter=self.hyperparameter,
                            C=10)
            self.clients.append(client)
            print("| client {:2d} | EMD: {:8f}     δ: {:8f}     quote: {:8f} |".format(i, EMDs[i], delta, client.C))
        print("-----------------------------------------------------------------\n")

    def reset_server_clients(self, R):
        # 重置服务器
        self.server.net.load_state_dict(deepcopy(self.model.state_dict()))
        self.server.R = R
        # 重置服务器
        for client in self.clients:
            client.net.load_state_dict(deepcopy(self.model.state_dict()))
            client.C = client.cost

    def main(self, R, aggregation_algorithm, incentive_mechanism):
        # 打印实验相关信息
        print("\r---------------------------------------------")
        print("|\033[93m{:^43s}\033[0m|".format("experiment info"))
        print("---------------------------------------------")
        print("|{:>25}  |  {:<13d}|".format("budget", R))
        print("|{:>25}  |  {}         |".format("device", self.device))
        print("|{:>25}  |  {:<13s}|".format("dataset", self.dataset))
        print("|{:>25}  |  {:<13d}|".format("data num", config["Num_Data"]))
        print("|{:>25}  |  {:<13d}|".format("clients num", config["Num_Clients"]))
        print("|{:>25}  |  {:<13d}|".format("classes num", config["Num_Classes"]))
        print("|{:>25}  |  {:<13s}|".format("data divide", self.divide))
        print("|{:>25}  |  {:<13s}|".format("incentive mechanism", incentive_mechanism))
        print("|{:>25}  |  {:<13s}|".format("aggregation algorithm", aggregation_algorithm))
        print("---------------------------------------------\n")
        ################################################################################################################
        # 重置服务器和客户端
        # ------------------------------------------------------------------------------------------------------------ #
        self.reset_server_clients(R)

        R_copy = R
        federated_learning_done = False
        accuracy = 0
        epoch = 1
        # 除非服务器预算用尽或没有被挑选中
        while not federated_learning_done:
            ############################################################################################################
            # 使用激励机制挑选客户端
            # -------------------------------------------------------------------------------------------------------- #
            X, P = globals()[incentive_mechanism](self.server, self.clients)

            if R_copy > sum(P):
                # R_copy -= sum(P)
                # self.server.R -= sum(P)
                # 如果存在被挑选中的客户端
                if sum(X) != 0:
                    print("\r----------------------------------------------------------------------------------")
                    print("{:^86}".format("Epoch {:3d}  R:\033[93m{:6f}\033[0m  left:\033[93m{:6f}\033[0m").format(
                        epoch, R, self.server.R))
                    print("----------------------------------------------------------------------------------")
                    self.server.R -= sum(P)
                    clients_selected = []
                    for i in range(len(self.clients)):
                        if X[i] == 1:
                            # 训练模型
                            self.clients[i].train()
                            # 增加到被选中的客户端中
                            clients_selected.append(self.clients[i])
                            print(
                                "|client {:2d} | selected: {:1s}    quote: {:8f}    paid: {:8f}|".format(
                                    self.clients[i].id, "Y", self.clients[i].C, P[i]))
                        else:
                            print(
                                "|client {:2d} | selected: {:1s}    quote: {:8f}    paid: {:8f}|".format(
                                    self.clients[i].id, "N", self.clients[i].C, P[i]))

                    ####################################################################################################
                    # 使用聚合算法进行全局模型更新
                    # ------------------------------------------------------------------------------------------------ #
                    globals()[aggregation_algorithm](self.server, clients_selected)

                    accuracy = self.server.evaluate()
                    print("|server model accuracy: \033[93m{:57s}\033[0m|".format(str(accuracy * 100) + "%"))

                    ####################################################################################################
                    # 将服务器中的全局最优模型下发
                    # ------------------------------------------------------------------------------------------------ #
                    for client in self.clients:
                        client.net.load_state_dict(self.server.net.state_dict())

                    epoch += 1
                else:
                    federated_learning_done = True
            else:
                federated_learning_done = True
        print("----------------------------------------------------------------------------------\n")

        return accuracy


if __name__ == '__main__':
    # 实验使用设备
    #   可选：【CPU、CUDA】
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 使用的数据集
    #   可选：【MNIST、CIFAR10】
    dataset = config["Options_Dataset"][1]
    # 使用的数据划分方式
    #   可选：【A、B、C】
    divide = config["Options_Divide"][2]
    # 使用的聚合算法
    #   可选：【FedAvg】
    aggregation_algorithm = config["Options_Aggregation_Algorithm"][0]
    # 使用的激励机制
    #   可选：【FMore、FLIM、EMD_FLIM、EMD_Greedy】
    # incentive_mechanism = config["Options_Incentive_Mechanism"][1]

    # 实例化实验对象
    experiment = Experiment(dataset=dataset,
                            device=device,
                            divide=divide)

    # 设置不同的预算（从100~1000，等差100递增），观察实验情况
    Accuracy = {
        "FMore": [],
        "FLIM": [],
        "EMD_Greedy": [],
        "EMD_FLIM": []
    }
    for incentive_mechanism in config["Options_Incentive_Mechanism"]:
        for R in range(100, 1100, 100):
            Accuracy[incentive_mechanism].append(experiment.main(R, aggregation_algorithm, incentive_mechanism) * 100)

    ####################################################################################################################
    # 画图
    # ---------------------------------------------------------------------------------------------------------------- #
    x = range(100, 1100, 100)
    plt.plot(x, Accuracy["FMore"], 'b', marker='.', markersize=5, linestyle='--', linewidth=1)
    plt.plot(x, Accuracy["FLIM"], 'r', marker='*', markersize=5, linestyle='--', linewidth=1)
    plt.plot(x, Accuracy["EMD_Greedy"], 'y', marker='D', markersize=5, linestyle='--', linewidth=1)
    plt.plot(x, Accuracy["EMD_FLIM"], 'g', marker='s', markersize=5, linestyle='--', linewidth=1)
    # 折线图标题
    plt.title('实验复现')
    # x轴标题
    plt.xlabel('预算')
    # y轴标题
    plt.ylabel('准确率%')
    plt.xlim((100, 1100))
    plt.ylim((0, 110))
    plt.xticks(np.arange(100, 1100, 100))
    plt.yticks(np.arange(0, 110, 10))
    plt.legend(["FMore(truthfulness)", "FLIM(truthfulness)", "EMD_Greedy", "EMD_FLIM(truthfulness)"])
    # 绘制图例
    plt.savefig("results/{}.jpg".format(dataset + "-" + divide))
    # 显示图像
    plt.show()
