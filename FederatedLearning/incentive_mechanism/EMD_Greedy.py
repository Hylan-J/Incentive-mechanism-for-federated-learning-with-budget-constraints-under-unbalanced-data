from math import e
from .utils import *


def EMD_Greedy(server, clients):
    # 获取客户端个数
    num = len(clients)
    # 初始化分配向量X
    X = [0] * num
    # 初始化支付向量P
    P = [0.0] * num

    # 论文中的delta（模型精度相关信息）
    delta = []
    # 论文中的 c/delta
    sigma = []
    for client in clients:
        delta_each = 1 / e ** client.EMD
        delta.append(delta_each)
        sigma.append(client.C / delta_each)

    # 根据sigma从小到大排序
    sigma_sorted_value, sigma_sorted_index = sort_with_value(sigma)
    # 对应的delta根据其排序
    delta_sorted = restore_with_index(delta, sigma_sorted_index)

    for j in range(1, num + 1):
        _index_ = j - 1
        if sigma_sorted_value[_index_] > (server.R / sum(delta_sorted[:_index_ + 1])):
            for i in range(1, j):
                __index__ = i - 1
                # 计算支付向量 P
                P[__index__] = delta_sorted[__index__] * sigma_sorted_value[__index__]
                X[__index__] = 1
            break
        else:
            P[_index_] = delta_sorted[_index_] * sigma_sorted_value[_index_]
            X[_index_] = 1

    # 返回被挑选结果，对应X
    X = restore_with_index(X, sigma_sorted_index)
    # 返回原报价，对应P
    P = restore_with_index(P, sigma_sorted_index)

    return X, P
