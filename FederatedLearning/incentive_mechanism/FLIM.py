from .utils import *


def FLIM(server, clients):
    # 获取客户端个数
    num = len(clients)

    X, P = [0] * num, [0.0] * num

    # 获得客户端的报价
    C = [client.C for client in clients]
    # 获取排序后的报价
    C_sorted_value, C_sorted_index = sort_with_value(C)

    # 1<=class_num<=num_clients
    for j in range(1, num + 1):
        _index_ = j - 1
        # 如果第j个客户端（下标j-1）的报价大于分给j个客户端的平均预算
        if C_sorted_value[_index_] > (server.R / j):
            # 1<=index<=class_num-1
            for i in range(1, j):
                __index__ = i - 1
                # 第i个客户端（下标i-1）
                P[__index__] = min(server.R / (j - 1), C_sorted_value[__index__])
                X[__index__] = 1
            break
        else:
            P[_index_] = C_sorted_value[_index_]
            X[_index_] = 1

    # 返回原报价对应X
    X = restore_with_index(X, C_sorted_index)
    # 返回原报价对应的P
    P = restore_with_index(P, C_sorted_index)

    return X, P