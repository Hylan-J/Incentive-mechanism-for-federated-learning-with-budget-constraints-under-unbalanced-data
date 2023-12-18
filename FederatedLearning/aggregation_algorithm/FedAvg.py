def FedAvg(server, clients):
    # 将客户端模型参数取平均并更新全局模型
    global_params = server.net.state_dict()
    for client in clients:
        client_params = client.net.state_dict()
        for param_name in global_params:
            global_params[param_name] += client_params[param_name]
        # 可以添加权重的调整，根据每个客户端的样本数量等进行调整
    for param_name in global_params:
        global_params[param_name] /= len(clients)
    server.net.load_state_dict(global_params)