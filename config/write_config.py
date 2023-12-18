import yaml

config = {
    "Dataset_Parameter": {
        "MNIST":
            {"batch_size": 10,
             "local_epochs": 5,
             "learning_rate": 0.01},
        "CIFAR_10":
            {"batch_size": 10,
             "local_epochs": 5,
             "learning_rate": 0.1}
    },

    "Num_Clients": 20,
    "Num_Classes": 10,
    "Num_Data": 1000,

    "Options_Dataset": ["MNIST", "CIFAR_10"],
    "Options_Divide": ["A", "B", "C"],
    "Options_Aggregation_Algorithm": ["FedAvg"],
    "Options_Incentive_Mechanism": ["FMore", "FLIM", "EMD_Greedy", "EMD_FLIM"],

    "EMD_Desired_Group": {
        "A": [0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6,
              0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
        "B": [0.7, 0.7, 0.7, 0.7, 0.7, 0.3, 0.3, 0.3, 0.3, 0.3,
              0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
        "C": [0.27]*20
    }
}

with open('config.yaml', 'w') as file:
    yaml.dump(config, file)