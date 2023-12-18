import numpy as np
from scipy.optimize import minimize


def generate_data(EMD_desired, data_num, classes_num):
    def objective(x):
        # 计算列表元素与均匀分布之间的平方差和再开根
        norm_diff = abs(np.sqrt(np.sum((x - [0.1] * len(x)) ** 2)) - EMD_desired)
        return norm_diff

    def constraint1(x):
        # 约束条件：列表元素之和为1
        return np.sum(x) - 1.0

    def constraint2(x):
        # 约束条件：列表元素在[0, 1]内
        return np.array(x) - 0

    def adjust_sum(arr):
        """
        微调数据，以防四舍五入时会丢失一点精度。
        """
        diff = data_num - np.sum(arr)  # 计算与目标和的差值
        random_index = np.random.randint(0, len(arr))
        arr[random_index] += diff  # 将差值加到随机素上
        return arr

    # 随机生成初始猜测值
    x0 = np.random.rand(classes_num)

    # 设置约束条件
    cons = [{'type': 'eq', 'fun': constraint1},
            {'type': 'ineq', 'fun': constraint2}]

    # 进行优化
    sol = minimize(objective, x0, method='SLSQP', constraints=cons)
    # print(objective(sol.x))

    if sol.success:
        return True, adjust_sum(np.round(sol.x * data_num))
    else:
        print("未找到满足条件的列表")
        return False, None


def cal_EMD(x):
    return np.sqrt(np.sum((x - [0.1] * len(x)) ** 2))


def check_data(divide_type, EMDs, distributions):
    num_classes = len(distributions[0])
    threshold = 0.5
    over_threshold = 0
    below_threshold = 0
    for EMD in EMDs:
        if EMD >= threshold:
            over_threshold += 1
        else:
            below_threshold += 1
    if divide_type == "A":
        if below_threshold != over_threshold:
            return False
    elif divide_type == "B":
        if below_threshold != 15:
            return False
    elif divide_type == "C":
        if below_threshold != 20:
            return False

    """for i in range(num_classes):
        if sum(distributions[:][i]) < 300:
            return False"""
    return True
