import numpy as np
import random
import math

# 定义目标函数
def objective_function(x):
    return x ** 2 + 1e-10  # 添加小偏移量避免除以 0

# 定义模拟退火算法
def simulated_annealing(initial_temp, final_temp, cooling_rate, num_iterations):
    # 初始化当前解和最优解
    current_solution = random.uniform(-10, 10)
    best_solution = current_solution

    # 开始迭代
    temperature = initial_temp
    for _ in range(num_iterations):
        # 生成新解
        new_solution = current_solution + random.uniform(-1, 1)

        # 计算目标函数值的差异
        current_value = objective_function(current_solution)
        new_value = objective_function(new_solution)
        if current_value == 0 and new_value == 0:
            delta = 0  # 如果当前值和新值都为 0,则将差异设为 0
        else:
            delta = new_value - current_value

        # 根据温度决定是否接受新解
        # if delta < 0 or random.uniform(0, 1) < math.exp(-delta / temperature):
        if delta < 0 or random.uniform(0,1)< 0.02:
            current_solution = new_solution

        # 更新最优解
        if objective_function(current_solution) < objective_function(best_solution):
            best_solution = current_solution

        # 降温
        temperature *= (1 - cooling_rate)

    return best_solution

# 主程序
if __name__ == "__main__":
    initial_temperature = 100
    final_temperature = 1
    cooling_rate = 0.95
    num_iterations = 1000

    best_solution = simulated_annealing(initial_temperature, final_temperature, cooling_rate, num_iterations)
    print("Best solution:", best_solution)
    print("Best objective function value:", objective_function(best_solution))
