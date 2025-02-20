import numpy as np

# 目标函数
def fitness(x):
    return x * np.sin(4 * x) + 3

# 粒子群算法
def particle_swarm_optimization(population_size=50, max_iter=100, x_min=-10, x_max=10):
    # 初始化粒子群
    positions = np.random.uniform(x_min, x_max, (population_size, 1))
    velocities = np.zeros((population_size, 1))
    personal_best_positions = positions.copy()
    personal_best_fitness = np.array([fitness(x) for x in positions])
    global_best_position = positions[np.argmax(personal_best_fitness)]
    global_best_fitness = np.max(personal_best_fitness)

    # 迭代优化
    for _ in range(max_iter):
        # 更新速度和位置
        w = 0.8  # 惯性权重
        c1 = 2  # 个体学习因子
        c2 = 2  # 群体学习因子
        r1 = np.random.rand(population_size, 1)
        r2 = np.random.rand(population_size, 1)
        velocities = w * velocities + c1 * r1 * (personal_best_positions - positions) + c2 * r2 * (global_best_position - positions)
        positions = positions + velocities

        # 更新个体最优和全局最优
        fitness_values = np.array([fitness(x) for x in positions])
        personal_best_mask = fitness_values > personal_best_fitness
        personal_best_positions[personal_best_mask] = positions[personal_best_mask]
        personal_best_fitness[personal_best_mask] = fitness_values[personal_best_mask]
        global_best_index = np.argmax(personal_best_fitness)
        global_best_position = personal_best_positions[global_best_index]
        global_best_fitness = personal_best_fitness[global_best_index]

    return global_best_position, global_best_fitness

# 主程序
if __name__ == "__main__":
    best_position, best_fitness = particle_swarm_optimization()
    print("Best position:", best_position)
    print("Best fitness:", best_fitness)
