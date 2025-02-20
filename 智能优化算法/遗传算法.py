import numpy as np


# 目标函数：求解 x^2 的最大值
def fitness_function(x):
    return x ** 2


def encode(x, bits):
    # 将解进行二进制编码
    x_int = int((x + 5) * (2 ** bits) / 10)  # 将x映射到[0, 2^bits]
    return format(x_int, '0' + str(bits) + 'b')  # 转换为二进制字符串


# 解码：将二进制字符串转换为实数
def decode(binary_str):
    x_int = int(binary_str, 2)
    return x_int * 10 / (2 ** bits) - 5


# 初始化种群
bits = 10  # 每个染色体的二进制位数
population_size = 10  # 种群大小
population = [encode(np.random.uniform(0, 5), bits) for _ in range(population_size)]

# 遗传算法主循环
max_iterations = 500  # 最大迭代次数
for iteration in range(max_iterations):
    # 评估适应度
    fitness_scores = [fitness_function(decode(individual)) for individual in population]

    # 选择
    selected = np.random.choice(population, size=population_size, replace=True,
                                p=np.array(fitness_scores) / sum(fitness_scores))

    # 交叉
    np.random.shuffle(selected)  # 打乱种群以避免相同的个体总是相互交叉
    crossover_point = np.random.randint(1, bits)  # 随机选择交叉点
    offspring = [selected[i][:crossover_point] + selected[j][crossover_point:] for i, j in
                 zip(range(0, population_size, 2), range(1, population_size, 2))]

    # 变异
    mutation_rate = 0.05  # 变异率
    for i in range(len(offspring)):
        if np.random.rand() < mutation_rate:
            mutation_point = np.random.randint(0, bits - 1)  # 避免索引越界
            offspring[i] = offspring[i][:mutation_point] + ('0' if offspring[i][mutation_point] == '1' else '1') + \
                           offspring[i][mutation_point + 1:]

    # 更新种群
    population = offspring

    # 打印当前最优解
    fitness_scores = [fitness_function(decode(individual)) for individual in population]  # 重新计算适应度
    best_fitness = max(fitness_scores)
    best_individual = population[fitness_scores.index(best_fitness)]
    print(
        f"Iteration {iteration + 1}: Best fitness = {best_fitness}, Best individual = {best_individual}, Decoded x = {decode(best_individual)}")

# 输出最终结果
fitness_scores = [fitness_function(decode(individual)) for individual in population]  # 重新计算适应度
best_fitness = max(fitness_scores)
best_individual = population[fitness_scores.index(best_fitness)]
print(
    f"Final result: Best fitness = {best_fitness}, Best individual = {best_individual}, Decoded x = {decode(best_individual)}")
