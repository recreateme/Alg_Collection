import numpy as np
import random

# 定义城市之间的距离矩阵
DISTANCES = np.array([[0, 2, 3, 1, 4],
                      [2, 0, 1, 3, 2],
                      [3, 1, 0, 4, 1],
                      [1, 3, 4, 0, 2],
                      [4, 2, 1, 2, 0]])

# 定义蚁群算法的参数
NUM_CITIES = 5
NUM_ANTS = 10
NUM_ITERATIONS = 100
ALPHA = 1  # 信息素重要程度因子
BETA = 2  # 启发式因子
RHO = 0.5  # 信息素蒸发系数
Q = 100  # 信息素增加强度


# 定义蚁群算法的函数
def ant_colony_optimization():
    # 初始化信息素矩阵
    pheromone = np.ones((NUM_CITIES, NUM_CITIES))

    best_tour = None
    best_tour_length = float('inf')

    for _ in range(NUM_ITERATIONS):
        # 每只蚂蚁构建一条路径
        for _ in range(NUM_ANTS):
            tour = [i for i in range(NUM_CITIES)]
            random.shuffle(tour)
            tour_length = calculate_tour_length(tour)

            if tour_length < best_tour_length:
                best_tour = tour.copy()
                best_tour_length = tour_length

            # 更新信息素
            for i in range(NUM_CITIES):
                for j in range(NUM_CITIES):
                    if i != j:
                        pheromone[i][j] *= (1 - RHO)
                        pheromone[i][j] += RHO * Q / tour_length

    return best_tour, best_tour_length


def calculate_tour_length(tour):
    total_distance = 0
    for i in range(NUM_CITIES):
        total_distance += DISTANCES[tour[i]][tour[(i + 1) % NUM_CITIES]]
    return total_distance


# 主程序
if __name__ == "__main__":
    best_tour, best_tour_length = ant_colony_optimization()
    print("Best tour:", best_tour)
    print("Best tour length:", best_tour_length)
