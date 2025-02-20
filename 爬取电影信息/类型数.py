# 打开文件并读取数据
with open('movies', 'r', encoding='utf-8') as file:
    # 初始化一个空列表来保存所有电影类型
    genres_list = []

    # 逐行读取文件
    for line in file:
        # 分割每行数据，获取电影类型部分
        # 格式：2::Jumanji (1995)::Adventure|Children's|Fantasy
        parts = line.strip().split('::')
        if len(parts) >= 3:  # 确保数据格式正确
            genres = parts[2].split('|')  # 获取电影类型并分割
            genres_list.extend(genres)  # 将类型添加到列表中

    # 统计电影类型的总数（包括重复类型）
    total_genres_count = len(genres_list)

    # 打印结果
    print("所有电影类型列表:", genres_list)
    print("电影类型总数（包括重复）:", len(set(genres_list)))