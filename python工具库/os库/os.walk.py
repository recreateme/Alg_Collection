import os

'''
    每次迭代返回一个三元组(dirpath, dirnames, filenames)：
    dirpath：当前遍历目录的完整路径（字符串）
    dirnames：当前目录下的子目录名称列表（不包含路径）
    filenames：当前目录下的文件名称列表（不包含路径）
'''
example_dir = r'D:\Develop\tianchi\train-data'
i = 0
for root, dirs, files in os.walk(example_dir):
    i += 1
    print(f"当前目录：{root}")
    print(f"子目录：{dirs}")
    print(f"文件：{files}")
    print("----")

print(i)