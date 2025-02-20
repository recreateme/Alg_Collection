import os


def insert_char_in_filenames(folder_path, index, ch):
    """
    在指定文件夹内所有文件的文件名中，在索引index位置添加字符ch。

    :param folder_path: 文件夹路径
    :param index: 插入字符的位置索引
    :param ch: 要插入的字符
    """
    try:
        # 遍历文件夹中的所有文件和文件夹
        for filename in os.listdir(folder_path):
            # 构造完整的文件路径
            file_path = os.path.join(folder_path, filename)

            # 检查是否为文件
            if os.path.isfile(file_path):
                # 分离文件名和扩展名
                name_part, ext_part = os.path.splitext(filename)

                # 确保索引不会超出文件名长度
                if index > len(name_part) or index < 0:
                    print(f"索引 {index} 对于文件 '{filename}' 超出范围，跳过此文件。")
                    continue

                # 在指定索引位置插入字符
                new_name_part = name_part[:index] + ch + name_part[index:]

                # 新的完整文件名
                new_filename = new_name_part + ext_part

                # 新旧文件的完整路径
                new_file_path = os.path.join(folder_path, new_filename)

                # 重命名文件
                os.rename(file_path, new_file_path)
                print(f"文件已重命名为: {new_filename}")
    except Exception as e:
        print(f"处理过程中发生错误: {e}")


# 示例用法

insert_char_in_filenames(r"D:\rs\data\nnUNet_raw\Dataset002_Heart\imagesTs", 3, "0")