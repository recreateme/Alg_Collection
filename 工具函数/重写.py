import os
import codecs
import chardet


def is_text_file(file_path):
    """
    检查文件是否为文本文件
    """
    try:
        with open(file_path, 'rb') as file:
            chunk = file.read(1024)
            return not bool(b'\0' in chunk)  # 二进制文件通常包含空字节
    except IOError:
        return False


def convert_to_utf8(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if is_text_file(file_path):
                try:
                    # 检测文件编码
                    with open(file_path, 'rb') as f:
                        raw_data = f.read()
                    detected = chardet.detect(raw_data)
                    encoding = detected['encoding']

                    if encoding and encoding.lower() != 'utf-8':
                        # 使用检测到的编码读取文件
                        with codecs.open(file_path, 'r', encoding) as f:
                            content = f.read()
                        # 以 UTF-8 编码写入文件
                        with codecs.open(file_path, 'w', 'utf-8') as f:
                            f.write(content)
                        print(f"已将 {file_path} 从 {encoding} 转换为 UTF-8")
                    else:
                        print(f"{file_path} 已经是 UTF-8 编码，无需转换")
                except Exception as e:
                    print(f"转换 {file_path} 时出错: {str(e)}")


# 使用方法：将你的项目路径作为参数传入
convert_to_utf8(r"D:\Develop\qatrackplusCN")
