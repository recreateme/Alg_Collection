import os
import codecs


def convert_to_utf8(directory, type=[".py"]):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.split(".")[-1] in type:
                file_path = os.path.join(root, file)
                try:
                    with codecs.open(file_path, 'r', 'gbk') as f:
                        content = f.read()
                    with codecs.open(file_path, 'w', 'utf-8') as f:
                        f.write(content)
                    print(f"Converted {file_path} to UTF-8")
                except UnicodeDecodeError:
                    print(f"Failed to convert {file_path}")


# 传入目录和需要转码的文件类型
convert_to_utf8(r"D:\backupQA\qatrackplusCN")
