import os
import logging
import chardet

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def detect_encoding(file_path):
    """
    Try to automatically detect the encoding of a file using chardet.

    :param file_path: Path to the file.
    :return: Detected encoding or None if detection fails.
    """
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            if result['confidence'] > 0.95:
                return result['encoding']
    except Exception as e:
        logging.error(f"Error detecting encoding for {file_path}: {e}")
    return None


def convert_py_files_to_utf8(directory):
    """
    Convert all .py files in the given directory and its subdirectories to UTF-8 encoding.

    :param directory: The directory path to search for .py files.
    """
    if not os.path.isdir(directory):
        logging.error(f"The provided path '{directory}' is not a valid directory.")
        return

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)

                # 尝试自动检测文件编码
                encoding = detect_encoding(file_path)
                if encoding is None:
                    logging.warning(f"Could not determine encoding for {file_path}. Skipping...")
                    continue

                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()

                    # 检查是否已经是 UTF-8
                    if isinstance(content, str):
                        try:
                            content.encode('utf-8').decode('utf-8')
                            logging.info(f"{file_path} is already in UTF-8")
                        except UnicodeEncodeError:
                            # 写入 UTF-8 编码的内容
                            with open(file_path, 'w', encoding='utf-8') as f:
                                f.write(content)
                            logging.info(f"Converted {file_path} to UTF-8")

                except UnicodeDecodeError as e:
                    logging.error(f"Error decoding {file_path}: {e}")
                except Exception as e:
                    logging.error(f"Unexpected error processing {file_path}: {e}")


# 使用示例
if __name__ == '__main__':
    convert_py_files_to_utf8(r"D:\Develop\qatrackplusCN")