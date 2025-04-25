import os
import json
from tqdm import tqdm


def merge_train_json(root_dir, output_file):
    merged_data = []
    error_log = []

    # 递归查找所有train.json（网页8）
    json_files = []
    for root, _, files in os.walk(root_dir):
        if "train.json" in files:
            json_files.append(os.path.join(root, "train.json"))

    # 处理文件（综合网页3、6、8）
    for file_path in tqdm(json_files, desc="处理进度"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

                # 结构校验（网页7）
                if not isinstance(data, list):
                    raise ValueError("文件结构应为数组")

                for item in data:
                    # 字段校验（网页5）
                    required_fields = ["title", "content"]
                    if not all(field in item for field in required_fields):
                        error_log.append(f"{file_path} 缺少必要字段")
                        continue

                    # 类型校验（网页6）
                    if not isinstance(item.get("keywords", {}), dict):
                        error_log.append(f"{file_path} 中 {item['title']} 的keywords类型错误")
                        continue

                    merged_data.append(item)

        except Exception as e:
            error_log.append(f"{file_path} 错误: {str(e)}")

    # 保存结果（网页1）
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)

    # 错误报告（网页7）
    if error_log:
        print("\n处理过程中发现以下错误：")
        for err in error_log[-10:]:  # 显示最后10条错误
            print(f"• {err}")

    print(f"\n成功合并 {len(merged_data)} 条数据到 {output_file}")


if __name__ == "__main__":
    merge_train_json(r"D:\Develop\tianchi\train-data", r"D:\Develop\tianchi\merge.json")
