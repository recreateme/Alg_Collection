import json
from pathlib import Path


def recursive_replace(data, replacements):
    """递归遍历JSON数据结构并替换占位符[8,6](@ref)"""
    if isinstance(data, dict):
        for key, value in data.items():
            data[key] = recursive_replace(value, replacements)
    elif isinstance(data, list):
        for i in range(len(data)):
            data[i] = recursive_replace(data[i], replacements)
    elif isinstance(data, str):
        for placeholder, new_value in replacements.items():
            data = data.replace(placeholder, new_value)
    return data


def main():
    # 配置参数
    replacements = {
        "{{name}}": "AI小助手",
        "{{author}}": "开发者团队"
    }
    json_path = Path("identity.json")
    backup_path = json_path.with_suffix(".bak")

    try:
        # 创建备份文件[7](@ref)
        json_path.rename(backup_path)

        # 读取并处理JSON[3,8](@ref)
        with open(backup_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        modified_data = recursive_replace(data, replacements)

        # 写入格式化后的JSON[4,6](@ref)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(modified_data, f,
                      indent=2,
                      ensure_ascii=False,
                      separators=(',', ': '))

        print(f"成功更新 {json_path}，原始文件已备份为 {backup_path}")

    except FileNotFoundError:
        print(f"错误：文件 {json_path} 不存在")
    except json.JSONDecodeError as e:
        print(f"JSON解析失败：{str(e)}")
        backup_path.rename(json_path)  # 恢复原始文件


if __name__ == "__main__":
    main()