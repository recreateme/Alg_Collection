import re
import json
from datasets import load_dataset, DatasetDict


# —— 工具函数 —— #
def split_sentences(text: str):
    """
    按中文标点切分古诗为独立句子，保留标点。
    """
    text = text.strip()
    if text and text[-1] not in "。！？；":
        text += "。"
    parts = re.split(r'(?<=[。！？；])', text)
    return [p.strip() for p in parts if p.strip()]


# 训练集情感映射（示例，可按实际标签补充）
EMOTION2ID = {
    "爱国": 0,
    "庆祝": 1,
    "咏史": 2,
    "思乡": 3,
    "惆怅与无奈": 4,
    # ……
}


def encode_emotion_train(emotion: str) -> int:
    return EMOTION2ID.get(emotion, -1)


# —— 预处理函数 —— #
def preprocess_train(example):
    """
    训练集样本包含：title, content, keywords(dict), trans, emotion(str)
    输出：
      - input_text: prompt
      - label_text: JSON 字符串，含 ans_qa_words、ans_qa_sents（直接用 trans 拆解）和 choose_id
    """
    title = example["title"].strip()
    content = example["content"].strip()
    keywords = list(example["keywords"].keys())
    trans = example["trans"].strip()
    emotion = example["emotion"].strip()

    # 句子列表：从 content 拆分
    sentences = split_sentences(content)
    # 从 trans 拆分得到对应的白话句列表
    trans_sents = split_sentences(trans)

    # 构造 prompt
    prompt = (
        f"诗题：{title}；内容：{content}；\n"
        f"关键词：{keywords}；\n"
        f"请为每个关键词和以下每一句诗生成白话解释，并判断情感类别。\n"
        f"诗句列表：{sentences}。\n"
        "输出格式（JSON）：{ans_qa_words:…, ans_qa_sents:…, choose_id:…}"
    )
    # 构造 label
    ans_qa_words = {kw: example["keywords"][kw] for kw in keywords}
    ans_qa_sents = {s: t for s, t in zip(sentences, trans_sents)}
    choose_id = encode_emotion_train(emotion)

    label = {
        "ans_qa_words": ans_qa_words,
        "ans_qa_sents": ans_qa_sents,
        "choose_id": choose_id
    }
    return {
        "input_text": prompt,
        "label_text": json.dumps(label, ensure_ascii=False)
    }


def preprocess_val(example):
    """
    验证集样本包含：idx, title, author, content, qa_words(list), qa_sents(list), choose(dict)
    输出：
      - input_text: prompt（与训练一致的格式，供模型推理）
      - qa_words, qa_sents, choose: 原始字段保留，用于后续评估
    """
    title = example["title"].strip()
    content = example["content"].strip()
    qa_words = example["qa_words"]
    qa_sents = example["qa_sents"]
    choose = example["choose"]  # e.g. {"A":"欢快", "B":"无奈", …}

    # 构造 prompt，与训练时一致，只不过不附带答案
    prompt = (
        f"诗题：{title}；内容：{content}；\n"
        f"关键词：{qa_words}；\n"
        f"请为每个关键词和以下每一句诗生成白话解释，并判断情感类别。\n"
        f"诗句列表：{qa_sents}。\n"
        "输出格式（JSON）：{ans_qa_words:…, ans_qa_sents:…, choose_id:…}"
    )
    # 返回输入，以及保留的原始选项和 idx，便于后续解析和评测
    return {
        "idx": example["idx"],
        "input_text": prompt,
        "qa_words": qa_words,
        "qa_sents": qa_sents,
        "choose": choose
    }


# —— 主流程示例 —— #
if __name__ == "__main__":
    # 1. 加载原始训练/验证 JSON
    # 假设文件 train.json、val.json 分别为训练、验证集
    raw = load_dataset("json", data_files={"train": "./train_data.json", "validation": "./eval_data.json"})

    # 2. 分别 map 训练/验证预处理
    processed = DatasetDict({
        "train": raw["train"].map(preprocess_train, remove_columns=raw["train"].column_names),
        "validation": raw["validation"].map(preprocess_val, remove_columns=raw["validation"].column_names)
    })

    # 3. 保存到磁盘（可选）
    processed.save_to_disk("processed_poetry")
    # 或者导出为 JSON lines
    processed["train"].to_json("processed_poetry/train.jsonl")
    processed["validation"].to_json("processed_poetry/val.jsonl")

    print("预处理完成，输出目录：processed_poetry/")
