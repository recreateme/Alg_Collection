"""
给你一个字符串 s，由若干单词组成，单词前后用一些空格字符隔开。返回字符串中 最后一个 单词的长度。
单词 是指仅由字母组成、不包含任何空格字符的最大子字符串。

示例 1：

输入：s = "Hello World"
输出：5
解释：最后一个单词是“World”，长度为 5。
"""

def lengthOfLastWord(s: str) -> int:
    i = len(s) - 1
    # 跳过末尾空格
    while i >= 0 and s[i] == ' ':
        i -= 1
    # 统计最后一个单词长度
    length = 0
    while i >= 0 and s[i] != ' ':
        length += 1
        i -= 1
    return length