"""
编写一个函数来查找字符串数组中的最长公共前缀。

如果不存在公共前缀，返回空字符串 ""
"""


def longestCommonPrefix(strs: list[str]) -> str:
    if not strs:
        return ""

    min_len = min(len(s) for s in strs)
    for i in range(min_len):
        char = strs[0][i]
        if any(s[i] != char for s in strs[1:]):
            return strs[0][:i]

    return strs[0][:min_len]