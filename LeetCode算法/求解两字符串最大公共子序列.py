#! /usr/bin/env python3
# -*- coding:utf-8 -*-

# Author   : mayi
# Blog     : http://www.cnblogs.com/mayi0312/
# Date     : 2019/5/16
# Name     : test03
# Software : PyCharm
# Note     : 用于实现求解两个字符串的最长公共子序列


def longestCommonSequence(str_one, str_two, case_sensitive=True):
    """
    str_one 和 str_two 的最长公共子序列
    :param str_one: 字符串1
    :param str_two: 字符串2（正确结果）
    :param case_sensitive: 比较时是否区分大小写，默认区分大小写
    :return: 最长公共子序列的长度
    """
    len_str1 = len(str_one)
    len_str2 = len(str_two)
    # 定义一个列表来保存最长公共子序列的长度，并初始化
    record = [[0 for i in range(len_str2 + 1)] for j in range(len_str1 + 1)]
    for i in range(len_str1):
        for j in range(len_str2):
            if str_one[i] == str_two[j]:
                record[i + 1][j + 1] = record[i][j] + 1
            elif record[i + 1][j] > record[i][j + 1]:
                record[i + 1][j + 1] = record[i + 1][j]
            else:
                record[i + 1][j + 1] = record[i][j + 1]

    return record[-1][-1]

if __name__ == '__main__':
    # 字符串1
    s1 = "BDCABA"
    # 字符串2
    s2 = "ABCBDAB"
    # 计算最长公共子序列的长度
    res = longestCommonSequence(s1, s2)
    # 打印结果
    print(res) # 4