"""
给定一个字符串 s ，请你找出其中不含有重复字符的 最长 子串 的长度
"""

class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        char_index = {}     # 存储字符最后出现的位置
        max_len = 0
        left = 0            # 窗口左边界
        
        for right, c in enumerate(s):
            # 如果字符c已存在且位置≥left，则更新左边界
            if c in char_index and char_index[c] >= left:
                left = char_index[c] + 1
            
            # 更新字符c的位置，并计算当前窗口长度
            char_index[c] = right
            max_len = max(max_len, right - left + 1)
        
        return max_len