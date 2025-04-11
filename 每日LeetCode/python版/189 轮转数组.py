'''
给定一个整数数组 nums，将数组中的元素向右轮转 k 个位置，其中 k 是非负数。

示例 1:

输入: nums = [1,2,3,4,5,6,7], k = 3
输出: [5,6,7,1,2,3,4]
解释:
向右轮转 1 步: [7,1,2,3,4,5,6]
向右轮转 2 步: [6,7,1,2,3,4,5]
向右轮转 3 步: [5,6,7,1,2,3,4]
'''

"""
最优解：三次翻转
1. 翻转整体数组
2. 翻转前 k 个元素
3. 翻转后 len(nums) - k 个元素
"""

from typing import List


class Solution(object):
    def rotate(self, nums: List[int], k: int) -> None:
        n = len(nums)
        k %= n  # 处理 k > n 的情况
        self.reverse(nums, 0, n - 1)
        self.reverse(nums, 0, k - 1)
        self.reverse(nums, k, n - 1)

    def reverse(self, nums: List[int], start: int, end: int) -> None:
        while start < end:
            nums[start], nums[end] = nums[end], nums[start]
            start += 1
            end -= 1