"""
给你一个非负整数数组 nums ，你最初位于数组的 第一个下标 。数组中的每个元素代表你在该位置可以跳跃的最大长度。

判断你是否能够到达最后一个下标，如果可以，返回 true ；否则，返回 false
"""


class Solution(object):
    def canJump(self, nums):
        n = len(nums)   # 数组长度
        max_reach = 0  # 当前能到达的最远位置

        for i in range(n):   # 遍历数组
            if i > max_reach:  # 当前位置无法到达
                return False
            max_reach = max(max_reach, i + nums[i])
            if max_reach >= n - 1:  # 可以到达终点
                return True
        return False