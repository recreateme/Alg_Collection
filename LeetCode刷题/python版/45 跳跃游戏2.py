"""
给定一个长度为 n 的 0 索引整数数组 nums。初始位置为 nums[0]。
每个元素 nums[i] 表示从索引 i 向后跳转的最大长度。换句话说，如果你在 nums[i] 处，你可以跳转到任意 nums[i + j] 处
"""

class Solution:
    def jump(self, nums):
        n = len(nums)    # 数组长度
        max_pos, end, steps = 0, 0, 0    # 当前跳跃位置，跳跃边界，跳跃次数

        # 遍历前 n-1 个元素（最后一个元素不需要遍历）
        for i in range(n - 1):
            max_pos = max(max_pos, i + nums[i])         # 计算当前跳跃位置的最大值
            if i == end:  # 到达当前跳跃边界
                steps += 1
                end = max_pos
        return steps
    