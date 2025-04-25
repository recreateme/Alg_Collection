"""
给你一个整数数组 nums，返回 数组 answer ，其中 answer[i] 等于 nums 中除 nums[i] 之外其余各元素的乘积 。
题目数据 保证 数组 nums之中任意元素的全部前缀元素和后缀的乘积都在  32 位 整数范围内。
请 不要使用除法，且在 O(n) 时间复杂度内完成此题
"""

# 初始版本，空间复杂度 O(n)
class Solution:
    def productExceptSelf(self, nums):
        n = len(nums)    # 数组长度
        # 分别保存前缀乘积、后缀乘积、结果数组
        left, right, answer = [1] * n, [1] * n, [1] * n

        # 计算前缀乘积
        for i in range(1, n):
            left[i] = left[i - 1] * nums[i - 1]

        # 计算后缀乘积
        for i in range(n - 2, -1, -1):
            right[i] = right[i + 1] * nums[i + 1]

        # 合并结果
        for i in range(n):
            answer[i] = left[i] * right[i]

        return answer

# 优化版本，空间复杂度 O(1)
class Solution:
    def productExceptSelf(self, nums):
        n = len(nums)           # 数组长度
        answer = [1] * n        # 结果数组

        # 计算前缀乘积并存入 answer
        for i in range(1, n):
            answer[i] = answer[i - 1] * nums[i - 1]

        # 动态计算后缀乘积并直接更新 answer
        right = 1
        for i in range(n - 1, -1, -1):
            answer[i] *= right
            right *= nums[i]

        return answer