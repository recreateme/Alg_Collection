"""
给定一个含有 n 个正整数的数组和一个正整数 target 。

找出该数组中满足其总和大于等于 target 的长度最小的 子数组 [numsl, numsl+1, ..., numsr-1, numsr] ，
并返回其长度。如果不存在符合条件的子数组，返回 0
"""


def minSubArrayLen(target: int, nums: list[int]) -> int:
    left, sum_val, min_len = 0, 0, float('inf')
    for right in range(len(nums)):
        sum_val += nums[right]  # 扩展窗口
        while sum_val >= target:
            min_len = min(min_len, right - left + 1)  # 更新最小长度
            sum_val -= nums[left]  # 收缩窗口
            left += 1
    return min_len if min_len != float('inf') else 0