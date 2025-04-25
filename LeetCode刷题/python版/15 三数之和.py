"""
给你一个整数数组 nums ，判断是否存在三元组 [nums[i], nums[j], nums[k]] 满足 i != j、i != k 且 j != k ，
同时还满足 nums[i] + nums[j] + nums[k] == 0 。请你返回所有和为 0 且不重复的三元组。

注意：答案中不可以包含重复的三元组
"""


"""
1 排序：将数组排序，便于跳过重复元素和使用双指针。
2 固定一个数：遍历数组，固定 nums[i] 作为三元组的第一个数。
3 双指针搜索：用左右指针 left 和 right 在 nums[i] 右侧搜索满足 nums[left] + nums[right] == -nums[i] 的组合。
4 去重处理：跳过重复的 nums[i]、nums[left] 和 nums[right]
"""


def threeSum(nums: list[int]) -> list[list[int]]:
    nums.sort()
    res = []
    for i in range(len(nums) - 2):
        if nums[i] > 0:  # 第一个数大于0，后续无解
            break
        if i > 0 and nums[i] == nums[i - 1]:  # 跳过重复的nums[i]
            continue
        left, right = i + 1, len(nums) - 1
        while left < right:
            total = nums[i] + nums[left] + nums[right]
            if total < 0:
                left += 1
            elif total > 0:
                right -= 1
            else:
                res.append([nums[i], nums[left], nums[right]])
                while left < right and nums[left] == nums[left + 1]:  # 跳过重复的nums[left]
                    left += 1
                while left < right and nums[right] == nums[right - 1]:  # 跳过重复的nums[right]
                    right -= 1
                left += 1
                right -= 1
    return res