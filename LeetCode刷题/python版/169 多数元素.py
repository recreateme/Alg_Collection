from typing import List

class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        # 设置第一个为候选值，计数为1
        candidate, count = nums[0], 1
        # 遍历数组
        for num in nums[1:]:
            # 如果当前count为0，则更新候选值为当前元素
            if count == 0:
                candidate = num
            # 如果当前元素与候选值不同，计数减1，否则计数加1
            count += (1 if num == candidate else -1)
        return candidate

    # def mm(self, nums: List[int]):
    #     candidate = nums[0], count = 1
    #     
    #     for num in nums[1:]:
    #         if count == 0:
    #             candidate = num
    #         count += (1 if num == candidate else -1)
    #     return candidate