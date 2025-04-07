'''
给你一个数组 nums 和一个值 val，你需要 原地 移除所有数值等于 val 的元素。元素的顺序可能发生改变。然后返回 nums 中与 val 不同的元素的数量。

假设 nums 中不等于 val 的元素数量为 k，要通过此题，您需要执行以下操作：

更改 nums 数组，使 nums 的前 k 个元素包含不等于 val 的元素。nums 的其余元素和 nums 的大小并不重要。
返回 k。
'''

def removeElement(nums, val):
    # 设置初始慢指针位置
    slow = 0
    
    # 遍历数组
    for fast in range(len(nums)):
        # 如果当前元素不等于 val，则将其赋值给慢指针指向的位置，并将慢指针位置加 1
        if nums[fast] != val:
            nums[slow] = nums[fast]
            slow += 1
        # 如果当前元素等于 val，则不做任何操作

    # 返回慢指针位置，即数组中不等于 val 的元素数量
    return slow