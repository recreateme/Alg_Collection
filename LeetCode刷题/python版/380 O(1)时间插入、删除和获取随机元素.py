import random



class RandomizedSet:
    def __init__(self):
        self.nums = []              # 元素列表
        self.val_to_index = {}      # 元素值到索引的映射

    def insert(self, val: int) -> bool:
        if val in self.val_to_index:
            return False        # 元素已存在，不进行操作

        # 元素不存在，插入元素并更新映射
        self.val_to_index[val] = len(self.nums)
        # 插入元素到值列表
        self.nums.append(val)
        return True

    def remove(self, val: int) -> bool:
        if val not in self.val_to_index:
            return False
        # 交换待删除元素与末尾元素
        last_val = self.nums[-1]
        idx = self.val_to_index[val]
        self.nums[idx] = last_val
        self.val_to_index[last_val] = idx
        
        # 删除末尾元素
        self.nums.pop()
        del self.val_to_index[val]
        return True

    def getRandom(self) -> int:
        return random.choice(self.nums)