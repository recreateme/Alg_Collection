class Solution(object):
    def intToRoman(self, num):
        """
        :type num: int
        :rtype: str
        """
        values = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
        symbols = ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"]

        result = []
        for i in range(len(values)):
            # 计算当前符号能出现的次数
            while num >= values[i]:
                num -= values[i]
                result.append(symbols[i])
        return ''.join(result)
