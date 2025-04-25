"""
将一个给定字符串 s 根据给定的行数 numRows ，以从上往下、从左到右进行 Z 字形排列。

比如输入字符串为 "PAYPALISHIRING" 行数为 3 时，排列如下：

P   A   H   N
A P L S I I G
Y   I   R
"""

def convert(s: str, numRows: int) -> str:
    if numRows == 1 or numRows >= len(s):
        return s

    rows = [''] * numRows
    current_row = 0
    direction = 1  # 1表示向下，-1表示向上

    for char in s:
        rows[current_row] += char
        current_row += direction
        if current_row == 0 or current_row == numRows - 1:
            direction *= -1  # 到达边界时反转方向

    return ''.join(rows)