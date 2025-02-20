import hashlib
import binascii  # 用于转换二进制数据到十六进制表示
import os

x = hashlib.pbkdf2_hmac("sha256", b"I_love_python", b"", 1)
print("x_1 = " + binascii.hexlify(x).decode())

x = hashlib.pbkdf2_hmac("sha256", b"I_love_python", b"", 1)  # 相同盐值
print("x_2 = " + binascii.hexlify(x).decode())

x = hashlib.pbkdf2_hmac("sha256", b"I_love_python", b"", 10)  # 相同盐值，不同迭代次数
print("x_3 = " + binascii.hexlify(x).decode())

x = hashlib.pbkdf2_hmac("sha256", b"I_love_python", b"dsa", 1)  # 不同盐值，相同迭代次数
print("x_4 = " + binascii.hexlify(x).decode())

y = hashlib.pbkdf2_hmac("sha256", b"I_love_python", os.urandom(16), 1)  # 随机生成盐值
print("y_1 = " + binascii.hexlify(y).decode())
y = hashlib.pbkdf2_hmac("sha256", b"I_love_python", os.urandom(16), 1)  # 相同盐值
