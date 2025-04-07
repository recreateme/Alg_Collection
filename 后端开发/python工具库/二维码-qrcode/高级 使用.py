import qrcode

qr = qrcode.QRCode(
    version=1,  # 二维码版本
    error_correction=qrcode.constants.ERROR_CORRECT_L,  # 二维码纠错级别
    box_size=10,    # 二维码每个格子的像素大小
    border=4,        # 二维码边框的宽度
)

qr.add_data('定制信息')  # 二维码包含的信息
qr.make(fit=True)      # 自动调整二维码大小以适应图片大小

img = qr.make_image(fill_color="black", back_color="white")  # 生成二维码图片
img.save('qrcode.png')  # 保存二维码图片