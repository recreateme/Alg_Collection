# 示例参考网页6[6](@ref)的API调用逻辑
import base64

from fastapi import requests


def face_swap(source_img, target_img):
    payload = {
        "source_image": base64.b64encode(source_img).decode(),
        "target_image": base64.b64encode(target_img).decode(),
        "model": "inswapper_128.onnx"
    }

    response = requests.post(SD_API_ENDPOINT, json=payload)
    return response.json()["output"]