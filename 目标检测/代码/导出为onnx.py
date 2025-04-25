from ultralytics import YOLO

# 加载预训练模型
model = YOLO("yolo11n.pt")

# 导出为ONNX格式（关键参数配置）
model.export(
    format="onnx",        # 指定导出格式为ONNX[1,6](@ref)
    imgsz=640,            # 输入图像尺寸（与训练时一致）[4](@ref)
    dynamic=True,         # 启用动态批量维度[3,5](@ref)
    simplify=True,        # 简化模型图结构[7](@ref)
    opset=12              # 指定ONNX算子集版本[6](@ref)
)