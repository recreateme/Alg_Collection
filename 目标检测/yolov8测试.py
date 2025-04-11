from ultralytics import YOLO
import cv2

model = YOLO("yolo11n.pt")  # 加载预训练模型
cap = cv2.VideoCapture(0)    # 打开摄像头

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 方法一：禁用stream模式（直接获取结果）
    results = model.predict(frame, conf=0.3, imgsz=640)
    annotated_frame = results[0].plot()  # 绘制检测框

    # 方法二：使用stream模式时遍历生成器
    # for result in model.predict(frame, stream=True, conf=0.5, imgsz=640):
    #     annotated_frame = result.plot()

    cv2.imshow('YOLO Real-Time Detection', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()