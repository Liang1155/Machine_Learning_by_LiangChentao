from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo11n.pt")
    results = model.train(
        data=r"D:\desk\programming\labelling\train.yaml"，  # 直接写本地绝对路径
        epochs=100,
        imgsz=640
)
