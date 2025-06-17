import argparse
import os
from ultralytics import YOLO

def main(args):
    # Создаём директорию для чекпоинтов, если не существует
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Загружаем модель (можно выбрать yolov8n, yolov8s, yolov8m и т.д.)
    model = YOLO('yolo11n.pt')
    # Запускаем обучение через встроенный API Ultralytics
    model.train(
        data=args.data_yaml,             # путь к data.yaml (COCO-формат)
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.img_size,
        lr0=args.lr,
        device=args.device,
        project=args.checkpoint_dir,     # директория для чекпоинтов
        name='detector',                 # имя эксперимента (папка внутри project)
        workers=4,                       # число потоков для загрузки данных
        exist_ok=True,                   # не ругаться если папка уже есть
        # augmentations и другие параметры можно также тут задать
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_yaml', type=str, default='InsPLAD-det/data.yaml', help='Путь к data.yaml')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Директория для чекпоинтов')
    parser.add_argument('--batch_size', type=int, default=4, help='Размер батча')
    parser.add_argument('--epochs', type=int, default=50, help='Число эпох')
    parser.add_argument('--lr', type=float, default=1e-4, help='Начальный learning rate')
    parser.add_argument('--img_size', type=int, default=640, help='Размер картинки (imgsz)')
    parser.add_argument('--device', type=str, default='cuda', help='cuda или cpu')
    args = parser.parse_args()
    main(args)
