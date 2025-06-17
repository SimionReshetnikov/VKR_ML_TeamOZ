import argparse
import os
from ultralytics import YOLO

def main(args):
    # Создаём директорию для чекпоинтов, если не существует
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Загружаем модель для классификации (например, yolov8n-cls.pt)
    model = YOLO('yolo11n-cls.pt')  # или другой pretrain, если есть

    # Запускаем обучение через встроенный API Ultralytics
    model.train(
        data=args.data_dir,               # путь к директории с данными (ImageNet-style)
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.img_size,
        lr0=args.lr,
        device=args.device,
        project=args.checkpoint_dir,      # директория для чекпоинтов
        name='classifier',                # имя эксперимента (папка внутри project)
        workers=4,                        # число потоков для загрузки данных
        exist_ok=True,                    # не ругаться если папка уже есть
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='InsPLAD-fault/supervised_fault_classification/defect_supervised', help='Путь к директории с классами')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Директория для чекпоинтов')
    parser.add_argument('--batch_size', type=int, default=32, help='Размер батча')
    parser.add_argument('--epochs', type=int, default=70, help='Число эпох')
    parser.add_argument('--lr', type=float, default=1e-3, help='Начальный learning rate')
    parser.add_argument('--img_size', type=int, default=224, help='Размер картинки (imgsz)')
    parser.add_argument('--device', type=str, default='cuda', help='cuda или cpu')
    args = parser.parse_args()
    main(args)