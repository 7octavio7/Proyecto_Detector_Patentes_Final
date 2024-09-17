# Importamos las librerias
from ultralytics import YOLO
import cv2

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    model = YOLO('data.yaml')
    resultados = model.train(data="data.yaml", epochs=3)

