import torch
import torchvision
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pygame
import os

# Инициализация pygame для воспроизведения аудио
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound('alarm.wav')

# Загрузка предобученной модели (например, модель определения позы)
# Пример загрузки стандартной модели из torchvision
model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Использование камеры ноутбука (здесь предполагается, что камера имеет индекс 0)
cap = cv2.VideoCapture(0)

while True:
    # Захват изображения с камеры
    ret, frame = cap.read()
    if not ret:
        break

    # Преобразование изображения для модели
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    input_image = transform(image_rgb).unsqueeze(0)

    # Предсказания модели
    with torch.no_grad():
        prediction = model(input_image)

    # Обработка и отображение результатов
    # Предполагаем, что prediction - это результат модели определения позы
    # Получение координат ключевых точек
    if len(prediction) > 0 and 'keypoints' in prediction[0]:
        keypoints = prediction[0]['keypoints'][0].detach().numpy()
        # Извлечь координаты нужных ключевых точек (в зависимости от вашей модели)
        head = keypoints[0, 0:2]  # Индекс головы
        neck = keypoints[1, 0:2]  # Индекс шеи
        shoulder_left = keypoints[5, 0:2]  # Индекс левого плеча
        shoulder_right = keypoints[2, 0:2]  # Индекс правого плеча

        def calculate_angle(a, b, c):
            """Вычисление угла между тремя точками"""
            ba = a - b
            bc = c - b
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            angle = np.arccos(cosine_angle)
            return np.degrees(angle)

        # Вычисление угла между головой, шеей и позвоночником
        spine_top = np.array([neck[0], neck[1] - 10])  # Примерная точка верха позвоночника
        angle = calculate_angle(head, neck, spine_top)
        some_threshold_angle = 140
        # Определение сутулости
        if angle > some_threshold_angle:  # Угол становится острым при сутулости
            print("Пользователь сутулится")
            print(angle, spine_top, neck, head)
            # Затемнение экрана
            os.system("brightness 0")
            # Воспроизведение звука
            alarm_sound.play()
        else:
            print("Пользователь не сутулится")
            print(angle, spine_top, neck, head)
            os.system("brightness 1")
    # Показать изображение с наложенными ключевыми точками (необязательно)
    #cv2.imshow('Pose Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов камеры и закрытие окна
cap.release()
cv2.destroyAllWindows()
