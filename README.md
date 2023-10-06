# Классификатор изображений CIFAR-10

Для демонстрации работы число эпох обучения было уменьшено до минимума, из-за этого сеть не успевает обучиться и полученные предсказания получаются не самыми качественными

## Поверка:
Клонируем репозиторий
```bash
git clone git@github.com:i-alkisev/cifar10_classifier.git
```
```bash
cd cifar10_classifier/
```
Создаем чистое виртуальное окружение с питоном
```bash
conda create --name test-env python=3.11
```
```bash
conda activate test-env
```
Устанавливаем зависимости и запускаем хуки
```bash
poetry install
```
```bash
pre-commit install
```
```bash
pre-commit run -a
```
Запускаем обучение (занимает около 4 минут)
```bash
python cifar10_classifier/train.py
```
Смотрим метрики на трейне и тесте, а также сохраняем предсказание для теста в .csv файл
```bash
python cifar10_classifier/infer.py
```
