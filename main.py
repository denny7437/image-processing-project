import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import corner_harris, corner_peaks
from PIL import Image
import os

def load_and_display_image(image_path):
    """Загрузка и отображение изображения"""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError("Не удалось загрузить изображение. Проверьте путь к файлу.")
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Создаем директорию для результатов задания 1
    os.makedirs("image_processing_project/results/task1", exist_ok=True)
    
    # Отображение и сохранение
    plt.imshow(img_rgb)
    plt.title("Исходное изображение")
    plt.axis("off")
    plt.savefig("image_processing_project/results/task1/original_image.png", bbox_inches='tight', pad_inches=0)
    plt.show()
    
    return img_bgr, img_rgb

def color_space_conversion(image):
    """Преобразование цветовых пространств"""
    # Создаем директорию для результатов задания 2
    os.makedirs("image_processing_project/results/task2", exist_ok=True)
    
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Отображение всех цветовых пространств
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # RGB
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    axes[0].imshow(img_rgb)
    axes[0].set_title("RGB")
    axes[0].axis("off")
    
    # Grayscale
    axes[1].imshow(img_gray, cmap="gray")
    axes[1].set_title("Grayscale")
    axes[1].axis("off")
    
    # HSV
    img_hsv_to_rgb = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
    axes[2].imshow(img_hsv_to_rgb)
    axes[2].set_title("HSV (конвертировано в RGB)")
    axes[2].axis("off")
    
    plt.savefig("image_processing_project/results/task2/color_spaces.png", bbox_inches='tight', pad_inches=0)
    plt.show()
    
    # Построение гистограмм
    plt.figure(figsize=(10, 4))
    
    # Гистограмма для Grayscale
    plt.subplot(1, 2, 1)
    plt.hist(img_gray.ravel(), bins=256, range=(0, 256))
    plt.title("Гистограмма яркости (Grayscale)")
    
    # Гистограмма для V-канала
    v_channel = img_hsv[:, :, 2]
    plt.subplot(1, 2, 2)
    plt.hist(v_channel.ravel(), bins=256, range=(0, 256))
    plt.title("Гистограмма яркости (V-канал HSV)")
    
    plt.savefig("image_processing_project/results/task2/histograms.png", bbox_inches='tight', pad_inches=0)
    plt.show()
    
    # Сохранение отдельных изображений
    cv2.imwrite("image_processing_project/results/task2/grayscale.png", img_gray)
    cv2.imwrite("image_processing_project/results/task2/hsv.png", cv2.cvtColor(img_hsv_to_rgb, cv2.COLOR_RGB2BGR))
    
    return img_gray, img_hsv

def apply_filters(gray_image):
    """Применение различных фильтров"""
    # Создаем директорию для результатов задания 3
    os.makedirs("image_processing_project/results/task3", exist_ok=True)
    
    # Гауссово сглаживание с разными значениями сигма
    gaussian_sigma1 = cv2.GaussianBlur(gray_image, (5, 5), 1.0)
    gaussian_sigma2 = cv2.GaussianBlur(gray_image, (5, 5), 2.0)
    gaussian_sigma3 = cv2.GaussianBlur(gray_image, (5, 5), 4.0)
    
    # Медианный фильтр с разными размерами ядра
    median_3 = cv2.medianBlur(gray_image, 3)
    median_5 = cv2.medianBlur(gray_image, 5)
    median_7 = cv2.medianBlur(gray_image, 7)
    
    # Фильтр Лапласа
    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))
    
    # Отображение результатов
    # Гауссово сглаживание
    plt.figure(figsize=(15, 5))
    plt.subplot(141)
    plt.imshow(gray_image, cmap='gray')
    plt.title('Исходное изображение')
    plt.axis('off')
    
    plt.subplot(142)
    plt.imshow(gaussian_sigma1, cmap='gray')
    plt.title('Гаусс (σ=1.0)')
    plt.axis('off')
    
    plt.subplot(143)
    plt.imshow(gaussian_sigma2, cmap='gray')
    plt.title('Гаусс (σ=2.0)')
    plt.axis('off')
    
    plt.subplot(144)
    plt.imshow(gaussian_sigma3, cmap='gray')
    plt.title('Гаусс (σ=4.0)')
    plt.axis('off')
    
    plt.savefig('image_processing_project/results/task3/gaussian_filters.png', bbox_inches='tight', pad_inches=0)
    plt.show()
    
    # Медианная фильтрация
    plt.figure(figsize=(15, 5))
    plt.subplot(141)
    plt.imshow(gray_image, cmap='gray')
    plt.title('Исходное изображение')
    plt.axis('off')
    
    plt.subplot(142)
    plt.imshow(median_3, cmap='gray')
    plt.title('Медианный (3x3)')
    plt.axis('off')
    
    plt.subplot(143)
    plt.imshow(median_5, cmap='gray')
    plt.title('Медианный (5x5)')
    plt.axis('off')
    
    plt.subplot(144)
    plt.imshow(median_7, cmap='gray')
    plt.title('Медианный (7x7)')
    plt.axis('off')
    
    plt.savefig('image_processing_project/results/task3/median_filters.png', bbox_inches='tight', pad_inches=0)
    plt.show()
    
    # Фильтр Лапласа
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.imshow(gray_image, cmap='gray')
    plt.title('Исходное изображение')
    plt.axis('off')
    
    plt.subplot(122)
    plt.imshow(laplacian, cmap='gray')
    plt.title('Фильтр Лапласа')
    plt.axis('off')
    
    plt.savefig('image_processing_project/results/task3/laplacian_filter.png', bbox_inches='tight', pad_inches=0)
    plt.show()
    
    # Сохранение отдельных результатов
    cv2.imwrite('image_processing_project/results/task3/gaussian_sigma1.png', gaussian_sigma1)
    cv2.imwrite('image_processing_project/results/task3/gaussian_sigma2.png', gaussian_sigma2)
    cv2.imwrite('image_processing_project/results/task3/gaussian_sigma3.png', gaussian_sigma3)
    cv2.imwrite('image_processing_project/results/task3/median_3x3.png', median_3)
    cv2.imwrite('image_processing_project/results/task3/median_5x5.png', median_5)
    cv2.imwrite('image_processing_project/results/task3/median_7x7.png', median_7)
    cv2.imwrite('image_processing_project/results/task3/laplacian.png', laplacian)
    
    return {
        'gaussian': [gaussian_sigma1, gaussian_sigma2, gaussian_sigma3],
        'median': [median_3, median_5, median_7],
        'laplacian': laplacian
    }

def edge_corner_detection(image, gray_image):
    """Определение краев и углов"""
    # Создаем директорию для результатов задания 4
    os.makedirs("image_processing_project/results/task4", exist_ok=True)
    
    # Оператор Собеля
    sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    sobelx = np.uint8(np.absolute(sobelx))
    sobely = np.uint8(np.absolute(sobely))
    sobel_combined = cv2.bitwise_or(sobelx, sobely)
    
    # Детектор Кэнни
    edges_canny = cv2.Canny(gray_image, 100, 200)
    
    # Детектор углов Харриса
    corners = cv2.cornerHarris(gray_image, blockSize=2, ksize=3, k=0.04)
    corners = cv2.dilate(corners, None)
    
    # Наложение углов на изображение
    img_corners = image.copy()
    img_corners[corners > 0.01 * corners.max()] = [0, 0, 255]  # Красные точки для углов
    
    # Отображение результатов Собеля
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.imshow(sobelx, cmap='gray')
    plt.title('Собель X')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(sobely, cmap='gray')
    plt.title('Собель Y')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(sobel_combined, cmap='gray')
    plt.title('Собель комбинированный')
    plt.axis('off')
    
    plt.savefig('image_processing_project/results/task4/sobel_edges.png', bbox_inches='tight', pad_inches=0)
    plt.show()
    
    # Отображение результатов Кэнни
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.imshow(gray_image, cmap='gray')
    plt.title('Исходное изображение')
    plt.axis('off')
    
    plt.subplot(122)
    plt.imshow(edges_canny, cmap='gray')
    plt.title('Детектор Кэнни')
    plt.axis('off')
    
    plt.savefig('image_processing_project/results/task4/canny_edges.png', bbox_inches='tight', pad_inches=0)
    plt.show()
    
    # Отображение углов Харриса
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Исходное изображение')
    plt.axis('off')
    
    plt.subplot(122)
    plt.imshow(cv2.cvtColor(img_corners, cv2.COLOR_BGR2RGB))
    plt.title('Углы Харриса')
    plt.axis('off')
    
    plt.savefig('image_processing_project/results/task4/harris_corners.png', bbox_inches='tight', pad_inches=0)
    plt.show()
    
    # Сохранение отдельных результатов
    cv2.imwrite('image_processing_project/results/task4/sobel_x.png', sobelx)
    cv2.imwrite('image_processing_project/results/task4/sobel_y.png', sobely)
    cv2.imwrite('image_processing_project/results/task4/sobel_combined.png', sobel_combined)
    cv2.imwrite('image_processing_project/results/task4/canny.png', edges_canny)
    cv2.imwrite('image_processing_project/results/task4/harris_corners.png', img_corners)
    
    return {
        'sobel': {'x': sobelx, 'y': sobely, 'combined': sobel_combined},
        'canny': edges_canny,
        'harris': img_corners
    }

def morphological_operations(gray_image):
    """Морфологические операции"""
    # Создаем директорию для результатов задания 5
    os.makedirs("image_processing_project/results/task5", exist_ok=True)
    
    # Бинаризация изображения
    _, binary = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    
    # Создание структурных элементов разной формы и размера
    kernel_square = np.ones((5,5), np.uint8)
    kernel_cross = np.array([[0,1,0],
                            [1,1,1],
                            [0,1,0]], np.uint8)
    kernel_circle = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    
    # Применение эрозии с разными ядрами
    erosion_square = cv2.erode(binary, kernel_square, iterations=1)
    erosion_cross = cv2.erode(binary, kernel_cross, iterations=1)
    erosion_circle = cv2.erode(binary, kernel_circle, iterations=1)
    
    # Применение дилатации с разными ядрами
    dilation_square = cv2.dilate(binary, kernel_square, iterations=1)
    dilation_cross = cv2.dilate(binary, kernel_cross, iterations=1)
    dilation_circle = cv2.dilate(binary, kernel_circle, iterations=1)
    
    # Отображение результатов бинаризации
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.imshow(gray_image, cmap='gray')
    plt.title('Исходное изображение')
    plt.axis('off')
    
    plt.subplot(122)
    plt.imshow(binary, cmap='gray')
    plt.title('Бинаризованное изображение')
    plt.axis('off')
    
    plt.savefig('image_processing_project/results/task5/binary.png', bbox_inches='tight', pad_inches=0)
    plt.show()
    
    # Отображение результатов эрозии
    plt.figure(figsize=(15, 5))
    plt.subplot(141)
    plt.imshow(binary, cmap='gray')
    plt.title('Бинаризованное')
    plt.axis('off')
    
    plt.subplot(142)
    plt.imshow(erosion_square, cmap='gray')
    plt.title('Эрозия (квадрат)')
    plt.axis('off')
    
    plt.subplot(143)
    plt.imshow(erosion_cross, cmap='gray')
    plt.title('Эрозия (крест)')
    plt.axis('off')
    
    plt.subplot(144)
    plt.imshow(erosion_circle, cmap='gray')
    plt.title('Эрозия (круг)')
    plt.axis('off')
    
    plt.savefig('image_processing_project/results/task5/erosion.png', bbox_inches='tight', pad_inches=0)
    plt.show()
    
    # Отображение результатов дилатации
    plt.figure(figsize=(15, 5))
    plt.subplot(141)
    plt.imshow(binary, cmap='gray')
    plt.title('Бинаризованное')
    plt.axis('off')
    
    plt.subplot(142)
    plt.imshow(dilation_square, cmap='gray')
    plt.title('Дилатация (квадрат)')
    plt.axis('off')
    
    plt.subplot(143)
    plt.imshow(dilation_cross, cmap='gray')
    plt.title('Дилатация (крест)')
    plt.axis('off')
    
    plt.subplot(144)
    plt.imshow(dilation_circle, cmap='gray')
    plt.title('Дилатация (круг)')
    plt.axis('off')
    
    plt.savefig('image_processing_project/results/task5/dilation.png', bbox_inches='tight', pad_inches=0)
    plt.show()
    
    # Сохранение отдельных результатов
    cv2.imwrite('image_processing_project/results/task5/binary.png', binary)
    cv2.imwrite('image_processing_project/results/task5/erosion_square.png', erosion_square)
    cv2.imwrite('image_processing_project/results/task5/erosion_cross.png', erosion_cross)
    cv2.imwrite('image_processing_project/results/task5/erosion_circle.png', erosion_circle)
    cv2.imwrite('image_processing_project/results/task5/dilation_square.png', dilation_square)
    cv2.imwrite('image_processing_project/results/task5/dilation_cross.png', dilation_cross)
    cv2.imwrite('image_processing_project/results/task5/dilation_circle.png', dilation_circle)
    
    return {
        'binary': binary,
        'erosion': {
            'square': erosion_square,
            'cross': erosion_cross,
            'circle': erosion_circle
        },
        'dilation': {
            'square': dilation_square,
            'cross': dilation_cross,
            'circle': dilation_circle
        }
    }

def plot_histogram(image, title):
    """Построение гистограммы"""
    pass

def main():
    # Импорт os в начале файла
    import os
    
    # Создание основной директории для результатов
    os.makedirs("image_processing_project/results", exist_ok=True)
    
    # Путь к изображению
    image_path = "image_processing_project/images/14-1400x933.png"
    
    # Загрузка и отображение изображения (Задание 1)
    img_bgr, img_rgb = load_and_display_image(image_path)
    
    # Преобразование цветовых пространств и построение гистограмм (Задание 2)
    img_gray, img_hsv = color_space_conversion(img_bgr)

    # Применение фильтров (Задание 3)
    filtered_images = apply_filters(img_gray)

    # Определение краев и углов (Задание 4)
    edge_corner_results = edge_corner_detection(img_bgr, img_gray)

    # Морфологические операции (Задание 5)
    morphology_results = morphological_operations(img_gray)

if __name__ == "__main__":
    main() 