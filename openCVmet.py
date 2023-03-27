import cv2
import numpy as np
import os

pathFoto = 'data'

import face_recognition  # https://github.com/ageitgey/face_recognition#face-recognition


def findEncodings(images):
    ''' кодирование лиц методом face_recognition кродирование списка лиц для распознания
        вход: набор images
        возврящает: list (эмбеддинг лиц?) лиц'''
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def image_resize(image, scale_percent):
    '''  Изменение размера в процентах'''
    widht = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)

    return cv2.resize(image, (widht, height), interpolation=cv2.INTER_AREA)


def open_img(path_img):
    ''' Открывает картинку и преобразует  в COLOR_BGR2RGB'''
    image = cv2.imread(path_img)
    # image = cv2.imread(os.path.join(path, file), -1)
    # -1: Загружает картинку в том виде, в котором она есть, включая альфу.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # приведение к цвету
    # cv2.COLOR_RGB2GRAY) # сделать серым

    return image


# def save_file(res, file_uri):
#     # de-standartize
#     # Денормализуем картинку обратно, сохраняем со случайным именем
#     # и загружаем в наше хранилище   нк применял
#     res = (res * 0.5) + 0.5
#     tempfile = "/tmp/" + str(uuid.uuid4()) + ".jpg"
#     torchvision.utils.save_image(res, tempfile)
#     client.file(file_uri).putFile(tempfile)


def saveImage(file, path=pathFoto, finename='newfoto.jpg'):
    ''' Сохранение картинки по умолчанию относительный путь к папке источнику path=data/фото  + вложенная папка paper = 'newfoto'
    можно изменить на ваше усмотрение или передать новое при вызове'''

    # print('saveImage Сохранаяю :', os.path.join(path, finename), file.shape )

    cv2.imwrite(os.path.join(path, finename), file)


def viewImage(image, waiK=500, nameWindow='message windows', verbose=True):
    ''' Вывод в отдельное окошко
    image - картинка numpy подобный массив
    waiK - int время ожидания окна если 0- будет ждать нажатия клавиши
    nameWindow - название окна лучше по английски иначе проблемы с размером
    verbose - показывать или нет True/False
    '''
    if verbose:
        cv2.namedWindow(nameWindow, cv2.WINDOW_NORMAL)
        cv2.imshow(nameWindow, image)

        key = cv2.waitKey(waiK)
        if key == 27:
            print('Нажали клавишу 27')
        cv2.destroyAllWindows()
    else:
        pass
    return


def rotation(img, center, angel):
    ''' Вращение картинки методом openCV
    но много требует плясок и режет края numpy привычнее'''
    (h, w) = img.shape[:2]
    # center = (int(w / 2), int(h / 2))
    rotation_matrix = cv2.getRotationMatrix2D(center, -angel, 1)
    rotated = cv2.warpAffine(img, rotation_matrix, (w, h))
    return rotated


def findContur(image):
    ''' превращение картинки в контуры удобно убирает мусор '''
    # Convert to graycsale
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Blur the image for better edge detection
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)

    # Sobel Edge Detection
    sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)  # Sobel Edge Detection on the X axis
    sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)  # Sobel Edge Detection on the Y axis
    sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)  # Combined X and Y Sobel Edge Detection

    edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)  # Canny Edge Detection

    return edges


def DravRectangleImage(image, rectangle_NP):
    ''' рисует картинку и квадраты фич работает 13.05.22
        image: cv.imread
        rectangle_NP :<class 'numpy.ndarray'> (x, 4)
    return
        True -
        Fault - не найдены фичи
    '''
    faces_detected = "Fich find: " + format(len(rectangle_NP))
    if len(rectangle_NP) == 0:
        print('не найдены фичи')
        return 0

    image = np.ascontiguousarray(image, dtype=np.uint8)
    # Рисуем квадраты
    for (x, y, w, h) in rectangle_NP:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 20)  # отрисовка квадратов

    return image