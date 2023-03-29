
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import time
import ast
import face_recognition
import copy
from PIL import Image, ImageDraw
from openCVmet import *
import uuid
import torchvision


class Face_recognition():
    def __init__(self):
        pass
    def video2face_recognition(self,video_path,scenes_markup):
        """

        :param video: переменная типа 'moviepy.editor.VideoFileClip'. Входное видео.
                                                  кадр начала сцены          кадр конца сцены
        :param scenes_markup: {'scenes_markup':[{'start_frame_scene': <int>, 'end_frame_scene': <int> },
                                                {'start_frame_scene': <int>, 'end_frame_scene': <int> }, ... ]}

        :return:

        classes: {'index_label': 'label_name', ...} соответствие лейблов
        распознанных классов и их названий.

        scenes_with_people: {'index_class': ['frame_1', 'frame_2', ...]} ,
        """
        scenes_with_people = self.my_face_recognition(video_path)
        print(video_path[:-4]+'_out'+video_path[-4:])
        classes = self.labels_classifier(scenes_with_people,video_path)

        return classes, scenes_with_people

    def my_face_recognition(self, pathVideo, csv_path = 'timingVideo_2.csv', rename=False, loadClasses=False, verbose = False):
        """
        Распознавание лиц.

        :param video: переменная типа 'moviepy.editor.VideoFileClip'. Входное видео.

        :return:

        scenes_with_people: {'people':[ {'people_in_scene': ['имя 1', 'имя 2', ...],
                                         'frames': ['frame 1','frame 2',...]       }, ...]}
        """

        # Логика функции
        # ===================================================================
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

        def save_file(res, file_uri):
            # de-standartize
            # Денормализуем картинку обратно, сохраняем со случайным именем
            # и загружаем в наше хранилище   нк применял
            res = (res * 0.5) + 0.5
            tempfile = "/tmp/" + str(uuid.uuid4()) + ".jpg"
            torchvision.utils.save_image(res, tempfile)
            client.file(file_uri).putFile(tempfile)

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
            sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0,
                               ksize=5)  # Sobel Edge Detection on the X axis
            sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1,
                               ksize=5)  # Sobel Edge Detection on the Y axis
            sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1,
                                ksize=5)  # Combined X and Y Sobel Edge Detection

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

        # %% Рабочие библиотеки распознавания
        def viewImageS(image, waiK=500, nameWindow='message windows', verbose=True):
            ''' Вывод в отдельное окошко
            image - картинка numpy подобный массив
            waiK - int время ожидания окна если 0- будет ждать нажатия клавиши
            nameWindow - название окна лучше по английски иначе проблемы с размером
            verbose - показывать или нет True/False
            '''
            if verbose:
                cv2.namedWindow(nameWindow, cv2.WINDOW_NORMAL)
                cv2.namedWindow('settings')  # Окно настроек
                cv2.imshow(nameWindow, image)

                key = cv2.waitKey(waiK)
                if key == 27:
                    print('Нажали клавишу 27')
                cv2.destroyAllWindows()
            else:
                pass
            return

        def vievLandmark(image, face_landmarks_list):
            ''' Рисование губ глаз носа '''
            # Load the jpg file into a numpy array
            # image = face_recognition.load_image_file("biden.jpg")

            # Find all facial features in all the faces in the image
            # face_landmarks_list = face_recognition.face_landmarks(image)

            pil_image = Image.fromarray(image)
            for face_landmarks in face_landmarks_list:
                # d = ImageDraw.Draw(pil_image, 'RGBA')
                d = ImageDraw.Draw(pil_image, mode='RGB')

                # Make the eyebrows into a nightmare
                d.polygon(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 128))
                d.polygon(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 128))
                d.line(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 150), width=5)
                d.line(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 150), width=5)

                # Gloss the lips
                d.polygon(face_landmarks['top_lip'], fill=(150, 0, 0, 128))
                d.polygon(face_landmarks['bottom_lip'], fill=(150, 0, 0, 128))
                d.line(face_landmarks['top_lip'], fill=(150, 0, 0, 64), width=8)
                d.line(face_landmarks['bottom_lip'], fill=(150, 0, 0, 64), width=8)

                # Sparkle the eyes
                d.polygon(face_landmarks['left_eye'], fill=(255, 255, 255, 30))
                d.polygon(face_landmarks['right_eye'], fill=(255, 255, 255, 30))

                # Apply some eyeliner
                d.line(face_landmarks['left_eye'] + [face_landmarks['left_eye'][0]], fill=(0, 0, 0, 110), width=6)
                d.line(face_landmarks['right_eye'] + [face_landmarks['right_eye'][0]], fill=(0, 0, 0, 110), width=6)

                pil_image.show()

            return

        def DravRectangleImage_face_rekogn(image, rectangle_NP, size_reduction_factor):
            ''' рисует картинку и квадраты фич работает 13.05.22
                image: cv.imread
                rectangle_NP :<class 'numpy.ndarray'> (x, 4)
            return
                True -
                Fault - не найдены фичи
            '''
            k = int(1 / size_reduction_factor)
            print('k:', k)
            # faces_detected = "Fich find: " + format(len(rectangle_NP))
            if len(rectangle_NP) == 0:
                # print('не найдены фичи')
                return 0

            # image = np.ascontiguousarray(image, dtype=np.uint8)
            # Рисуем квадраты
            for (y1, x2, y2, x1) in rectangle_NP:
                cv2.rectangle(image, (x1 * k, y1 * k), (x2 * k, y2 * k), (255, 255, 0), 5)  # отрисовка квадратов

            return image

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

        def Find_cascade_fich(face_cascade, image):
            ''' ищет лица на фото  методом openCV2 работает 13.05.22 cv.CascadeClassifier очень посредственно
            face_cascad
            image frame
            return:
                type(faces): <class 'numpy.ndarray'> (x, 4)
            время работы  wait time: 0.15 сек
             https://tproger.ru/translations/opencv-python-guide/'''

            assert not face_cascade.empty(), 'cv.CascadeClassifier( не нашёл файл haarcascade_frontalface_default.xml) '

            try:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # сделать серым
            except:
                gray = image

            faces = face_cascade.detectMultiScale(
                # общая функция для распознавания как лиц, так и объектов. Чтобы функция искала именно лица, мы передаём ей соответствующий каскад.
                gray,  # Обрабатываемое изображение в градации серого.
                scaleFactor=1.1,
                # Параметр scaleFactor. Некоторые лица могут быть больше других, поскольку находятся ближе, чем остальные. Этот параметр компенсирует перспективу.
                minNeighbors=5,
                # Алгоритм распознавания использует скользящее окно во время распознавания объектов. Параметр minNeighbors определяет количество объектов вокруг лица. То есть чем больше значение этого параметра, тем больше аналогичных объектов необходимо алгоритму, чтобы он определил текущий объект, как лицо. Слишком маленькое значение увеличит количество ложных срабатываний, а слишком большое сделает алгоритм более требовательным.
                minSize=(10, 10)  # непосредственно размер этих областей
            )

            return faces

        def findfith(image):
            ''' Подготовка класификатора OpenCV '''
            # Собственно этой командой мы загружаем уже обученные классификаторы cv.data.haarcascades+'haarcascade_frontalface_default.xml'
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

            faces = Find_cascade_fich(face_cascade, image)  # находит работает #поиск фитч "лиц"
            # print('Время распознания:', str(time.time()-StartTime), 'type:', type(faces), faces)

            return faces

        def recognitionFacesCV(frame):
            '''  Распознание локаций лиц методом face_recognition'''
            rectangle_NP = findfith(frame)
            if len(rectangle_NP) > 0:
                normRectangle = []
                for (x, y, w, h) in rectangle_NP:
                    normRectangle.append([y, x + w, y + h, x])
                return normRectangle
            return rectangle_NP

        def recognitionFaces(frame):
            '''  Распознание локаций лиц методом face_recognition'''
            return face_recognition.face_locations(frame)

        def findFacesOnVideo(video_path, encodeListKnown=[], output=True, classNames=['unknown'], faces_names=[]):
            '''поиск лиц на вадео и запись их п пандас фрейм
            video_path :str путь к файлу,
            encodeListKnown  список известных лиц
            output = True,  вывод в файл
            classNames = 'unknown'  - имена известных лиц
            , faces_names'''

            # face_recognition.face_landmarks(image)
            # print('findFacesOnVideo' , video_path)
            size_recovery_multiplier = 2
            size_reduction_factor = 0.25  # коэф уменьшения изобр Время работы: 4.513 min
            # коэф 1 уменьшения изобр Время работы: 43 min
            start_time = time.time()
            numFace = len(faces_names)

            df = pd.DataFrame({'frame': [0], 'name': [0], 'xyhw': [0], 'encode': [0]})
            cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)  # загруз видео
            # cv.CAP_DSHOW DirectShow (via videoInput)
            # cv.CAP_FFMPEG Open and record video file or stream using the FFMPEG library.
            # CAP_IMAGES  cv.CAP_IMAGES OpenCV Image Sequence (e.g. img_%02d.jpg)
            frame_wight = int(cap.get(3))
            frame_height = int(cap.get(4))
            fps = cap.get(cv2.CAP_PROP_FPS)

            # print('findFacesOnVideo fps ' , fps, ' video_path', video_path)

            if output:  # формирование выходногопотока
                video_out_file = video_path.split('.')[-2] + '_out.mp4'
                outVid = cv2.VideoWriter(video_out_file, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps,
                                         (frame_wight, frame_height))

            streams = True
            num = 0
            blok = 10

            name = 'noName'
            while streams:  # num < 50: #streams:

                ret, fram = cap.read()  # захват кадра
                frame = copy.copy(fram)
                # frame = cv2.copyMakeBorder(fram,0,0,0,0,cv2.BORDER_REPLICATE)

                if not ret:  # если кадр прочитан плохохо то берем следкющий
                    blok -= 1
                    if blok == 0:
                        streams = False
                    continue
                # print(num, 'np.max(frame)', np.max(frame), '  np.average(frame)', round(np.average(frame)), frame.shape)
                frshape = frame.shape

                # viewImage(fram)
                frameRes = cv2.resize(frame.copy(), (0, 0), None, size_reduction_factor,
                                      size_reduction_factor)  # подготовка кадра

                frameResBRGB = cv2.cvtColor(frameRes.copy(), cv2.COLOR_BGR2RGB)
                #frameResBRGB = frameRes[:, :, ::-1]
                # frameResBRGB =   frameRes
                # подготовка кадра
                # viewImage(frameResBRGB, waiK=0)
                #        viewImage(frame, waiK=0)

                facesLocations = face_recognition.face_locations(frameResBRGB, number_of_times_to_upsample= 2, model='cnn')# Пишут что большая, работает быстрей на GPU
                encodingFaces = face_recognition.face_encodings(frameResBRGB, facesLocations)
                # landmark = face_recognition.face_landmarks(frameResBRGB)    # Поиск черт лица

                # =============================================================================
                #         print(f'кадр {num} количество bobox:', len(facesLocations), '\n facesLocations',type(facesLocations), facesLocations,
                #               '\n encodingFaces:', type(encodingFaces), len(encodingFaces),'\n encodingFaces[0].shape'
                #               '\n landmark:', type(landmark), len(landmark),
                #               '\n encodeListKnown', type(encodeListKnown), len(encodeListKnown))
                # =============================================================================
                # print('\r', num, ' фитчи ',len(facesLocations), end=' ')
                #        viewImage(DravRectangleImage_face_rekogn(frame.copy(), facesLocations, size_reduction_factor), waiK=500)

                # viewImage(fram)
                if len(facesLocations) > 0:
                    box = len(facesLocations)
                    # Поверка на новые лица
                    # синхронный перебор кодов лиц и локаций
                    for encodingFace, faseLoc in zip(encodingFaces, facesLocations, ):

                        # vievLandmark(frame, landmark) #подрисовка черт
                        # viewImage(DravRectangleImage_face_rekogn(frame, facesLocations, size_reduction_factor),nameWindow=f'frame {num} faces {box} shape: {frshape}')

                        # viewImage(DravRectangleImage(frame, landmark), waiK=0, nameWindow='landmark')
                        # faseDict = {}

                        y1, x2, y2, x1 = faseLoc  # преобразование координат
                        faseLoc = (int(x1 / size_reduction_factor), int(y1 / size_reduction_factor),
                                   int(x2 / size_reduction_factor), int(y2 / size_reduction_factor))
                        x1, y1, x2, y2 = faseLoc
                        # y1, x2, y2, x1 = y1 * size_recovery_multiplier, x2 * size_recovery_multiplier, y2 * size_recovery_multiplier, x1 * size_recovery_multiplier
                        # поиск знакомых лиц
                        if len(encodeListKnown) > 0:
                            # поиск  в знакомых код лицах    кода лицаа
                            matches = face_recognition.compare_faces(encodeListKnown, encodingFace)  # лица муж?
                            faceDist = face_recognition.face_distance(encodeListKnown, encodingFace)  # расстояние

                            minFaseIdInd = np.argmin(faceDist)  # самое ближнее лицо маска True/False
                            #        [True]      [0.43499996]
                            # print(f'кадр {num} Расстояние matches: ', matches, 'faceDist:', faceDist)

                            if faceDist[minFaseIdInd] < 0.58:  # по моему бред просто выбор ближнего
                                name = faces_names[minFaseIdInd]
                                # viewImage(frame[y1:y2, x1:x2,...], waiK=0)
                            else:  # если новая фитча
                                # print('shape:', frame.shape, frame[y1:y2, x1:x2,...].shape)
                                viewImage(frame[y1:y2, x1:x2, ...], waiK=0, verbose=verbose, nameWindow='2 video')
                                numFace += 1
                                name = str(numFace)
                                # name = str(input(f'Введите Имя клиента пака похож на {faces_names[minFaseIdInd]}>>>>>>'))

                        else:  # первая новая фитча
                            viewImage(frame[y1:y2, x1:x2, ...], waiK=0, verbose=verbose, nameWindow='1 video')
                            numFace += 1
                            name = str(numFace)
                            # name = str(input('Введите Имя клиента>>>>>>'))

                        # print('name:', name, type(encodingFace) )

                        dfN = pd.DataFrame(columns=['frame', 'name', 'xyhw', 'encode'])
                        dfN['encode'] = dfN['encode'].astype(object)
                        dfN['xyhw'] = dfN['xyhw'].astype(object)
                        dfN['encode'] = [np.array(encodingFace), ]
                        dfN['name'] = int(name)
                        dfN['frame'] = num
                        dfN['xyhw'] = [faseLoc]
                        # print(dfN)

                        df = pd.concat([df, dfN])
                        # print('df.shape', df.shape)
                        # Занесение новых лиу в базу
                        if name not in faces_names:
                            faces_names.append(name)
                            encodeListKnown.append(encodingFace)
                            classNames.append(name)

                        # Вывод боксов с именем
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 3)  # отрисовка квадратов )
                        cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 0, 255), cv2.FILLED)  # минмибокс для текста
                        font = cv2.FONT_HERSHEY_DUPLEX
                        cv2.putText(frame, str(name), (x1, y2 - 6), font, 1.0, (0, 0, 0), 1)  # вывод текста

                        viewImage(frame, waiK=0, verbose=verbose, nameWindow='video')

                if output:
                    # cv2.imshow('img RGB', frame)
                    outVid.write(frame)  # Сохраняется видео файл
                    # viewImage(frame, waiK=800)
                    pass
                num += 1  # счетчик кадров
            # print(df)
            df_copy = copy.deepcopy(df)
            # df.to_csv(path,index=False)

            # print('faces_names', len(faces_names),faces_names )
            # print('encodeListKnown', len(encodeListKnown))
            # print('df.shape', df.shape)
            # print('Время работы:', (time.time()-start_time)/60, 'min')
            # %% Чтение и подготовка df
            ########################################

            unique_names = list(df_copy['name'].unique())
            class_frames_dict = {}
            for name in unique_names:
                frames = df_copy.loc[df_copy['name'] == name, 'frame'].tolist()
                class_frames_dict[name] = frames
            print("Количество обноруженных классов: ")
            print(unique_names)
            print("Словарь класс : [номера кадра в котором обноружен] ")
            print(class_frames_dict)
            return class_frames_dict, unique_names  # , df_copy

        #######################################
        def from_np_array(array_string):
            ''' преобразоваие в np тз строчки pd '''
            array_string = ','.join(array_string.replace('[ ', '[').split())
            return np.array(ast.literal_eval(array_string))

        def from_np_array1(array_string):
            ''' преобразоваие в np тз строчки pd '''
            array_string = ''.join(array_string.replace('( ', ')').split())
            return np.array(ast.literal_eval(array_string))

        def wievFrame_getName(faceLoc, frameNum, pathVideo):  # вывод лица и ввод имени
            ''' Вывод на экран лиц из списка и ввод их имен'''
            # print('faceLoc', faceLoc, len(faceLoc), type(faceLoc) )
            x1, y1, x2, y2 = faceLoc
            # print(x1, y1, x2, y2, type(x1))
            cap = cv2.VideoCapture(pathVideo)
            #    for fra in range(frameNum):
            #        _, frameIm = cap.read()
            #        pass
            # cap.set(cv2.CAP_PROP_POS_FRAMES,frameNum ) # выставить в нужный frame
            _, frameIm = cap.read()
            # print(frameIm.shape)

            viewImage(frameIm[y1:y2, x1:x2, ...], waiK=0, nameWindow='Face')
            cap.release()  # отпустить поток возможно не нужно для файлов
            return str(input(' Введите название фитчи>> '))

        def from_df_face_name(path, pathVideo):
            ''' считывание из .csv имен и кодов лиц перобразование прочитанных  np.array из строчки в реальный np.appay
            вывод на экран
            возврат:
                faces_names, имя классов
                encodeListKnown вектор вица (face recognition)
                '''
            df = pd.read_csv(path, converters={'encode': from_np_array, 'xyhw': from_np_array1})
            faces_names = []
            encodeListKnown = []

            listName = df['name'].unique().tolist()

            for name in listName:
                name_df = df[df['name'] == name]  # ['encode'].values

                faceLoc = name_df['xyhw'].values
                frame = name_df['frame'].values
                encode = name_df['encode'].values
                # print( type(encode),encode.shape[0])
                if encode.shape[0] > 2:
                    # print(name, faceLoc)
                    name = wievFrame_getName(faceLoc[0], frame[0], pathVideo)  # вывод лица и ввод имени

                    faces_names.append(str(name))
                    encodeListKnown.append(encode[0])
            return faces_names, encodeListKnown

        # %% KMeans
        from sklearn.cluster import KMeans  # Импортируем библиотеки KMeans для кластеризации
        def klastMean(embedding):
            '''  метод для поиска центра класса векторов одного лица не проверен'''
            # n_clusters = 6        # максимальное  количество кластеров
            cost = []  # контейнер под список
            kmean = KMeans(1)  # Создаем объект KMeans с i-классами
            kmean.fit(embedding)  # Проводим классетризацию
            centers = kmean.cluster_centers_

            # print('центр кластера', centers.shape)
            return centers[0]

        # %%

        # embedding = np.concatenate(items_meta.iloc[:]['embeddings'].values).reshape((-1,312))

        def normalisClassesEncod(path):
            '''  считывание из .csv имен и кодов лиц перобразование прочитанных  np.array из строчки в реальный np.appay
            поиск центра занчения класса  через kmeans для кодов лица'''
            faces_names = []
            encodeListKnown = []

            df = pd.read_csv(path, converters={'encode': from_np_array, 'xyhw': from_np_array1})
            classes = df['name'].unique()
            # print(classes)
            for oneClas in classes:
                faces_names.append(str(oneClas))

                df_oneClass = df[df['name'] == oneClas]['encode'].values
                # print(df_oneClass)
                if df_oneClass != 0:
                    encodeListKnown.append(klastMean(np.concatenate(df_oneClass).reshape(-1, df_oneClass[0].shape[0])))
            return faces_names, encodeListKnown

        def from_df_to_nameEncod(path):
            ''' считывание из .csv имен и кодов лиц перобразование прочитанных  np.array из строчки в реальный np.appay
            берется первый из кодов лица'''
            df = pd.read_csv(path, converters={'encode': from_np_array})
            faces_names = []
            encodeListKnown = []

            listName = df['name'].unique().tolist()
            for name in listName:
                encode = df[df['name'] == name]['encode'].values
                # print( type(encode),encode.shape[0])
                if encode.shape[0] > 2:
                    faces_names.append(str(name))
                    encodeListKnown.append(encode[0])
            return faces_names, encodeListKnown

        def renameClacess(faces_names):
            '''  Переименование загруженных классов
            перебирает по очереди и просит ввести новое если ентер то конец ввода'''
            for i in range(len(faces_names)):
                # print('class:', faces_names[i])
                inp = str(input(f' Class : {faces_names[i]} new name or enter>> '))
                if inp != '':
                    faces_names[i] = inp
                else:
                    break

            return faces_names

        def load_df(path):
            df = pd.read_csv(path)
            return df

        ''' Пооход по видео и сохранение найденных фитч в пандас потом вывод и запрос имен
        rename: - меню переименивание загруженных классов

        loadClasses - Загрузка сохраненных кодов лиц
        rename = True задание имен классов

        '''
        encodeListKnown = []
        faces_names = []

        print('go')
        if loadClasses:
            print('Загрузка сохраненных кодов лиц ВКЛЮЧЕННА')
            if rename:
                print('Переименивание загруженных классов включено')
            else:
                print('Переименивание загруженных классов ВЫКЛЮЧЕНО')
        else:
            print('Базы по лицам НЕТ')

        if loadClasses:  # агрузка сохраненных кодов лиц ВКЛЮЧЕННА'
            faces_names, encodeListKnown = normalisClassesEncod(csv_path)

            if rename:  # Переименивание загруженных классов
                # faces_names = renameClacess(faces_names)
                faces_names, encodeListKnown = from_df_face_name(csv_path, pathVideo)  # вариант с просмотром

            print('найденны имена:', faces_names, len(faces_names))
            print('найденны encod', type(encodeListKnown), len(encodeListKnown), encodeListKnown[0].shape)

        try:
            df = findFacesOnVideo(pathVideo, encodeListKnown=encodeListKnown, faces_names=faces_names)
        except:
            df = load_df(csv_path)

        scenes_with_people = {}
        for cls_id in pd.unique(df['name']):
            scenes_with_people[cls_id] = list(df[df['name']==cls_id]['frame'])
        print(scenes_with_people)
        # ====================================================================
        return scenes_with_people

    def labels_classifier(self,scenes_with_people,video_path):
        """
        Присваивание названий для уникальных классов лиц.

        :param scenes_with_people: {'people':[ {'people_in_scene': ['имя 1', 'имя 2', ...],
                                         'frames': ['frame 1','frame 2',...]       }, ...]}

        :return:

        classes: {'index_label': 'label_name', ...} соответствие лейблов
        распознанных классов и их названий.

        (предусмотреть имя = 'delete' для классов, в которых встречаются ошибки распознавания)
        """
        # Логика функции
        # ====================================================================
        classes = {}
        # Функция вывода примеров изображени обнаруженных классов (лиц)
        cap = cv2.VideoCapture(video_path)
        frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        duration = frames_count / fps

        classes_frames_count = {}
        classes_for_delete = []

        for class_id, frames_list in scenes_with_people.items():
            print(class_id, frames_list)
            if len(frames_list) < 5:
                print(f"Число кадров у класса {class_id} меньше 5")
                classes_for_delete.append(class_id)
                continue
            classes_frames_count[class_id] = len(frames_list)
            frames_to_show = np.random.choice(frames_list, 5, replace=False)
            fig, axs = plt.subplots(1, 5, figsize=(30, 20))
            for i, frame_num in enumerate(frames_to_show):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if not ret:
                    continue
                axs[i].imshow(frame)
                axs[i].set_title(f"{class_id} ({frame_num})")
                axs[i].axis('off')
            plt.show()
            classes[class_id] = input('Введите назвение класса: (0 - если класс не валидный)')
        cap.release()
        print(f'Список классов для удаления < 5 кадров')
        print(classes_for_delete)
        print(f'Словарь с классами и количеством кадров где обнаружены')
        print(classes_frames_count)
        print(classes)
        # ====================================================================
        return classes

if __name__ == '__main__':
    test_face_rec = Face_recognition()
    test_face_rec.labels_classifier(test_face_rec.my_face_recognition('Острые_козырьки.mp4',csv_path='timingVideo_2Павел04.csv'),'Острые_козырьки_out (2)Павел.mp4')
