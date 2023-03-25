
class Tiflo_system():
    def __init__(self):
        model_cutter = None
        model_vice_recognition = None
        model_face_recognition = None
        model_image2text = None

    def processing(self, video):
        """
        Функция создаёт разметку и генерирует аудиосопровождение. Сохраняет результаты в папку temp либо
        сразу передаёт их на выход в виде словаря.

        :param video: переменная типа 'moviepy.editor.VideoFileClip'. Входное видео.

        :return: - dict {'status':
                         'voice_scenes':
                         'labels':
                         'audios':
                         'face_classes':
                         'faces_on_frames':}
        """
        # Разметка на логические элементы, которые нужно озвучивать
        without_voice_markup = self.voice_divide()
        scenes_markup = self.scene_divide()
        union_markup = self.union_divide(without_voice_markup, scenes_markup) # Ситуативно

        # Генерация описаний к видео-фрагментам
        # (С переводом на русский / можно попробовать с union_markup вместо without_voice_markup)
        descriptions = self.video2description(video,without_voice_markup)      # Выводов столько же, сколько и разбитых диапазонов для аудиосопровождения

        # Поиск лиц внутри сцен, для описания диалогов
        classes, scenes_with_people = self.video2face_recognition(video,scenes_markup) # Выводов столько же, сколько и сцен найдено

        # Генерация озвучки text2speech
        self.text2speech(descriptions)
        self.text2speech(classes)

    def database_request(self):
        """
        Функция передаёт все данные обработанного видео в базу данных на сайте

        :return:
        """
        pass

