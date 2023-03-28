from Separator import Separator
from Face_recognition import Face_recognition
from Video2description import Video2description
from Speaker import Speaker

import json
import time
import os
from moviepy.editor import VideoFileClip
from scenedetect import AdaptiveDetector


class Tiflo_system():
    def __init__(self, sep_threshold=27.0, type_of_scene_detector=None, speaker_language='ru'):
        self.separator = Separator(threshold=sep_threshold,
                                   type_of_detector=type_of_scene_detector,
                                   path_cutter='model_film_cut_v9_93')
        self.type_of_scene_detector = type_of_scene_detector
        self.face_rec = Face_recognition()
        self.video2desc = Video2description()
        self.speaker = Speaker(language=speaker_language)

    def processing(self, path_video):
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
        master_start_time = time.time()
        # Разметка на логические элементы, которые нужно озвучивать
        if not os.path.exists('voice_markup.json'):
            without_voice_markup = self.separator.voice_divide(path_video, min_time_scene=3)
            with open('voice_markup.json', 'w') as file:
                json.dump(without_voice_markup, file)

        else:
            print('Загружены предыдущие разметки видео')
            with open('voice_markup.json', 'r') as file:
                without_voice_markup = json.load(file)

        if not os.path.exists('scenes_markup.json'):
            if self.type_of_scene_detector:
                scenes_markup = self.separator.scene_divide_v1(path_video, min_time_scene=0)
            else:
                scenes_markup = self.separator.scene_divide_v2(path_video, min_time_scene=0)

            with open('scenes_markup.json', 'w') as file:
                json.dump(scenes_markup, file)

        else:
            print('Загружены предыдущие разметки видео')
            with open('scenes_markup.json', 'r') as file:
                scenes_markup = json.load(file)

        print(without_voice_markup)
        print(scenes_markup)

        # union_markup = self.union_divide(without_voice_markup, scenes_markup) # Ситуативно

        # Генерация описаний к видео-фрагментам
        # (С переводом на русский / можно попробовать с union_markup вместо without_voice_markup)
        # Тип воборки кадров для описания сцены:                                        'cpu_version', 'light', 'hard'
        descriptions = {}
        descriptions = self.video2desc.video2description(path_video, without_voice_markup,procces_type='cpu_version')  # Выводов столько же, сколько и разбитых диапазонов для аудиосопровождения
        if not os.path.exists('descriptions.json'):
            with open('descriptions.json', 'w') as file:
                json.dump(descriptions, file)
        else:
            with open('descriptions.json', 'r') as file:
                descriptions = json.load(file)
        print(descriptions)

        # Поиск лиц внутри сцен, для описания диалогов
        classes, scenes_with_people = self.face_rec.video2face_recognition(path_video,scenes_markup)  # Выводов столько же, сколько и сцен найдено

        with open('classes.json','w') as file:
           json.dump(classes, file)
        with open('scenes_with_people.json','w') as file:
           json.dump(scenes_with_people, file)

        print(classes)
        print(scenes_with_people)
        print(f'Время выполнения всех вычислений: {round(time.time() - master_start_time,2)} секунд')
        # Генерация озвучки text2speech
        video = VideoFileClip(path_video)
        fps = int(video.fps)
        self.speaker.text2speech(descriptions, markup=without_voice_markup, fps=fps)
        return
        #self.speaker.text2speech(classes, fps=fps)

    def database_request(self):
        """
        Функция передаёт все данные обработанного видео в базу данных на сайте

        :return:
        """
        pass


if __name__ == '__main__':
    video_path = 'Острые_козырьки.mp4'
    threshold = 27.0

    # Передать в систему, как тип разбиения видео на сцены
    type_of_detector = AdaptiveDetector
    test = Tiflo_system(sep_threshold=threshold)
    test.processing(path_video=video_path)
