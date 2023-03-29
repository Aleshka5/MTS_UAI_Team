import pprint
import json
import time

from Cutter import Cutter
from Video import Video
from Utils import Utils

import torch
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
from scenedetect import AdaptiveDetector, open_video, SceneManager
#pip install PySoundFile
#pip install SoundFile

class Separator():
    def __init__(self,threshold,type_of_detector,path_cutter):
        # Voice divide
        self.model, self.utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                model='silero_vad',
                                force_reload=True)
        self.SAMPLING_RATE = 16000 # Гц
        self.min_time_scene = 3    # секунды

        # Scene divide_v1
        self.threshold = threshold
        self.type_of_detector = type_of_detector
        # v2
        self.path_cutter_model = path_cutter

    def voice_divide(self, path_video, min_time_scene):
        """
        Создание разметки участков исходя из присутствия/отсутвия на них речи.

        :param path_video: Путь до исследуемого видео. Входное видео.

        :param min_time_scene: переменная типа int. Количество секунд.
        Нужна для регулирования минимальной длинны сцены для озвучки.
        (Априорная оценка использования этого парамметра - 3 секунды)

        :return:
                   кадр начала промежутка без человеческой речи     кадр конца промежутка без человеческой речи
        dict_markup: {'voice_markup':[{'start_frame': <int>,          'end_frame': <int> },
                                      {'start_frame': <int>,          'end_frame': <int> }, ... ]}
        """

        # вытаскиваем аудио из видео
        self.video = VideoFileClip(path_video)
        audio = self.video.audio
        audio.write_audiofile('audio.wav')
        duration = audio.duration

        # определяем количество кадров
        fps = int(self.video.fps)

        # загружаем утитлиты
        (get_speech_timestamps, _, read_audio, VADIterator, _) = self.utils

        # получаем timestemp из всего видео
        vad_iterator = VADIterator(self.model)

        # словарь, в который будем собирать
        # метки начала и конца сцены без голоса
        dict_markup = {}

        # вспомогательный список для сбора меток голоса
        l_speech = []

        # вспомогательный список для сбора меток без голоса
        l_without_speech = []

        window_size_samples = 1536  # размер окна рекомендован автором модели

        # загружаем аудио в формате wav
        wav = AudioSegment.from_wav('audio.wav')
        wav = read_audio('audio.wav', sampling_rate=self.SAMPLING_RATE)

        for i in range(0, len(wav), window_size_samples):
            chunk = wav[i: i + window_size_samples]
            if len(chunk) < window_size_samples:
                break
            speech_dict = vad_iterator(chunk, return_seconds=True)  # получаем метки голоса
            if speech_dict:
                l_speech.append(speech_dict)

        # собираем метки участков без голоса
        for i in range(len(l_speech)):
            d = {}  # вспомогательный словарь для сбора меток без речи

            # определяем отрезок без голоса до первого высказывания
            if i == 0:
                d['start_frame'] = 0
                d['end_frame'] = int(l_speech[i]['start'] * fps)
                segment = (d['end_frame'] - d['start_frame']) / fps
                if segment > min_time_scene:  # проверяем размер озвученного отрезка в соотвествии с заданным значением
                    l_without_speech.append(d)
                else:
                    continue
            if i != 0:

                # т.к. структура полученного списка с метками голоса имеет следующий вид:
                # {'start': 1.8} {'end': 2.3} {'start': 2.7} {'end': 4.8} ... {'start': 138.3} {'end': 140.9}
                # мы выбираем 'end' как начало отрезка без голоса и 'start', как конец отрезка без голоса
                # т.о. берем только четные i

                if i % 2 == 0:
                    d['start_frame'] = int(l_speech[i - 1]['end'] * fps)
                    d['end_frame'] = int(l_speech[i]['start'] * fps)
                    segment = (d['end_frame'] - d['start_frame']) / fps
                    if segment > min_time_scene: # проверяем размер озвученного отрезка в соотвествии с заданным значением
                        l_without_speech.append(d)
                        continue
                    else:
                        continue

                # определяем отрезок тишины в конце видеоролика
                if i == len(l_speech) - 1:
                    d['start_frame'] = int(l_speech[i]['end'] * fps)  # находим последнюю метку речи
                    d['end_frame'] = int(duration * fps)  # последняя метка - конец ролика
                    segment = (d['end_frame'] - d['start_frame']) / fps
                    if segment > min_time_scene:  # проверяем размер озвученного отрезка в соотвествии с заданным значением
                        l_without_speech.append(d)
                        continue
                    else:
                        continue

                # пропускаем нечетные i
                if i % 2 != 0:
                    continue
                    # определяем отрезок тишины в конце видеоролика

        dict_markup['voice_markup'] = l_without_speech
        # ====================================================================
        return dict_markup


    def scene_divide_v1(self, path_video, min_time_scene=0):
        """
        Создание разметки участков исходя из смены сцен.

        :param path_video: Путь до исследуемого видео. Входное видео.

        :param min_time_scene: переменная типа int. Нужна для регулирования
        минимальной длинны сцены для озвучки. (Ставим 0, если используем поиск сцен
        только для face_recognition)

        :return:
                                         кадр начала сцены          кадр конца сцены
        dict_markup - {'scenes_markup':[{'start_frame': <int>, 'end_frame': <int> },
                                       {'start_frame': <int>, 'end_frame': <int> }, ... ]}
        """
        dict_markup = {}
        # Логика функции
        # ====================================================================
        dict_markup = {}
        self.path_video = path_video
        video = open_video(self.path_video)

        fps = int(video.frame_rate)

        scene_manager = SceneManager()
        scene_manager.add_detector(self.type_of_detector(self.threshold))

        # детектириуем все сцены в видео от начала к концу
        scene_manager.detect_scenes(video)
        l_detect = scene_manager.get_scene_list()
        # get_scene_list - возвращает список сцен с метками начала и конца
        # в секундах и с указанием номера кадра
        # вспомогательный список, в который будем собирать кадры начало/конец сцен
        l_scene = []
        for i in range(len(l_detect)):
            if (l_detect[i][1].get_seconds() - l_detect[i][0].get_seconds()) > min_time_scene:
                l_scene.append({'start_frame': int(l_detect[i][0].get_seconds()*fps), 'end_frame': int(l_detect[i][1].get_seconds()*fps)})
        dict_markup['scenes_markup'] = l_scene
        # ====================================================================
        return dict_markup

    def scene_divide_v2(self,path_video, min_time_scene=0, batch_size=1500, parts_count=-1):
        video = VideoFileClip(path_video)

        fps = int(video.fps)
        #start_index = len(path_video) - path_video[::-1].find('/')
        #dir_name = path_video[start_index:-4]

        cutter = Cutter(path=self.path_cutter_model)
        my_utils = Utils()
        iteration = 0
        frames_offset = 0
        all_cut_indexes = [0]
        process_time = time.time()
        for batch_frames, size in Video.generator_particular_frames_from_video(
                video_path=path_video,
                W=128,
                H=54,
                part_size=batch_size,
                part_count=parts_count):
            # Если размер прочитанного фрагмента видео меньше батча - указать, что идёт последняя итерация
            # Для этого рассчитаем количество частей как "текущая итерация + 1"
            if size != batch_size:
                parts_count = iteration + 1
            #          --------------------------
            # Разделение на сцены
            predict = cutter.predict(batch_frames, 10)
            my_utils.cutter_correcter(predict)  # Убрать соседние разделения на кадры, если такие есть

            # Добавление в батч начала сцены прошлого батча
            batch_frames, cut_indexes = my_utils.batch_correcting(batch_frames, predict, iteration,
                                                                  parts_count)

            if cut_indexes.shape[0] > 1:
                all_cut_indexes.extend((cut_indexes[1:] + frames_offset).tolist())
                frames_offset += cut_indexes[-1] + 1
            else:
                frames_offset += size + 1
            iteration += 1
            # Если были прочитаны все кадры исходного видео - закончить обработку
            if size != batch_size:
                break

        cut_pairs = []
        write_index = 0
        for i in range(len(all_cut_indexes) - 1):
            if all_cut_indexes[i + 1] - all_cut_indexes[i] > min_time_scene * fps:
                cut_pairs.append({'start_frame': all_cut_indexes[i], 'end_frame': all_cut_indexes[i+1]})
                write_index += 1

        dict_markup = {}
        dict_markup['scenes_markup'] = cut_pairs
        print(time.time() - process_time, 'sec')
        return dict_markup


    # Ситуативная
    def union_divide(self, without_voice_markup, scenes_markup):
        """
        Объединение двух разметок в одну более подробную, учитывающую как речь, так и смену сцен.
        (Опциональная функция. Пока не является необходимой)
                                                 кадр начала сцены          кадр конца сцены
        :param scenes_markup: {'scenes_markup':[{'start_frame_scene': <int>, 'end_frame_scene': <int> },
                                                {'start_frame_scene': <int>, 'end_frame_scene': <int> }, ... ]}
                           кадр начала промежутка без человеческой речи     кадр конца промежутка без человеческой речи
        :param without_voice_markup: {'voice_markup':[{'start_frame_without_speech': <int>, 'end_frame_without_speech': <int> },
                                                      {'start_frame_without_speech': <int>, 'end_frame_without_speech': <int> }, ... ]}
        :return:
        dict_markup: {'voice_markup':[{'start_frame_without_speech': <int>, 'end_frame_without_speech': <int> },
                                      {'start_frame_without_speech': <int>, 'end_frame_without_speech': <int> }, ... ]}
        Пример:
        # - кадры, на которых нет речи людей (подходящие)
        \ - start_frame
        / - end_frame
        | - и start_frame и end_frame одновременно: "/\". Другими словами это склейка двух сцен
                           start                                       end
        scenes_markup        # \   |    | |      |    |    |   |     |  /
        without_voice_markup # \#####/      \#######/    \########/     /
        dict_markup          # \###|#/      \####|##/    \#|###|##/     /
        """
        dict_markup = {}
        # Логика функции
        # ====================================================================
        final = []

        # переведем словари в списки
        # по участкам без голоса
        l_without_voice_markup = []
        for i in range(len(without_voice_markup['voice_markup'])):
            l_without_voice_markup.append([without_voice_markup['voice_markup'][i]['start_frame'],
                                           without_voice_markup['voice_markup'][i]['end_frame']])
        # по сценам
        l_scenes_markup = []
        for i in range(len(scenes_markup['scenes_markup'])):
            l_scenes_markup.append([scenes_markup['scenes_markup'][i]['start_frame_scene'],
                                    scenes_markup['scenes_markup'][i]['end_frame_without_scene']])

        # ищем пересечения

        i = 0
        j = 0

        while i < len(l_without_voice_markup) and j < len(l_scenes_markup):
            interval1 = l_without_voice_markup[i]
            interval2 = l_scenes_markup[j]

            if interval1[1] <= interval2[0]:
                i += 1
            elif interval2[1] <= interval1[0]:
                j += 1
            else:
                start = max(interval1[0], interval2[0])
                end = min(interval1[1], interval2[1])
                final.append([start, end])
                if interval1[1] > interval2[1]:
                    j += 1
                else:
                    i += 1
        dict_markup['voice_markup_final'] = final
        # ====================================================================
        return dict_markup


if __name__ == '__main__':
    video_path = 'Острые_козырьки.mp4'

    threshold = 27.0
    type_of_detector = AdaptiveDetector

    test_divide = Separator(threshold=threshold,
                            type_of_detector=type_of_detector,
                            path_cutter='model_film_cut_v9_93')

    print(test_divide.scene_divide_v1(video_path,min_time_scene=3))
    print(test_divide.scene_divide_v2(video_path,min_time_scene=3))
    #print(test_divide.voice_divide(video_path,min_time_scene=3))