
#!git lfs install
#!git clone https://huggingface.co/nlpconnect/vit-gpt2-image-captioning
#!pip install happytransformer
#!pip install sacremoses

from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer, pipeline
from transformers import FSMTForConditionalGeneration, FSMTTokenizer
import torch
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from skimage import io
import cv2
import os
import time

import imageio
import imageio.plugins.ffmpeg as ffmpeg
import moviepy
import moviepy.editor

from pydub import AudioSegment

import torch
torch.set_num_threads(1)

from IPython.display import Audio
from pprint import pprint
import json

from Utils import Utils

class Video2description():
    def __init__(self):
        self.SAMPLING_RATE = 16000  # рекомендовано авторами модели
        self.size = 3
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # translate model
        mname = "facebook/wmt19-en-ru"
        self.tokenizer_translate = FSMTTokenizer.from_pretrained(mname)
        self.model_translate = FSMTForConditionalGeneration.from_pretrained(mname)
        #self.model_translate.to(self.device)
        # summarization model
        self.summarizer = pipeline("summarization", model="philschmid/bart-large-cnn-samsum")
        # model image2text
        self.model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

        print(f'Выбрано устройство: {self.device}')
        self.model.to(self.device)

        # параметры модели
        self.max_length = 16
        self.num_beams = 4
        self.gen_kwargs = {"max_length": self.max_length, "num_beams": self.num_beams}

    def video2description(self, path_video, markup, procces_type):
        """
        Генерация русских описаний к видео-фрагментам.

        :param video: переменная типа 'moviepy.editor.VideoFileClip'. Входное видео.

                                              начало описываемого промежутка | его конец
        :param markup: разметка видео типа: {'voice_markup':[ {'start_frame': <int>, 'end_frame': <int> },
                                                       {'start_frame': <int>, 'end_frame': <int> }, ... ] }

        :return:
        rus_descriptions: вывод предсказаний на русском языке следующего типа:
        {'descriptions':[{'описание временного промежутка'}, ... ]} ,
        где каждый элемент {'описание временного промежутка'} соответствует своему промежутку из markup.

        """
        eng_descriptions = self._video2description_eng(path_video, markup, procces_type)
        rus_descriptions = self.interpreter(eng_descriptions)
        return rus_descriptions


    def _video2description_eng(self, path_video, markup, procces_type):
        """
        Генерация английских описаний к видео-фрагментам.

        :param video: переменная типа 'moviepy.editor.VideoFileClip'. Входное видео.

        :param markup: разметка  типа: {'voice_markup':[ {'start_frame_without_speech': <int>, 'end_frame_without_speech': <int> },
                                                         {'start_frame_without_speech': <int>, 'end_frame_without_speech': <int> }, ... ] }

        :return:

        descriptions: вывод предсказаний на английском языке следующего типа:
        {'descriptions':[{'scene description text'}, ... ]} ,
        где каждый элемент {'scene description text'} соответствует своему промежутку из markup.

        """

        video = cv2.VideoCapture(path_video)

        all_predictions = []

        iteration_num = 0
        for pair in markup['voice_markup']:
            start, end = pair['start_frame'], pair['end_frame']
            if ((procces_type == 'cpu_version') or (self.device == 'cpu')):
                frames_indexes = (start,end)
            elif procces_type == 'light':
                frames_indexes = Utils.light_dynamic_cutter(start,end)
            elif procces_type == 'hard':
                frames_indexes = Utils.dynamic_cutter(start, end)
            else:
                print('Не выбран ни один тип выборки кадров для описания.')
                return {}
            print(f'Выбранные индексы:{frames_indexes}')
            images = []
            for frame_id in frames_indexes:
                video.set(cv2.CAP_PROP_POS_FRAMES, frame_id-1)
                ret, frame = video.read()
                images.append(frame)

            print(np.array(images).shape)

            pixel_values = self.feature_extractor(images=np.array(images), return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)

            start_time_part = time.time()
            output_ids = self.model.generate(pixel_values, **self.gen_kwargs)

            preds = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            preds = [pred.strip() for pred in preds]

            all_predictions.append(preds)

            start_time_part = time.time() - start_time_part
            if iteration_num <= 1:
                print(f'Примерное время ожидания генерации текста: {round(start_time_part*len(markup["voice_markup"])-1,2)} секунд')
                iteration_num += 1

        string_list = all_predictions.copy()
        # Цикл по всем элементам списка списков
        for i in range(len(string_list)):
            # Цикл по всем элементам текущего списка
            for j in range(len(string_list[i])):
                # Получаем текущую строку из списка
                string = string_list[i][j]

                # Добавляем точку к строке с помощью оператора +
                string = string + "."

                # Заменяем элемент списка на новую строку
                string_list[i][j] = string

        # Выводим измененный список
        print(f'Список строк для всего фильма: {string_list}')

        merged_list = []
        # Обходим список списков строк
        for i in string_list:
            # Объединяем элементы текущего списка в одну строку
            merged_string = ''.join(i)

            # Добавляем объединенную строку в новый список
            merged_list.append(merged_string)

        # Выводим новый список с объединенными строками
        print(f'Объединённый список: {merged_list}')

        summarization = []
        for item in merged_list:
            try:
                summary = self.summarizer(item)
                summarization.append(summary)
            except IndexError:
                # Разбиваем строку на две части
                half = len(item) // 2
                first_half = item[:half]
                second_half = item[half:]
                # Запускаем обработку каждой половины
                first_summary = self.summarizer(first_half)
                second_summary = self.summarizer(second_half)
                # Склеиваем результаты
                summary = first_summary + second_summary
                summarization.append(summary)

        result_dict = {}
        for i in range(len(summarization)):
            key = i
            if isinstance(summarization[i], list):
                value = summarization[i][0]["summary_text"]
                if len(summarization[i]) > 1:
                    for j in range(1, len(summarization[i])):
                        value += " " + summarization[i][j]["summary_text"]
            else:
                value = summarization[i]["summary_text"]
            result_dict[key] = value
        print(f'Описания для каждой сцены: {result_dict}')

        return result_dict


    def interpreter(self, eng_descriptions):
        """
        Переводчик.

        :param eng_descriptions: вывод предсказаний на английском языке следующего типа:
        {'descriptions':[{'scene description text'}, ... ]} ,
        где каждый элемент {'scene description text'} соответствует своему промежутку из markup.

        :return:

        rus_descriptions: вывод предсказаний на русском языке следующего типа:
        {'descriptions':[{'описание временного промежутка'}, ... ]} ,
        где каждый элемент {'описание временного промежутка'} соответствует своему промежутку из markup.
        """
        # Логика функции
        # ====================================================================
        rus_descriptions = {}
        for key, value in eng_descriptions.items():
            input = value
            input_ids = self.tokenizer_translate.encode(input, return_tensors="pt")
            outputs = self.model_translate.generate(input_ids)
            decoded = self.tokenizer_translate.decode(outputs[0], skip_special_tokens=True)
            rus_descriptions[key] = decoded
        # ====================================================================
        return rus_descriptions

if __name__ == '__main__':
    test = Video2description()
    path_video = '/content/drive/MyDrive/ML/MTS CUP/НЕ ВРЕМЯ УМИРАТЬ_720.mp4'
    test.video2description_eng(path_video)