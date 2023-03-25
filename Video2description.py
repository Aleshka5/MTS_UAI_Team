
class Video2description():
    def __init__(self):
        pass
    def video2description(self, video, markup):
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
        eng_descriptions = self.video2description_eng(video, markup)
        rus_descriptions = self.interpreter(eng_descriptions)
        return rus_descriptions


    def video2description_eng(self, video, markup):
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
        descriptions = {'descriptions':[]}
        for pair in markup['voice_markup']:
            start, end = pair['start_frame_without_speech'], pair['end_frame_without_speech']

            # Логика функции
            # ====================================================================
            description = 'predict'  # вместо 'predict' полжны попадать настоящие сгенерированные тексты
            # ====================================================================

            descriptions['descriptions'].append(description)
        return descriptions


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
        rus_descriptions = 'описание'
        # ====================================================================
        return rus_descriptions