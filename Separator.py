
class Separator():
    def __init__(self):
        pass

    def voice_divide(self, video, min_time_scene):
        """
        Создание разметки участков исходя из присутствия/отсутвия на них речи.

        :param video: переменная типа 'moviepy.editor.VideoFileClip'. Входное видео.

        :param min_time_scene: переменная типа int. Нужна для регулирования
        минимальной длинны сцены для озвучки. (Ставим 0, если используем поиск сцен
        только для face_recognition)

        :return:
                                         кадр начала сцены          кадр конца сцены
        dict_markup - {'scenes_markup':[{'start_frame_scene': <int>, 'end_frame_scene': <int> },
                                       {'start_frame_scene': <int>, 'end_frame_scene': <int> }, ... ]}
        """
        dict_markup = {}
        # Логика функции
        # ====================================================================

        # ====================================================================
        return dict_markup


    def scene_divide(self, video, min_time_scene):
        """
        Создание разметки участков исходя из смены сцен.

        :param video: переменная формата 'moviepy.editor.VideoFileClip'. Входное видео.

        :param min_time_scene: переменная типа int. Количество секунд.
        Нужна для регулирования минимальной длинны сцены для озвучки.
        (Априорная оценка использования этого парамметра - 3 секунды)

        :return:
                           кадр начала промежутка без человеческой речи     кадр конца промежутка без человеческой речи
        dict_markup: {'voice_markup':[{'start_frame_without_speech': <int>, 'end_frame_without_speech': <int> },
                                      {'start_frame_without_speech': <int>, 'end_frame_without_speech': <int> }, ... ]}
        """
        dict_markup = {}
        # Логика функции
        # ====================================================================

        # ====================================================================
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

        # ====================================================================
        return dict_markup