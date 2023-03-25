
class Face_recognition():
    def __init__(self):
        pass
    def video2face_recognition(self,video,scenes_markup):
        """

        :param video: переменная типа 'moviepy.editor.VideoFileClip'. Входное видео.
                                                  кадр начала сцены          кадр конца сцены
        :param scenes_markup: {'scenes_markup':[{'start_frame_scene': <int>, 'end_frame_scene': <int> },
                                                {'start_frame_scene': <int>, 'end_frame_scene': <int> }, ... ]}

        :return:

        classes: {'index_label': 'label_name', ...} соответствие лейблов
        распознанных классов и их названий.

        scenes_with_people: {'people':[ {'people_in_scene': ['имя 1', 'имя 2', ...],
                                         'frames': ['frame 1','frame 2',...]       }, ...]} ,
        где каждый список имён ['имя 1', 'имя 2', ...] соответствует своей сцене из scenes_markup.
        (предусмотреть имя = 'delete' для классов, в которых встречаются ошибки распознавания)
        """
        scenes_with_people = self.face_recognition(video)
        classes = self.labels_classifier(scenes_with_people)

        return classes, scenes_with_people

    def face_recognition(self):
        """
        Распознавание лиц.

        :param video: переменная типа 'moviepy.editor.VideoFileClip'. Входное видео.

        :return:

        scenes_with_people: {'people':[ {'people_in_scene': ['имя 1', 'имя 2', ...],
                                         'frames': ['frame 1','frame 2',...]       }, ...]}
        """

        # Логика функции
        # ====================================================================

        # ====================================================================

    def labels_classifier(self,scenes_with_people):
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

        # ====================================================================