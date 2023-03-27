#              Imports
#         -----------------
import cv2
import numpy as np
import time

from Utils import Utils
#         -----------------

class Video():
    def __init__(self,temp_path,fps,width,height):
        self.W = width
        self.H = height
        self.temp_path = temp_path
        self.count_frames_video = 0
        self.count_frames_pause = 0

        self.out = cv2.VideoWriter(f'{temp_path}/1.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (self.W, self.H))

    @staticmethod
    def generator_particular_frames_from_video( video_path, W, H, part_size, part_count):
        cap = cv2.VideoCapture(video_path)
        # cap.set(cv2.CAP_PROP_FPS, 1)
        numpy_images = np.zeros((part_size, 2, H, W, 3), dtype='uint8')
        success, img1 = cap.read()
        resized1 = cv2.resize(img1, (W, H), interpolation=cv2.INTER_AREA)
        success, img2 = cap.read()
        i = 0
        while i != part_count:
            start_time = time.time()
            real_size = 0
            while ((real_size < part_size) and (success)):
                resized2 = cv2.resize(img2, (W, H), interpolation=cv2.INTER_AREA)

                numpy_images[real_size, 0], numpy_images[real_size, 1] = resized1, resized2

                resized1 = resized2
                success, img2 = cap.read()
                real_size += 1
            # print(real_size)
            i += 1
            yield numpy_images[:real_size], real_size

    def print_frames_info(self):
        print('')
        print(f'Количество кадров видео: {self.count_frames_video}')
        print(f'Количество кадров пауз:  {self.count_frames_pause}')

    def save_batch(self, frames, cut_indexes, offset_list):
        for i in range(0, cut_indexes.shape[0] - 1):
            print(cut_indexes[i]+1,cut_indexes[i+1])
            # Запись первого кадра в батче
            if i == 0:
                self.out.write(frames[cut_indexes[i], 0])
                self.count_frames_video += 1

            for index in range(cut_indexes[i] + 1, cut_indexes[i + 1]+1):
                # Пауза для озвучки первого кадра в батче
                if index == cut_indexes[i] + 1:
                    for _ in range(offset_list[i]):
                        self.out.write(frames[index,0])
                        self.count_frames_pause += 1

                # Запись кадра
                self.out.write(frames[index, 0])
                self.count_frames_video += 1

        self.print_frames_info()