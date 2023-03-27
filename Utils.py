#              Imports
#         -----------------
import numpy as np
import cv2
#         -----------------

class Utils():
    def __init__(self):
        self.sub_last_cadr = []
        self.add_last_cadr = []

    @staticmethod
    def dynamic_cutter(index_0, index_1):
        interval = 5
        length = index_1 - index_0
        intervals = np.array([10, 20, 50, 100, 200])
        cuts = [3, 5, 9, 17, 33]
        indexes = [0]
        if length >= interval:
            indexes.append(length)
            if length < 10:
                return np.array(indexes, dtype='int32') + index_0

            else:
                i = np.where(intervals <= length)[0][-1]
                while cuts[i] > len(indexes):
                    temp_indexes = []
                    for index in range(len(indexes) - 1):
                        new_index = int((indexes[index] + indexes[index + 1]) / 2)
                        temp_indexes.append(new_index)
                    indexes.extend(temp_indexes)
                    indexes.sort()
        return np.sort(np.array(indexes, dtype='int32') + index_0)

    @staticmethod
    def light_dynamic_cutter(index_0, index_1):
        interval = 5
        length = index_1 - index_0
        intervals = np.array([10, 20, 50, 100, 200])
        cuts = [3, 5, 5, 9, 9]
        indexes = [0]
        if length >= interval:
            indexes.append(length)
            if length < 10:
                return np.array(indexes, dtype='int32') + index_0

            else:
                i = np.where(intervals <= length)[0][-1]
                while cuts[i] > len(indexes):
                    temp_indexes = []
                    for index in range(len(indexes) - 1):
                        new_index = int((indexes[index] + indexes[index + 1]) / 2)
                        temp_indexes.append(new_index)
                    indexes.extend(temp_indexes)
                    indexes.sort()
        return np.sort(np.array(indexes, dtype='int32') + index_0)

    def predict2dict(self,obj_det_predict):
        dict_count = {}
        dict_mean = {}

        for classes in obj_det_predict:
            count_classes = classes.count(',')
            if count_classes == 1:
                # print(classes[:-1].split(' '))
                count, class_ = classes[:-1].split(' ')
                if class_ not in dict_count.keys():
                    dict_count[class_] = 1
                    dict_mean[class_] = int(count)
                else:
                    dict_count[class_] += 1
                    dict_mean[class_] += int(count)

            else:
                for predict in classes[:-1].split(','):
                    if predict.count(',') == 1:
                        count, class_ = predict[:-1].split(' ')
                    else:
                        count, class_ = predict.split(' ')

                    if class_ not in dict_count.keys():
                        dict_count[class_] = 1
                        dict_mean[class_] = int(count)
                    else:
                        dict_count[class_] += 1
                        dict_mean[class_] += int(count)

        for key in dict_mean.keys():
            dict_mean[key] = int(dict_mean[key] / dict_count[key])

        return dict_mean

    def back_relation(self, frames, cut_indexes, iteration, parts_count, debug=False):
        if iteration != parts_count - 1:
            #self.last_cadr = np.array(frames[cut_indexes[cut_indexes.shape[0]-2]+1:])
            if debug:
                for frame in self.last_cadr:
                    cv2.imshow('render', frame)
                    cv2.waitKey(100)
                cv2.destroyAllWindows()

    def merge_video_audio(self):
        pass

    def cutter_correcter(self,predict):
        all_indexes = np.where(predict > 40)[0]
        missed_indexes = np.array([0])

        for i in range(1, all_indexes.shape[0]):
            if all_indexes[i] - all_indexes[i - 1] in [1, 2, 3]:  # Какой промежуток между кадрами считать случайным
                missed_indexes = np.append(missed_indexes, all_indexes[i])

        predict[missed_indexes] = 0
        predict[np.where(predict < 40)] = 0

    def batch_correcting(self, batch_frames, predict, iteration, parts_count):
        pred_indexes = np.where(predict != 0)[0]
        if np.array([predict.shape[0] - 1])[0] not in pred_indexes:
            cut_indexes = np.hstack([np.array([0]), pred_indexes, np.array([predict.shape[0] - 1])])
        else:
            cut_indexes = np.hstack([np.array([0]), pred_indexes])

        self.sub_last_cadr = np.array(batch_frames[cut_indexes[-2] + 1:])
        #Utils.scroll_batch_images(self.sub_last_cadr)
        # Если идёт не последняя итерация (батч изображений)
        if iteration != parts_count - 1:

            # Если сцена оказалась больше батча
            if cut_indexes.shape[0] <= 2:
                self.add_last_cadr = np.concatenate([np.array(self.add_last_cadr), np.array(batch_frames)], axis=0)
                return np.array([]), np.array([0])

            # Если не сущесвует кешированной сцены с прошлого батча
            if self.add_last_cadr == []:
                batch_frames = batch_frames[:cut_indexes[-2]+1]
                self.add_last_cadr = np.array(self.sub_last_cadr)
            else:
                batch_frames = np.concatenate([np.array(self.add_last_cadr), batch_frames[:cut_indexes[-2] + 1]],axis=0)

                cut_indexes[1:] += np.array(self.add_last_cadr.shape[0])

                self.add_last_cadr = np.array(self.sub_last_cadr)

            return batch_frames, cut_indexes[:-1]

        else:
            # Если последняя сцена оказалась больше батча
            if cut_indexes.shape[0] <= 2:
                self.add_last_cadr = np.concatenate([np.array(self.add_last_cadr), np.array(batch_frames)], axis=0)
                return np.array(self.add_last_cadr), cut_indexes[:]

            if self.add_last_cadr != []:
                batch_frames = np.concatenate([np.array(self.add_last_cadr), batch_frames],axis=0)

                cut_indexes[1:] += np.array(self.add_last_cadr.shape[0])

                self.add_last_cadr = np.array(self.sub_last_cadr)
            return batch_frames, cut_indexes[:]

    @staticmethod
    def scroll_batch_images(images_list):
        H = 540
        W = 1280


        finish_data_preprocessing = False
        current_cadr_number = 0
        if images_list.shape[1] == 2:
            resized1 = np.array(images_list[:, 0])
        else:
            resized1 = images_list

        while not finish_data_preprocessing:
            try:
                # ===================================================================================
                cv2.imshow('Result', cv2.resize(resized1[current_cadr_number], (W, H), interpolation=cv2.INTER_AREA))
                # ===================================================================================
                key = cv2.waitKey(1)
                #if key != -1:
                    #print(key)
                if key == 54:  # Right scroll (6 on Numpad)
                    current_cadr_number += 1
                if key == 52:  # Left scroll (4 on Numpad)
                    current_cadr_number -= 1
                if key == 27:  # Esc - Escape with save progress
                    cv2.destroyAllWindows()
                    finish_data_preprocessing = True
                    return
            except:
                current_cadr_number = 0