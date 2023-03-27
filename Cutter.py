#              Imports
#         -----------------
import keras

#         -----------------

class Cutter():
    def __init__(self,path):
        self.model = None
        self.path_model = path
        print('INFO: Make sure, that you input exactly path to your seved model (path = /.../.../saved_model_folder) (.pb file)')
        self.model = keras.models.load_model(self.path_model)
        print('INFO: Model has been loaded. Use "Cutter.model.predict(...)" to use it.')

    def predict(self,frames,resize_koef):
        size = frames.shape[0]
        H = 540
        W = 1280
        if frames[0,0].shape == (H // resize_koef,W // resize_koef, 3):
            super_resize_koef = 10 // resize_koef
            return ((self.model.predict([frames[:, :, ::super_resize_koef, ::super_resize_koef][:, 0],frames[:, :, ::super_resize_koef, ::super_resize_koef][:, 1]])).reshape((size,)) * 100).astype('uint8')
        else:
            print('Input shape error')