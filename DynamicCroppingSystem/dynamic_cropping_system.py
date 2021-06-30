from tensorflow import keras
from keras.preprocessing import image
from tensorflow.keras.models import load_model

from PIL import Image
import numpy as np

import os




class DynamicCroppingSystem:
    def __init__(self):
        ImageDataGenerator = keras.preprocessing.image.ImageDataGenerator
        self.imgdatagen = ImageDataGenerator(
            rescale=1 / 255.,
            validation_split=0.2,
            horizontal_flip=True
        )

    def _create_dataset(self,
                        file_path,
                        batch_size=30,
                        img_size=(224, 224),
                        classes=('CENTER', 'LEFT','RIGHT')):
        """
        Private method to create training and validation sets for DynamicCroppingSystem()

        :param file_path: Path to the raw dataset
        :param batch_size: Number of samples processed before the model is updated. (Use default settings to save memory)
        :param img_size: Size of input image (Default settings recommended for RGB images)
        :param classes: Class names; Sub-folders in the raw dataset (Defaults to Center, Left and Right)
        """


        height, width = img_size
        self.train_dataset = self.imagedatagen.flow_from_directory(
            directory=file_path,
            target_size=(height, width),
            classes=classes,
            batch_size=batch_size,
            subset='training'
        )

        self.val_dataset = self.imagedatagen.flow_from_directory(
            directory=file_path,
            target_size=(height, width),
            classes=classes,
            batch_size=batch_size,
            subset='validation'
        )

    def create_model(self, path_to_trained_model):
        """
        Method to create and train the cropping model

        :param path_to_trained_model: Path to store the trained model
        :return: Saves the trained model in the desired location
        """
        model = keras.models.Sequential()

        initializers = {

        }
        model.add(
            keras.layers.Conv2D(
                24, 5, input_shape=(224, 224, 3),
                activation='relu',
            )
        )
        model.add(keras.layers.MaxPooling2D(2))
        model.add(
            keras.layers.Conv2D(
                48, 5, activation='relu',
            )
        )
        model.add(keras.layers.MaxPooling2D(2))
        model.add(
            keras.layers.Conv2D(
                96, 5, activation='relu',
            )
        )
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dropout(0.4))

        model.add(keras.layers.Dense(128, activation="relu"))
        model.add(keras.layers.Dense(
            3, activation='softmax',
        )
        )

        model.summary()

        model.compile(loss='categorical_crossentropy',
                      optimizer=keras.optimizers.Adamax(lr=0.001),
                      metrics=['acc'])

        history = model.fit_generator(
            self.train_dataset,
            validation_data=self.val_dataset,
            workers=10,
            epochs=10,
        )

        model.save(path_to_trained_model)

    def _evaluate_model_crop(self, img_fname, model):
        """
        Private function to classify an input image

        :param img_fname: Path to the image to be tested
        :param model: .model file
        :return: list; classification prediction
        """
        img = image.load_img(img_fname, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        return model.predict(x)

    def test_model_crop(self, path_to_model, path_to_test_data):
        """

        :param path_to_model: Path to the saved model
        :param path_to_test_data: Path to the test data
        :return: Cropped PIL image
        """
        full_model = load_model(path_to_model)
        x = (self._evaluate_model_crop(path_to_test_data, full_model))
        f, e = os.path.splitext(path_to_test_data)
        im = Image.open(path_to_test_data)
        if x[0][0] == 1.0:
            print("CENTRE")
            cropped_image = im.crop((400, 150, 1300, 1000))
        if x[0][1] == 1.0:
            cropped_image = im.crop((200, 250, 1050, 1000))
            print("LEFT")
        if x[0][2] == 1.0:
            print("RIGHT")
            cropped_image = im.crop((500, 300, 1400, 1100))
        return cropped_image



