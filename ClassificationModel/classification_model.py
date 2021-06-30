from tensorflow import keras
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import load_model

import os

class ClassificationModel:
    def __init__(self):
        ImageDataGenerator = keras.preprocessing.image.ImageDataGenerator
        self.imagedatagen = ImageDataGenerator(
            rescale=1 / 255.,
            horizontal_flip=True,
            vertical_flip=True,
            rotation_range=270,
            validation_split=0.2)

    def _create_dataset(self, file_path, batch_size=30, img_size=(224, 224), classes=('Normal', 'Anomaly')):
        """
        Private method to create training and validation sets

        :param file_path: Path to the training dataset
        :param batch_size: Number of samples processed before the model is updated. (Use default settings to save memory)
        :param img_size: Size of input image (Default settings recommended for RGB images)
        :param classes: Classification classes (Defaults to Normal and Anomaly)
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

    def create_model(self, path_to_dataset, path_to_trained_model, model_type='resnet', batch_size=30,
                     img_size=(224, 224),
                     classes=('Normal', 'Anomaly'),
                     no_classes=2,
                     epochs=20):

        """

        :param path_to_dataset: Path to training dataset
        :param path_to_trained_model: Path to store the trained model
        :param model_type: Type of transfer learning algorithm; Available options resnet, vgg16, vgg19, inception (Defaults to resent)
        :param batch_size: Number of samples processed before the model is updated. (Use default settings to save memory)
        :param img_size: Size of input image (Default to 224x224 for transfer learning algorithms)
        :param classes: Class labels (Defaults to Normal and Anomaly)
        :param no_classes: Number of classes (Defaults to 2; binary classification)
        :param epochs: Number of epochs
        :return: Returns .model file
        """

        self._create_dataset(path_to_dataset, batch_size=batch_size, img_size=img_size, classes=classes)
        if model_type == 'resnet':
            conv_model = ResNet50(weights=None,
                                  include_top=False,
                                  input_shape=(224, 224, 3),
                                  classes=no_classes)
        elif model_type == 'vgg16':
            conv_model = VGG16(weights=None,
                               include_top=False,
                               input_shape=(224, 224, 3),
                               classes=no_classes)
        elif model_type == 'vgg19':
            conv_model = VGG19(weights=None,
                               include_top=False,
                               input_shape=(224, 224, 3),
                               classes=no_classes)
        elif model_type == 'inception':
            conv_model = InceptionV3(weights=None,
                                     include_top=False,
                                     input_shape=(224, 224, 3),
                                     classes=no_classes)
        for layer in conv_model.layers:
            layer.trainable = False
        x = keras.layers.Flatten()(conv_model.output)
        x = keras.layers.Dense(100, activation='relu')(x)
        x = keras.layers.Dense(100, activation='relu')(x)
        x = keras.layers.Dense(100, activation='relu')(x)
        x = keras.layers.Dropout(0.2)(x)
        predictions = keras.layers.Dense(no_classes, activation='softmax')(x)
        full_model = keras.models.Model(inputs=conv_model.input, outputs=predictions)
        full_model.summary()

        full_model.compile(loss='binary_crossentropy',
                           optimizer=keras.optimizers.Adamax(lr=0.001),
                           metrics=['acc'])

        self.history = full_model.fit_generator(
            self.train_dataset,
            validation_data=self.val_dataset,
            workers=10,
            epochs=epochs,
        )

        full_model.save(path_to_trained_model)

        return full_model

    def _evaluate_model(self, img_fname, model):
        """
        Private function to classify an input image

        :param img_fname: Path to the image to be tested
        :param model: .model file
        :return: classification prediction
        """
        img = image.load_img(img_fname, target_size=(224, 224))
        x = image.img_to_array(img)
        x = x.reshape(-1, 224, 224, 3)
        print(model.predict([x])[0])

    def test_model(self, path_to_model, path_to_test_data):
        """
        Function to classify all images in a directory

        :param path_to_model: Path to saved model
        :param path_to_test_data: Path to test dataset
        :return: prints classification prediction
        """
        full_model = load_model(path_to_model)
        directory = os.listdir(path_to_test_data)
        for item in directory:
            if item == '.DS_Store':
                continue
            if os.path.isfile(path_to_test_data + item):
                self._evaluate_model(path_to_test_data + item, full_model)