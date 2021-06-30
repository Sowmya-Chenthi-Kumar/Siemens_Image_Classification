from DynamicCroppingSystem.dynamic_cropping_system import DynamicCroppingSystem
from ImagePreprocessing.image_preprocessing import ImagePreprocessor

from tensorflow.keras.models import load_model

import os

import pandas as pd

dynamic_cropping_system = DynamicCroppingSystem()
image_preprocessing_system = ImagePreprocessor()


def run_classification_pipeline(path_to_test_data,
                                path_to_cropping_model,
                                path_to_classification_model,
                                assembly_line,
                                to_csv=False):
    """

    :param path_to_test_data: Path to test data
    :param path_to_cropping_model: Path to cropping model (.model file)
    :param path_to_classification_model: Path to classification model (.model file)
    :param assembly_line: Select 2 for Assembly Line 2; 4 for Assembly line 4
    :param to_csv: Save predicted results to a csv file
    :return: predicted results in the following format for binary case
    Anomaly: [1,0,0]
    Normal: [0,1,0]
    Edge Case: [0,0,1]
    """
    classification_item = []
    classification_list = []

    classification_model = load_model(path_to_classification_model)
    directory = os.listdir(path_to_test_data)
    for item in directory:
        if item == '.DS_Store':
            continue

        if os.path.isfile(path_to_test_data + item):
            image = dynamic_cropping_system.test_model_crop(path_to_cropping_model,
                                                            path_to_test_data + item)
            x = image_preprocessing_system.process_image(image=image,
                                                         assembly_line=assembly_line)
            prediction = classification_model.predict([x])[0]
            print(item)
            print(prediction)

            classification_item.append(item)
            classification_list.append(prediction)

    if to_csv:
        classification_tuple = list(zip(classification_item,
                                        classification_list))
        classification_data_frame = pd.DataFrame(classification_tuple,
                                                 columns=['Item', 'Classification'])

        classification_data_frame.to_csv('predicted_results.csv')
        print('predicted_results.csv created at {}'.format(os.getcwd()))

if __name__ == '__main__':
    path_to_test_data = '/Users/adisal/Desktop/BiosensorClassificationSH/TestImage/TestImage_L2/'
    path_to_cropping_model = '/Users/adisal/Desktop/BiosensorClassificationSH/ModelArchive/Model_values_final'
    path_to_classification_model = '/Users/adisal/Desktop/BiosensorClassificationSH/ModelArchive/ClassificationModel_SH_binary_wEC.model'

    run_classification_pipeline(path_to_test_data,
                                path_to_cropping_model,
                                path_to_classification_model,
                                assembly_line=2,
                                to_csv=False)
