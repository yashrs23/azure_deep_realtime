
from resnet152 import ResNet152
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from azureml.contrib.services.aml_request import rawhttp
from azureml.core.model import Model
from azureml.contrib.services.aml_response import AMLResponse
from toolz import compose
import numpy as np
import timeit as t
from PIL import Image, ImageOps
import logging

_NUMBER_RESULTS = 3


def _image_ref_to_pil_image(image_ref):
    """ Load image with PIL (RGB)
    """
    return Image.open(image_ref).convert("RGB")


def _pil_to_numpy(pil_image):
    img = ImageOps.fit(pil_image, (224, 224), Image.ANTIALIAS)
    img = image.img_to_array(img)
    return img


def _create_scoring_func():
    """ Initialize ResNet 152 Model
    """
    logger = logging.getLogger("model_driver")
    start = t.default_timer()
    model_name = "resnet_model"
    model_path = Model.get_model_path(model_name)
    model = ResNet152()
    model.load_weights(model_path)
    end = t.default_timer()

    loadTimeMsg = "Model loading time: {0} ms".format(round((end - start) * 1000, 2))
    logger.info(loadTimeMsg)

    def call_model(img_array_list):
        img_array = np.stack(img_array_list)
        img_array = preprocess_input(img_array)
        preds = model.predict(img_array)
        # Converting predictions to float64 since we are able to serialize float64 but not float32
        preds = decode_predictions(preds.astype(np.float64), top=_NUMBER_RESULTS)
        return preds

    return call_model


def get_model_api():
    logger = logging.getLogger("model_driver")
    scoring_func = _create_scoring_func()

    def process_and_score(images_dict):
        """ Classify the input using the loaded model
        """
        start = t.default_timer()
        logger.info("Scoring {} images".format(len(images_dict)))
        transform_input = compose(_pil_to_numpy, _image_ref_to_pil_image)
        transformed_dict = {
            key: transform_input(img_ref) for key, img_ref in images_dict.items()
        }
        preds = scoring_func(list(transformed_dict.values()))
        preds = dict(zip(transformed_dict.keys(), preds))
        end = t.default_timer()

        logger.info("Predictions: {0}".format(preds))
        logger.info("Predictions took {0} ms".format(round((end - start) * 1000, 2)))
        return (preds, "Computed in {0} ms".format(round((end - start) * 1000, 2)))

    return process_and_score


def init():
    """ Initialise the model and scoring function
    """
    global process_and_score
    process_and_score = get_model_api()


@rawhttp
def run(request):
    """ Make a prediction based on the data passed in using the preloaded model
    """
    if request.method == 'POST':
        return process_and_score(request.files)
    if request.method == 'GET':
        resp_body = {
            "azEnvironment": "Azure",
            "location": "westus2",
            "osType": "Ubuntu 16.04",
            "resourceGroupName": "",
            "resourceId": "",
            "sku": "",
            "subscriptionId": "",
            "uniqueId": "PythonMLRST",
            "vmSize": "",
            "zone": "",
            "isServer": False,
            "version": ""
        }
        return resp_body
    return AMLResponse("bad request", 500)
