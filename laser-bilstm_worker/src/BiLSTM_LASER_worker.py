import json
import operator
import logging

import keras
import numpy as np
from nltk.tokenize import sent_tokenize
from laserembeddings import Laser
from keras import backend as K
import nlp_ws

log = logging.getLogger(__name__)


class BiLstmLaserWorker(nlp_ws.NLPWorker):
    @classmethod
    def static_init(self, config):
        self.config = config
        log.debug("static_init(%s)", config)

    def init(self):
        log.debug("init()")
        models = dict()
        for key in self.config["model"]:
            models[key] = json.loads(self.config["model"][key])
        self._classifier = BiLstmClassifier(models)

    def process(self, input_path, task_options, output_path):
        task = task_options.get("model", None)
        print("----MODEL-----")
        print(task)
        with open(input_path, "r") as f:
            text = f.read()
        lang = text.split('__label__')[1]
        text = text.split('__label__')[0]
        result = self._classifier.predict(text, lang=lang, task_options=task)
        print(result)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=4)


def loss_coeff_determination(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return SS_res / (SS_tot)


class BiLstmClassifier(object):

    def __init__(self, model_settings):
        self.models = dict()
        self.classes = []
        for key, value in model_settings.items():
            self.models[key] = {"model": keras.models.load_model(
                                value['file'], custom_objects={
                                    "loss_coeff_determination":
                                    loss_coeff_determination}),
                                "labels": value["labels"]}
        self.laser = Laser()

    def predict(self, ccl, models=None,
                task_options=None, k=1, threshold=0.0, lang=None):
        options = dict()
        if task_options is None:
            options = self.models
        else:
            for key in task_options:
                options[key] = self.models[key]

        language_short_codes = {'pl': 'polish', 'en': 'english',
                                'cs': 'czech', 'da': 'danish',
                                'nl': 'dutch', 'et': 'estonian',
                                'fi': 'finnish', 'fr': 'french',
                                'de': 'german', 'el': 'greek',
                                'it': 'italian', 'no': 'norwegian',
                                'pt': 'portuguese', 'ru': 'russian',
                                'sl': 'slovenian', 'es': 'spanish',
                                'sv': 'swedish', 'tu': 'turkish'}

        input_data = sent_tokenize(ccl, language=language_short_codes[lang])
        line_length = len(input_data)
        vector_dimension = 1024
        line_vector = np.zeros((1, line_length, vector_dimension))
        print("Detect language:")
        print(lang)
        embeddings = self.laser.embed_sentences(input_data, lang=lang)

        for sentence_i in range(line_length):
            line_vector[0][sentence_i] = embeddings[sentence_i]
        query_result = line_vector
        result = dict()
        for key, value in options.items():
            predict_result = next(iter(value["model"].predict(query_result)))
            predictions = {
                value["labels"][ind]: score
                for ind, score in enumerate(predict_result)
            }
            new_predictions = dict()
            for prediction in predictions:
                if predictions[prediction] > 1:
                    new_predictions[prediction] = str(1)
                elif predictions[prediction] < 0:
                    new_predictions[prediction] = str(0)
                else:
                    new_predictions[prediction] = str(predictions[prediction])
            predictions = new_predictions
            result[key] = max(predictions.items(),
                              key=operator.itemgetter(1))[0]
        return result
