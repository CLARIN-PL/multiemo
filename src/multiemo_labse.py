import operator
import keras
import numpy as np
from keras import backend as K

from src.custom_labse import CustomLabse


def loss_coeff_determination(y_true, y_pred):
    ss_res = K.sum(K.square(y_true - y_pred))
    ss_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return ss_res / ss_tot


class MultiEmoLabse(object):
    def __init__(self, path_models='models'):
        self.models = dict()
        self.classes = []

        self.labels_sent = ["__label__meta_minus_m",
                            "__label__meta_plus_m",
                            "__label__meta_zero",
                            "__label__meta_amb"]
        self.labels_sent_sen = ["__label__z_zero",
                                "__label__z_plus_m",
                                "__label__z_minus_m",
                                "__label__z_amb"]

        self.models = dict()
        self.sent = dict()
        self.sent_sen = dict()

        self.sent = {"model": keras.models.load_model(
            path_models + '/sent.hdf5', custom_objects={
                "loss_coeff_determination": loss_coeff_determination}),
            "labels": self.labels_sent}

        self.sent_sen = {"model": keras.models.load_model(
            path_models + '/sent-sen.hdf5', custom_objects={
                "loss_coeff_determination": loss_coeff_determination}),
            "labels": self.labels_sent_sen}

        self.labse = CustomLabse()

    def predict(self, text, lang='pl', document_level='text'):
        list_of_embeddings = self.labse.embbed(text)
        sentence_count = len(list_of_embeddings)
        # print(sentence_count)
        # print(list_of_embeddings)
        embeddings = np.stack(list_of_embeddings, axis=0)
        # print(embeddings)
        # print(embeddings.shape)
        fixed_embeddings = embeddings.reshape(1, embeddings.shape[0], embeddings.shape[1])

        result = dict()
        if document_level == "sentence":
            model = self.sent_sen
        else:
            model = self.sent

        predict_result = next(iter(model['model'].predict(fixed_embeddings)))
        predictions = {
            model['labels'][ind]: score
            for ind, score in enumerate(predict_result)
        }

        new_predictions = dict()
        for prediction in predictions:
            if predictions[prediction] > 1:
                new_predictions[prediction] = str(1)
            elif predictions[prediction] < 0:
                new_predictions[prediction] = str(0)
            else:
                new_predictions[prediction] = str(round(
                    predictions[prediction], 3))

        predictions = new_predictions
        result['labels'] = predictions
        result['decision'] = max(predictions.items(),
                                 key=operator.itemgetter(1))[0]
        result['lang'] = lang
        return result
