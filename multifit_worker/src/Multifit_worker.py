import json
import logging
import operator

import nlp_ws
from fastai.text import *
import torch

log = logging.getLogger(__name__)


class MultifitWorker(nlp_ws.NLPWorker):
    @classmethod
    def static_init(self, config):
        self.config = config
        log.debug("static_init(%s)", config)

    def init(self):
        log.debug("init()")
        self._classifier = MultifitClassifier()

    def process(self, input_path, task_options, output_path):
        task = task_options.get("type", None)
        with open(input_path, "r") as f:
            text = f.read()
        lang = text.split('__label__')[1]
        text = text.split('__label__')[0]
        result = self._classifier.predict(text, lang=lang, task_options=task)
        result["decision"] = max(result.items(), key=operator.itemgetter(1))[0]
        result["language"] = lang
        print(result)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=4)


class MultifitClassifier(object):

    def __init__(self):
        self.labels_text = ["__label__meta_amb", "__label__meta_minus_m",
                            "__label__meta_plus_m", "___label__meta_zero"]
        self.labels_sen = ["__label__z_amb", "__label__z_minus_m",
                           "__label__z_plus_m", "___label__z_zero"]

    def predict(self, ccl, lang=None, task_options=None):
        path = ""
        if task_options == "sentence":
            path = lang + "-sent-sen.pkl"
            labels = self.labels_sen
        else:
            path = lang + "-sent.pkl"
            labels = self.labels_text
        learner = load_learner("models", path)
        results = learner.predict("xxbos " + str(ccl))
        probabilities = [str(x) for x in to_np(results[2])]
        result = dict(zip(labels, probabilities))
        return result
