import json
import logging

import nlp_ws
from simpletransformers.classification import ClassificationModel

log = logging.getLogger(__name__)


class XlmRobertaWorker(nlp_ws.NLPWorker):
    @classmethod
    def static_init(self, config):
        self.config = config
        log.debug("static_init(%s)", config)

    def init(self):
        log.debug("init()")
        models = dict()
        list_models = dict()
        for key in self.config["model"]:
            models[key] = json.loads(self.config["model"][key])
        for key, value in models.items():
            list_models[key] = ClassificationModel("xlmroberta",
                                                   value["file"],
                                                   num_labels=4,
                                                   use_cuda=False)
            print(value["file"])
        self._classifier = XlmRobertaClassifier(list_models)

    def process(self, input_path, task_options, output_path):
        task = task_options.get("type", None)
        with open(input_path, "r") as f:
            text = f.read()
        lang = text.split('__label__')[1]
        text = text.split('__label__')[0]
        result = self._classifier.predict(text, lang, task)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=4)


class XlmRobertaClassifier(object):

    def __init__(self, models):
        self.models = models
        self.labels_text = ["__label__meta_amb", "__label__meta_minus_m",
                            "__label__meta_plus_m", "___label__meta_zero"]
        self.labels_sen = ["__label__z_amb", "__label__z_minus_m",
                           "__label__z_plus_m", "___label__z_zero"]

    def predict(self, ccl, lang, task_options):
        if task_options == "sentence":
            task = "_sent_sen"
            labels = self.labels_sen
        else:
            task = "_sent"
            labels = self.labels_text
        model = self.models[lang + task]
        decision, raw = model.predict([ccl])
        print(raw)
        print(labels)
        result = dict(zip(labels, raw[0]))
        print(result)
        result['decision'] = labels[decision[0]]
        result['lang'] = lang
        return result
