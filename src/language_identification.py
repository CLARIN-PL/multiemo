import fasttext


class LanguageIdentification(object):

    def __init__(self):
        self.model = fasttext.load_model("models/lid.176.bin")
        self.language = ['es', 'it', 'zh', 'ja', 'ru', 'de', 'fr']

    def predict_lang(self, ccl, language):
        ccl = ccl.replace('\n', '')
        if language in self.language:
            predict = language
        else:
            predict = self.model.predict(ccl)[0][0]
        return predict.split('__label__')[-1]
