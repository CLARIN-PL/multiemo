import fasttext
import nlp_ws


class PredictLangWorker(nlp_ws.NLPWorker):

    def __init__(self):
        self.model = fasttext.load_model("lid.176.bin")

    def process(self, input_file: str,
                task_options: dict,
                output_file: str) -> None:
        with open(input_file, "r") as f:
            text = f.read().replace('\n', ' ')
        print(text)
        predict = self.model.predict(text)[0][0]
        with open(output_file, "w") as f:
            f.write(text + predict)
        print(predict)
        return predict
