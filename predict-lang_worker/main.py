import nlp_ws
from src.predict_lang import PredictLangWorker


if __name__ == '__main__':
    nlp_ws.NLPService.main(PredictLangWorker)
