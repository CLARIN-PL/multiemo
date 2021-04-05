import nlp_ws
from src.BiLSTM_LASER_worker import BiLstmLaserWorker


if __name__ == '__main__':
    nlp_ws.NLPService.main(BiLstmLaserWorker)
