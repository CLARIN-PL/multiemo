from mosestokenizer import MosesSentenceSplitter
import torch
from transformers import AutoTokenizer, AutoModel

from src.language_identification import LanguageIdentification


class CustomLabse:
    def __init__(self):
        self.language_identification = LanguageIdentification()
        self.labse_tokenizer = AutoTokenizer.from_pretrained("pvl/labse_bert", do_lower_case=False)
        self.labse_model = AutoModel.from_pretrained("pvl/labse_bert")

    def embbed(self, text):
        lang = self.language_identification.predict_lang(text, None)

        with MosesSentenceSplitter(lang) as split_sents:
            text = split_sents([text])
            print(text)
            embeddings = []
            for sentence in text:
                embeddings.append(custom_labse(sentence, self.labse_tokenizer, self.labse_model).detach().numpy())
            return embeddings


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def custom_labse(text, tokenizer, model):
    encoded_input = tokenizer(text, padding=True, truncation=True, max_length=64, return_tensors='pt')
    encoded_input = encoded_input.to('cpu')

    with torch.no_grad():
        model_output = model(**encoded_input)

    return mean_pooling(model_output, encoded_input['attention_mask'])[0]
