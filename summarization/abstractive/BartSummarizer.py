import math
import torch
from nltk import sent_tokenize
from transformers import MBartTokenizer, MBartForConditionalGeneration


class BartSummarizer:
    def __init__(self):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model_name = "IlyaGusev/mbart_ru_sum_gazeta"
        self._tokenizer = MBartTokenizer.from_pretrained(self._model_name)
        self._model = MBartForConditionalGeneration.from_pretrained(self._model_name).to(self._device)

    def _summarize(self, text, max_length, min_length) -> str:
        tokenized_text = self._tokenizer.encode(text, return_tensors="pt").to(self._device)
        max_tokens = int(max_length * tokenized_text.size(1))
        min_tokens = int(min_length * tokenized_text.size(1))
        output_ids = self._model.generate(
            max_length=max_tokens,
            input_ids=tokenized_text,
            no_repeat_ngram_size=7,
            num_beams=12,
            early_stopping=True,
            length_penalty=0.5,
            min_length=min_tokens
        )[0]
        return self._tokenizer.decode(output_ids, skip_special_tokens=True)

    def _get_chunks(self, text, max_length) -> list:
        sentences = sent_tokenize(text)
        chunks = []
        current_sentence = ''
        for sentence in sentences:
            if len(current_sentence) + len(sentence) <= max_length:
                current_sentence += f' {sentence}' if current_sentence else sentence
            else:
                chunks.append(current_sentence)
                current_sentence = sentence
        if current_sentence:
            if len(current_sentence) <= 150:
                chunks[-1] = f'{chunks[-1]} {current_sentence}'
            else:
                chunks.append(current_sentence)
        return chunks

    def summarize_text(self, text, max_length=0.7, min_length=0.4) -> str:
        tokenized_text = self._tokenizer.encode(text, return_tensors="pt").to(self._device)
        if tokenized_text.size(1) > 1024:
            chunks = self._get_chunks(text, math.ceil(len(text) / math.ceil(tokenized_text.size(1) / 1024)))
            summarized_texts = []
            for chunk in chunks:
                summarized_texts.append(self._summarize(chunk, max_length, min_length))
            return ' '.join(summarized_texts)
        return self._summarize(text, max_length, min_length)
