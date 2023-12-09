from transformers import MBartTokenizer, MBartForConditionalGeneration
import textwrap
import torch


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
            no_repeat_ngram_size=4,
            num_beams=12,
            early_stopping=True,
            length_penalty=0.5,
            min_length=min_tokens
        )[0]

        summary = self._tokenizer.decode(output_ids, skip_special_tokens=True)
        return summary

    def summarize_text(self, text, chunk_size=500, max_length=0.7, min_length=0.4) -> str:
        summaries = []
        for chunk in textwrap.wrap(text, chunk_size):
            summary = self._summarize(chunk, max_length, min_length)
            summaries.append(summary)
        final_summary = ' '.join(summaries)
        return final_summary
