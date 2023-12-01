from transformers import MBartTokenizer, MBartForConditionalGeneration
import textwrap


class BartSummarizer:
    def __init__(self):
        self._model_name = "IlyaGusev/mbart_ru_sum_gazeta"
        self._tokenizer = MBartTokenizer.from_pretrained(self._model_name)
        self._model = MBartForConditionalGeneration.from_pretrained(self._model_name)

    def _summarize(self, text, max_length):
        input_ids = self._tokenizer(
            [text],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=len(text)
        )["input_ids"]

        output_ids = self._model.generate(
            max_length=max_length,
            input_ids=input_ids,
            no_repeat_ngram_size=3,
            num_beams=8,
            early_stopping=True
        )[0]

        summary = self._tokenizer.decode(output_ids, skip_special_tokens=True)
        return summary

    def summarize_text(self, text, chunk_size=600):
        summaries = []
        for chunk in textwrap.wrap(text, chunk_size):
            # Обработка каждой части текста
            summary = self._summarize(chunk, len(chunk))
            summaries.append(summary)
        final_summary = ' '.join(summaries)
        return final_summary
