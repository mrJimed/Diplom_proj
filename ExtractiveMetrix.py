from rouge import Rouge
import pandas as pd
from datasets import load_dataset
from ExtractiveSummarizer.TfidfSummarizer import TfidfSummarizer
from collections import defaultdict
from prettytable import PrettyTable


def create_table(n_rows=15) -> PrettyTable:
    dataset = load_dataset('IlyaGusev/gazeta', revision="v2.0")
    df_test = pd.DataFrame(dataset['test'], columns=['text', 'summary'])
    rouge = Rouge()
    rouges = defaultdict(list)
    for idx, (text, orig_summary) in enumerate(zip(df_test['text'][:n_rows], df_test['summary'][:n_rows])):
        summarize = TfidfSummarizer.summarize_text(text)
        scores = rouge.get_scores(summarize, orig_summary)
        for metric, value in scores.items():
            rouges[metric].append(value['f'])
    table = PrettyTable()
    table.field_names = ['', 'TF-IDF']
    for metric, values in rouges.items():
        avg_score = sum(values) / len(values)
        table.add_row([[metric, avg_score]])
    return table
