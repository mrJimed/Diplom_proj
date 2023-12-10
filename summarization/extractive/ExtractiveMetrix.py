from collections import defaultdict

import pandas as pd
from datasets import load_dataset
from nltk import word_tokenize
from nltk.translate import meteor_score
from rouge import Rouge


def extractive_metrix(summarize_method, sentences_count, n_rows=15):
    dataset = load_dataset('IlyaGusev/gazeta', revision="v2.0")
    df_test = pd.DataFrame(dataset['test'], columns=['text', 'summary'])
    rouge = Rouge()
    rouges = defaultdict(list)
    meteors = []
    for idx, (text, orig_summary) in enumerate(zip(df_test['text'][:n_rows], df_test['summary'][:n_rows])):
        print(f'Начал текст № {idx}')
        summary = summarize_method(text, sentences_count=sentences_count)
        tokenize_summary = word_tokenize(summary)
        tokenize_orig_summary = word_tokenize(orig_summary)
        meteors.append(meteor_score.meteor_score([tokenize_orig_summary], tokenize_summary))
        scores = rouge.get_scores(summary, orig_summary)[0]
        for metric, value in scores.items():
            rouges[metric].append(value['f'])
        print(f'Закончил текст № {idx}\n')
    print(f'METEOR: {sum(meteors) / len(meteors)}')
    for metric, values in rouges.items():
        print(f'{metric}: {sum(values) / len(values)}')
