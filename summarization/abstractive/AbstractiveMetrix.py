from collections import defaultdict

import pandas as pd
from datasets import load_dataset
from nltk.tokenize import word_tokenize
from nltk.translate import meteor_score
from rouge import Rouge


def abstractive_metrix(model, max_length, min_length, n_rows=15):
    dataset = load_dataset('IlyaGusev/gazeta', revision="v2.0")
    df_test = pd.DataFrame(dataset['test'], columns=['text', 'summary'])
    results = []
    for idx, (text, orig_summary) in enumerate(zip(df_test['text'][:n_rows], df_test['summary'][:n_rows])):
        print(f'Начал текст № {idx}')
        summarize = model.summarize_text(text, max_length=max_length, min_length=min_length, chunk_size=1000)
        results.append((summarize, orig_summary))
        print(f'Закончил текст № {idx}\n')
    rouge = Rouge()
    rouges = defaultdict(list)
    meteors = []
    for (summary, orig_summary) in results:
        tokenize_summary = word_tokenize(summary)
        tokenize_orig_summary = word_tokenize(orig_summary)
        meteors.append(meteor_score.meteor_score([tokenize_orig_summary], tokenize_summary))
        scores = rouge.get_scores(summary, orig_summary)[0]
        for metric, value in scores.items():
            rouges[metric].append(value['f'])
    print(f'METEOR: {sum(meteors) / len(meteors)}')
    for metric, values in rouges.items():
        print(f'{metric}: {sum(values) / len(values)}')
