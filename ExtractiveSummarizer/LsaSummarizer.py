import math

import numpy as np
from nltk import word_tokenize, WordNetLemmatizer, sent_tokenize
from nltk.corpus import stopwords
from numpy.linalg import svd
import re


class LsaSummarizer:
    @staticmethod
    def _text_preprocessing(text) -> list:
        lemmatizer = WordNetLemmatizer()
        stop_words = stopwords.words('russian')
        clear_text = re.sub(r'[^\w\s]', '', text.lower())
        clear_text = [word for word in word_tokenize(clear_text) if word not in stop_words]
        lemmatized_tokens = [lemmatizer.lemmatize(word) for word in clear_text]
        return lemmatized_tokens

    @staticmethod
    def _create_matrix(sentences, words) -> np.ndarray:
        words_count = len(words)
        sentences_count = len(sentences)
        matrix = np.zeros((words_count, sentences_count))
        for column, sentence in enumerate(sentences):
            for word in sentence:
                row = words[word]
                matrix[row, column] += 1
        return matrix

    @staticmethod
    def _normalization_matrix(matrix) -> np.ndarray:
        max_word_frequencies = np.max(matrix, axis=0)
        rows, columns = matrix.shape
        for row in range(rows):
            for column in range(columns):
                max_word_frequency = max_word_frequencies[column]
                if max_word_frequency != 0:
                    matrix[row, column] /= max_word_frequency
        return matrix

    @staticmethod
    def summarize_text(text, sentences_count=3):
        orig_sentences = [sentence for sentence in sent_tokenize(text)]
        preprocessed_sentences = [LsaSummarizer._text_preprocessing(sentence) for sentence in orig_sentences]
        unique_words = set()
        for sentence_words in preprocessed_sentences:
            unique_words.update(sentence_words)
        word_index = {word: idx for idx, word in enumerate(unique_words)}
        matrix = LsaSummarizer._create_matrix(preprocessed_sentences, word_index)
        matrix = LsaSummarizer._normalization_matrix(matrix)
        _, sigma, v_matrix = svd(matrix, full_matrices=False)
        min_length = max(3, sentences_count)
        weight_topics = [s ** 2 for s in sigma[:min_length]]
        scores = []
        for sentence_column in v_matrix.T:
            sentence_column = sentence_column[:min_length]
            score = math.sqrt(sum(wt * sc ** 2 for wt, sc in zip(weight_topics, sentence_column)))
            scores.append(score)
        ordered_indices_score = [idx for idx in range(len(orig_sentences))]
        ordered_indices_score = [idx for _, idx in sorted(zip(scores, ordered_indices_score), reverse=True)]
        ordered_indices_score = ordered_indices_score[:sentences_count]
        return " ".join([orig_sentences[idx] for idx in ordered_indices_score])

    @staticmethod
    def summarize_text_average(text):
        orig_sentences = [sentence for sentence in sent_tokenize(text)]
        preprocessed_sentences = [LsaSummarizer._text_preprocessing(sentence) for sentence in orig_sentences]
        unique_words = set()
        for sentence_words in preprocessed_sentences:
            unique_words.update(sentence_words)
        word_index = {word: idx for idx, word in enumerate(unique_words)}
        matrix = LsaSummarizer._create_matrix(preprocessed_sentences, word_index)
        matrix = LsaSummarizer._normalization_matrix(matrix)
        _, sigma, v_matrix = svd(matrix, full_matrices=False)
        weight_topics = [s ** 2 for s in sigma[:len(orig_sentences)]]
        scores = []
        for sentence_column in v_matrix.T:
            score = math.sqrt(sum(wt * sc ** 2 for wt, sc in zip(weight_topics, sentence_column)))
            scores.append(score)
        average_score = sum(scores) / len(scores)
        ordered_indices_score = [idx for idx, score in enumerate(scores) if score > average_score]
        return " ".join([orig_sentences[idx] for idx in ordered_indices_score])
