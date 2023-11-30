import re
from collections import defaultdict
import nltk
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfSummarizer:
    @staticmethod  # Суммаризация текста
    def summarize_text(text) -> str:
        article_text = re.sub(r'\s+', ' ', text)
        sentences = nltk.sent_tokenize(article_text)
        preprocessed_sentences = [TfidfSummarizer._text_preprocessing(sentence) for sentence in sentences]
        weighted_word_frequency = TfidfSummarizer._get_weighted_word_frequency(preprocessed_sentences)
        sentences_scores = defaultdict(float)
        for preprocessed_sentence, orig_sentence in zip(preprocessed_sentences, sentences):
            for word in nltk.word_tokenize(preprocessed_sentence):
                if word in weighted_word_frequency.keys():
                    sentences_scores[orig_sentence] += weighted_word_frequency[word]
        average_score = sum(sentences_scores.values()) / len(sentences_scores)
        selected_sentences = []
        for sentence, score in sentences_scores.items():
            if score > average_score:
                selected_sentences.append(sentence)
        return ' '.join(selected_sentences)

    @staticmethod  # Предобработка текста
    def _text_preprocessing(text) -> str:
        lemmatizer = WordNetLemmatizer()
        stop_words = stopwords.words('russian')
        clear_text = re.sub(r'[^\w\s]', '', text.lower())
        clear_text = [word for word in nltk.word_tokenize(clear_text) if word not in stop_words]
        lemmatized_tokens = [lemmatizer.lemmatize(word) for word in clear_text]
        return ' '.join(lemmatized_tokens)

    @staticmethod  # Подсчёт частоты употребления для каждого слова
    def _get_weighted_word_frequency(sentences) -> dict:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(sentences)
        feature_names = vectorizer.get_feature_names_out()
        word_scores = defaultdict(float)
        for i, sentence in enumerate(sentences):
            feature_index = tfidf_matrix[i, :].nonzero()[1]
            for idx in feature_index:
                word = feature_names[idx]
                word_scores[word] += tfidf_matrix[i, idx]
        return word_scores

    @staticmethod  # Суммаризация текста
    def summarize_text2(text) -> str:
        article_text = re.sub(r'\s+', ' ', text)
        sentences = nltk.sent_tokenize(article_text)
        preprocessed_sentences = [TfidfSummarizer._text_preprocessing(sentence) for sentence in sentences]
        sentence_scores = TfidfSummarizer._get_sentence_scores(preprocessed_sentences)
        sentence_scores = {sentence: score for sentence, score in zip(sentences, sentence_scores)}
        average_score = sum(sentence_scores.values()) / len(sentence_scores)
        selected_sentences = []
        for sentence, score in sentence_scores.items():
            if score > average_score:
                selected_sentences.append(sentence)
        return ' '.join(selected_sentences)

    @staticmethod
    def _get_sentence_scores(sentences):
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(sentences)
        tfidf_values = tfidf_matrix.toarray()
        sentence_scores = tfidf_values.sum(axis=1)
        return sentence_scores
