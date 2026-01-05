import re
import nltk
from nltk.corpus import stopwords
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


nltk.download('stopwords')


def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\n\w\s]', '', text)
    text = text.lower()
    text = re.sub(r"n't", " not", text)
    text = ' '.join(text.split())
    return text


def remove_stopwords(text):
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    return ' '.join(words)


def load_cnn_dailymail_sample(split='train', fraction=0.001):
    dataset, info = tfds.load('cnn_dailymail', split=split, with_info=True)
    data = list(dataset)
    data = data[:int(fraction * len(data))]
    articles = [example['article'].numpy().decode('utf-8') for example in data]
    summaries = [example['highlights'].numpy().decode('utf-8') for example in data]
    return articles, summaries


def prepare_data(articles, summaries, max_seq_length_source=400, max_seq_length_target=100, test_size=0.2):
    cleaned_articles = [clean_text(a) for a in articles]
    cleaned_summaries = [clean_text(s) for s in summaries]
    cleaned_articles = [remove_stopwords(a) for a in cleaned_articles]
    cleaned_summaries = [remove_stopwords(s) for s in cleaned_summaries]

    tokenizer_source = Tokenizer()
    tokenizer_target = Tokenizer()
    tokenizer_source.fit_on_texts(cleaned_articles)
    tokenizer_target.fit_on_texts(cleaned_summaries)

    vocab_size_source = len(tokenizer_source.word_index) + 1
    vocab_size_target = len(tokenizer_target.word_index) + 1

    source_sequences = tokenizer_source.texts_to_sequences(cleaned_articles)
    target_sequences = tokenizer_target.texts_to_sequences(cleaned_summaries)

    source_sequences = pad_sequences(source_sequences, maxlen=max_seq_length_source, padding='post')
    target_sequences = pad_sequences(target_sequences, maxlen=max_seq_length_target, padding='post')

    X_train, X_test, y_train, y_test = train_test_split(source_sequences, target_sequences, test_size=test_size, random_state=42)

    return {
        'tokenizer_source': tokenizer_source,
        'tokenizer_target': tokenizer_target,
        'vocab_size_source': vocab_size_source,
        'vocab_size_target': vocab_size_target,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'max_seq_length_source': max_seq_length_source,
        'max_seq_length_target': max_seq_length_target,
    }
