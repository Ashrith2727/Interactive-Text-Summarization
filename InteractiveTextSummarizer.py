import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
import re
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

dataset, info = tfds.load('cnn_dailymail', split='train', with_info=True)

# Access and preprocess the dataset
data = list(dataset)
data = data[:int(0.001 * len(data))]
articles = [example['article'].numpy().decode('utf-8') for example in data]
summaries = [example['highlights'].numpy().decode('utf-8') for example in data]


def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    text = re.sub(r"n't", " not", text)
    text = ' '.join(text.split())
    return text

def remove_stopwords(text):
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    return ' '.join(words)


cleaned_articles = [clean_text(article) for article in articles]
cleaned_summaries = [clean_text(summary) for summary in summaries]
cleaned_articles = [remove_stopwords(article) for article in cleaned_articles]
cleaned_summaries = [remove_stopwords(summary) for summary in cleaned_summaries]

# Tokenization and padding
tokenizer_source = Tokenizer()
tokenizer_target = Tokenizer()
tokenizer_source.fit_on_texts(cleaned_articles)
tokenizer_target.fit_on_texts(cleaned_summaries)

vocab_size_source = len(tokenizer_source.word_index) + 1
vocab_size_target = len(tokenizer_target.word_index) + 1

source_sequences = tokenizer_source.texts_to_sequences(cleaned_articles)
target_sequences = tokenizer_target.texts_to_sequences(cleaned_summaries)



max_seq_length_source = 400  # Adjust as needed
max_seq_length_target = 100  # Adjust as needed
source_sequences = pad_sequences(source_sequences, maxlen=max_seq_length_source, padding='post')
target_sequences = pad_sequences(target_sequences, maxlen=max_seq_length_target, padding='post')

# Updated max sequence lengths

max_seq_length = 400  # Choose a common max sequence length

# Adjust your preprocessing to use the same max sequence length
source_sequences = pad_sequences(source_sequences, maxlen=max_seq_length, padding='post')
target_sequences = pad_sequences(target_sequences, maxlen=max_seq_length, padding='post')


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(source_sequences, target_sequences, test_size=0.2, random_state=42)


model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(input_dim=vocab_size_source, output_dim=128, input_length=max_seq_length_source))
model.add(tf.keras.layers.LSTM(128, return_sequences=True))
model.add(tf.keras.layers.Dense(vocab_size_target, activation='softmax'))



# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=17, batch_size=32, validation_data=(X_test, y_test))


# Function to generate summaries
def generate_summary(input_text):
    input_seq = tokenizer_source.texts_to_sequences([input_text])[0]
    input_seq = pad_sequences([input_seq], maxlen=max_seq_length_source, padding='post')
    target_seq = np.zeros((1, 1))

    decoded_summary = []
    stop_condition = False
    while not stop_condition:
        output_tokens = model.predict([input_seq, target_seq])
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = tokenizer_target.index_word[sampled_token_index]
        decoded_summary.append(sampled_word)

        if sampled_word == 'eos' or len(decoded_summary) > max_seq_length_target:
            stop_condition = True

        target_seq[0, 0] = sampled_token_index

    generated_summary = ' '.join(decoded_summary)
    return generated_summary


def compute_bleu(reference, candidate):
    reference = reference.split()
    candidate = candidate.split()
    smooth = SmoothingFunction().method4
    score = sentence_bleu([reference], candidate, smoothing_function=smooth)
    return score


# Evaluate the model on test data
for i in range(len(X_test)):
    input_text = ' '.join([tokenizer_source.index_word[idx] for idx in X_test[i] if idx != 0])
    generated_summary = generate_summary(input_text)
    target_summary = ' '.join([tokenizer_target.index_word[idx] for idx in y_test[i] if idx != 0])
    bleu_score = compute_bleu(target_summary, generated_summary)
    print(f"Input Text: {input_text}")
    print(f"Generated Summary: {generated_summary}")
    print(f"Target Summary: {target_summary}")
    print(f"BLEU Score: {bleu_score}")
    print("-" * 50)
