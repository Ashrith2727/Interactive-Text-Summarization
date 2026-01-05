import tensorflow as tf
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def build_seq_model(vocab_size_source, vocab_size_target, max_seq_length_source):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(input_dim=vocab_size_source, output_dim=128, input_length=max_seq_length_source))
    model.add(tf.keras.layers.LSTM(128, return_sequences=True))
    model.add(tf.keras.layers.Dense(vocab_size_target, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def train_model(model, X_train, y_train, X_test, y_test, epochs=5, batch_size=32):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
    return model


def generate_summary(model, tokenizer_source, tokenizer_target, input_text, max_seq_length_source=400, max_seq_length_target=100):
    input_seq = tokenizer_source.texts_to_sequences([input_text])[0]
    input_seq = tf.keras.preprocessing.sequence.pad_sequences([input_seq], maxlen=max_seq_length_source, padding='post')

    decoded_summary = []
    for _ in range(max_seq_length_target):
        output_tokens = model.predict(input_seq)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = tokenizer_target.index_word.get(sampled_token_index, '')
        if not sampled_word or sampled_word == 'eos':
            break
        decoded_summary.append(sampled_word)
    generated_summary = ' '.join(decoded_summary)
    return generated_summary


def compute_bleu(reference, candidate):
    reference = reference.split()
    candidate = candidate.split()
    smooth = SmoothingFunction().method4
    score = sentence_bleu([reference], candidate, smoothing_function=smooth)
    return score


def evaluate_on_testset(model, X_test, y_test, tokenizer_source, tokenizer_target):
    for i in range(len(X_test)):
        input_text = ' '.join([tokenizer_source.index_word[idx] for idx in X_test[i] if idx != 0])
        generated_summary = generate_summary(model, tokenizer_source, tokenizer_target, input_text)
        target_summary = ' '.join([tokenizer_target.index_word[idx] for idx in y_test[i] if idx != 0])
        bleu_score = compute_bleu(target_summary, generated_summary)
        print(f"Input Text: {input_text}")
        print(f"Generated Summary: {generated_summary}")
        print(f"Target Summary: {target_summary}")
        print(f"BLEU Score: {bleu_score}")
        print("-" * 50)
