import argparse
from preprocessing import load_cnn_dailymail_sample, prepare_data
from seq_model import build_seq_model, train_model, evaluate_on_testset

def summarize_with_transformer(text, model_name="facebook/bart-large-cnn", max_length=150, min_length=40):
    # Lazy import to avoid requiring transformers when training the simple seq model
    try:
        from transformers import pipeline
    except Exception:
        raise RuntimeError("transformers library is not installed; install via requirements.txt")
    summarizer = pipeline("summarization", model=model_name)
    summary = summarizer(text, max_length=max_length, min_length=min_length, truncation=True)
    return summary[0]["summary_text"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Download sample data, build and train the simple seq model")
    parser.add_argument("--interactive", action="store_true", help="Run interactive transformer summarizer")
    parser.add_argument("--model", default="facebook/bart-large-cnn", help="Pretrained model name for summarization.")
    args = parser.parse_args()

    if args.train:
        print("Loading sample data...")
        articles, summaries = load_cnn_dailymail_sample()
        data = prepare_data(articles, summaries)
        print("Building model...")
        model = build_seq_model(data['vocab_size_source'], data['vocab_size_target'], data['max_seq_length_source'])
        print("Training model (this may take a while)...")
        model = train_model(model, data['X_train'], data['y_train'], data['X_test'], data['y_test'], epochs=2)
        print("Evaluating on test set...")
        evaluate_on_testset(model, data['X_test'], data['y_test'], data['tokenizer_source'], data['tokenizer_target'])
    elif args.interactive:
        print("Interactive summarizer (type 'quit' to exit)")
        try:
            while True:
                txt = input("Enter text> ").strip()
                if not txt or txt.lower() in ("quit", "exit"):
                    break
                try:
                    summary = summarize_with_transformer(txt, model_name=args.model)
                    print("Summary:\n", summary)
                except RuntimeError as e:
                    print(e)
                    break
        except (KeyboardInterrupt, EOFError):
            pass
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
