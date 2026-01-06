# 📝 Interactive Text Summarizer

An **Interactive Text Summarizer** built using **Python and Natural Language Processing (NLP)** that allows users to input text and receive a concise summary.  
The project supports **interactive mode** via the command line and can be extended for model-based summarization.

---

## 🚀 Features

- ✍️ Accepts user input text interactively  
- 📄 Generates a concise summary from long text  
- 🧠 Uses NLP techniques such as tokenization and stopword removal  
- ⚡ Fast and lightweight  
- 🔁 Supports training and inference modes  
- 🧪 Easy to test and extend  

---

## 🛠️ Technologies Used

- Python 3
- TensorFlow / Keras (for model support)
- NLTK (Natural Language Toolkit)
- NumPy
- argparse (for CLI interaction)

---

## 📂 Project Structure

```

interactive-text-summarizer/
├── run.py
├── model/
│   └── summarizer_model.h5
├── data/
│   └── sample_text.txt
├── requirements.txt
└── README.md

````

---

## ⚙️ Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/interactive-text-summarizer.git
cd interactive-text-summarizer
````

2. Create and activate virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Download NLTK resources (first run only):

```python
import nltk
nltk.download('stopwords')
```

---

## ▶️ Usage

### 🔹 Interactive Mode

```bash
python run.py --interactive
```

### 🔹 Train the Model

```bash
python run.py --train
```

### 🔹 Use a Specific Model

```bash
python run.py --model model/summarizer_model.h5
```

### 🔹 Help Menu

```bash
python run.py --help
```

---

