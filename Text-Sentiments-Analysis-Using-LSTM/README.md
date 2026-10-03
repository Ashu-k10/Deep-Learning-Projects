# 🎬 Movie Review Sentiment Analysis using LSTM

A deep learning project that uses an **LSTM (Long Short-Term Memory) neural network** to perform binary sentiment analysis on the **IMDB Movie Reviews dataset**.

The model takes encoded movie reviews, converts them into fixed-length sequences using padding, processes the sequences through an **Embedding layer and LSTM layer**, and predicts whether each review is **Positive** or **Negative**.

## 📌 Project Overview

This project demonstrates an end-to-end NLP/deep learning workflow:

```text
IMDB Movie Reviews
        ↓
Load Dataset
        ↓
Word Index / Dictionary
        ↓
Decode Sample Reviews
        ↓
Sequence Padding
        ↓
Embedding Layer
        ↓
LSTM Layer
        ↓
Dense + Sigmoid
        ↓
Positive / Negative
```

The source code uses the TensorFlow/Keras IMDB dataset, considers the **10,000 most frequent words**, and pads reviews to a maximum length of **200 words**. fileciteturn0file0L14-L27

---

## 🎯 Objective

The main objective is to build an LSTM-based sentiment classifier capable of:

- Loading the IMDB movie review dataset
- Understanding the encoded review representation
- Decoding encoded reviews back into readable text
- Preparing variable-length reviews using padding
- Learning word representations using an Embedding layer
- Learning sequential dependencies using LSTM
- Predicting positive or negative sentiment
- Comparing the predicted sentiment with the actual sentiment

The dataset labels are:

```text
0 → Negative
1 → Positive
```

as defined in the source code. fileciteturn0file0L34-L42

---

## 🧠 Model Architecture

The neural network follows this architecture:

```text
Input Review
     ↓
Embedding
     ↓
LSTM
     ↓
Dense
     ↓
Sigmoid
     ↓
Positive / Negative
```

The implemented model uses:

| Layer | Configuration | Purpose |
|---|---|---|
| Embedding | `input_dim=10000`, `output_dim=32` | Converts word IDs into 32-dimensional vectors |
| LSTM | `units=64` | Learns sequential dependencies in reviews |
| Dense | `units=1` | Produces a single classification output |
| Sigmoid | `activation="sigmoid"` | Produces a probability between 0 and 1 |

These layers and configurations are directly implemented in the source code. fileciteturn0file0L123-L150

---

## ⚙️ Configuration

The project uses:

```python
VOCAB_SIZE = 10000
MAX_LENTH = 200
```

Meaning:

- **VOCAB_SIZE = 10000** → uses the 10,000 most frequent words from the IMDB dataset.
- **MAX_LENTH = 200** → each review is converted to a sequence with a maximum length of 200 tokens.

The source code then applies `pad_sequences()` to both training and testing data. fileciteturn0file0L105-L120

> Note: The source code uses the variable name `MAX_LENTH`. It works because the same variable is used consistently, although `MAX_LENGTH` would be the conventional spelling.

---

## 📊 Dataset

This project uses the **IMDB Movie Reviews dataset provided by TensorFlow/Keras**.

The dataset is loaded with:

```python
(X_train, Y_train), (X_test, Y_test) = imdb.load_data(
    num_words=VOCAB_SIZE
)
```

The data contains:

```text
X_train → Training reviews
Y_train → Training sentiment labels

X_test  → Testing reviews
Y_test  → Testing sentiment labels
```

The source code also prints the number of training and testing reviews after loading the dataset. fileciteturn0file0L18-L32

---

## 🔤 Understanding Encoded Reviews

The IMDB dataset provides reviews as sequences of integer IDs rather than normal sentences.

For example:

```text
Encoded Review
[20, 56, 78, 43]
```

can represent words through the dataset's word-index mapping.

The project creates a reverse dictionary to convert integer IDs back into words and defines a `DecodeReview()` function for displaying readable reviews. fileciteturn0file0L45-L79

The project displays sample reviews before model training so that the encoded dataset can be understood more easily. fileciteturn0file0L81-L103

---

## 🔢 Padding

Movie reviews have different lengths, but neural networks require input sequences of a consistent shape.

The project uses:

```python
X_train_padded = pad_sequences(
    X_train,
    maxlen=MAX_LENTH
)

X_test_padded = pad_sequences(
    X_test,
    maxlen=MAX_LENTH
)
```

This converts the variable-length reviews into fixed-length sequences of 200 tokens. fileciteturn0file0L105-L120

---

## 🏋️ Model Compilation

The model is compiled using:

```python
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
```

### Optimizer

**Adam** is used to update the neural-network weights during training.

### Loss Function

**Binary Cross-Entropy** is used because this project performs binary classification:

```text
Positive
Negative
```

### Metric

**Accuracy** is used to measure classification performance during training and evaluation. fileciteturn0file0L152-L162

---

## 🚀 Model Training

The model is trained using:

```python
model.fit(
    X_train_padded,
    Y_train,
    epochs=3,
    batch_size=64,
    validation_split=0.2
)
```

Configuration:

| Parameter | Value |
|---|---:|
| Epochs | 3 |
| Batch Size | 64 |
| Validation Split | 20% |
| Optimizer | Adam |
| Loss | Binary Cross-Entropy |

The source uses 20% of the training data for validation during training. fileciteturn0file0L164-L178

---

## 📈 Model Evaluation

After training, the model is evaluated on the test dataset:

```python
accuracy = model.evaluate(
    X_test_padded,
    Y_test,
    verbose=0
)
```

The resulting evaluation output is then printed as:

```text
Testing accuracy : ...
```

The actual accuracy depends on the environment and training run, so this README does **not** hard-code an accuracy value that is not present in the source code. fileciteturn0file0L180-L190

---

## 🔮 Sentiment Prediction

The project selects one test review and performs prediction:

```python
TEST_REVIEW_NUMBER = 0
```

The review is:

1. Retrieved from the test dataset.
2. Decoded into readable text.
3. Compared against its actual sentiment.
4. Passed through the trained LSTM model.
5. Converted into a positive/negative prediction using a probability threshold of `0.5`. fileciteturn0file0L192-L235

### Prediction Logic

```python
if probability >= 0.5:
    predicted_sentiment = "POSITIVE"
else:
    predicted_sentiment = "NEGATIVE"
```

The final output displays:

```text
Prediction Probability : ...
Actual Sentiment       : ...
Predicted Sentiment    : ...
```

This final comparison is implemented in the source code. fileciteturn0file0L237-L245

---

## 🛠️ Tech Stack

- 🐍 **Python**
- 🧠 **TensorFlow**
- 🔗 **Keras**
- 🔤 **NLP**
- 🧮 **NumPy** (used internally by the TensorFlow/Keras workflow)
- 🤖 **LSTM**
- 📊 **Binary Classification**

### Main Libraries Used in the Source

```python
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
```

These are the libraries actually imported by the project source. fileciteturn0file0L1-L8

---

## 📂 Project Structure

Recommended GitHub repository structure:

```text
Movie-Sentiment-Analysis-LSTM/
│
├── 3_LSTMModel.py
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 💻 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/movie-sentiment-analysis-lstm.git
cd movie-sentiment-analysis-lstm
```

### 2. Create a Virtual Environment

#### macOS / Linux

```bash
python3 -m venv LSTM
source LSTM/bin/activate
```

#### Windows

```bash
python -m venv LSTM
LSTM\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Project

```bash
python 3_LSTMModel.py
```

On the first run, TensorFlow/Keras may download the IMDB dataset automatically.

---

## 🖥️ Expected Program Flow

When the program runs, it displays information similar to:

```text
----------------------------------------
Movie Review Sentiment Analysis using LSTM
----------------------------------------
Loading the dataset...
IMDB dataset loaded successfully

Number of training reviews : ...
Number of testing reviews  : ...

--------- Sample Reviews ---------

Review number : ...
Review :
...

Sentiment : POSITIVE
```

After preprocessing and training, the program evaluates the model and displays the final prediction:

```text
----------------------------------------
Final Result
----------------------------------------
Prediction Probability : ...
Actual sentiment       : ...
Predicted sentiment    : ...
----------------------------------------
```

The exact numerical values depend on the training run. The output stages correspond to the source code's dataset loading, sample-review, training, evaluation, and prediction sections. fileciteturn0file0L21-L32 fileciteturn0file0L85-L103 fileciteturn0file0L237-L245

---

## 📚 Concepts Demonstrated

This project is useful for learning:

- Natural Language Processing
- Sentiment Analysis
- Text Classification
- Word Embeddings
- Sequence Padding
- Recurrent Neural Networks
- Long Short-Term Memory Networks
- Binary Classification
- Sigmoid Activation
- Binary Cross-Entropy
- Adam Optimization
- Model Training
- Model Evaluation
- Model Prediction

---

## 🔬 Possible Improvements

The current source implements a straightforward **Embedding → LSTM → Dense/Sigmoid** architecture. Possible extensions include:

- [ ] Add a Dropout layer
- [ ] Add a Bidirectional LSTM
- [ ] Compare Simple RNN, LSTM, and GRU
- [ ] Increase/decrease vocabulary size
- [ ] Experiment with sequence length
- [ ] Tune the embedding dimension
- [ ] Tune the number of LSTM units
- [ ] Add training-history plots
- [ ] Add a confusion matrix
- [ ] Build a Streamlit interface
- [ ] Create an API for sentiment prediction
- [ ] Allow users to enter their own movie review
- [ ] Save and load the trained model

---

## 💼 Resume Description

**Movie Review Sentiment Analysis using LSTM**

> Built an LSTM-based deep learning model for binary sentiment classification of IMDB movie reviews. Preprocessed encoded review sequences using fixed-length padding, represented words using an Embedding layer, and designed an LSTM-based architecture with a sigmoid output for positive/negative sentiment prediction. Trained the model using Adam optimization and binary cross-entropy loss.

### Resume Tech Stack

```text
Python | TensorFlow | Keras | LSTM | NLP | Deep Learning
```

---

## ⭐ Project Highlights

- Implemented an **LSTM-based NLP classifier**
- Used the **IMDB movie review dataset**
- Processed the **10,000 most frequent words**
- Standardized reviews to **200 tokens**
- Used **32-dimensional word embeddings**
- Used **64 LSTM units**
- Used **Adam + Binary Cross-Entropy**
- Included **validation during training**
- Evaluated the model on unseen test data
- Implemented an end-to-end sentiment prediction pipeline

The architecture and numerical settings above are based on the uploaded source code. fileciteturn0file0L123-L175

---

## 👨‍💻 Author

**Your Name**

ASHUTOSH KADU
<h3>AI Engineer | Neural Network </h3>

---

## 📄 License

This project is intended for educational and learning purposes.
