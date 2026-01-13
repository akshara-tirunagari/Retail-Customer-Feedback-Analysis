# Retail-Customer-Feedback-Analysis 🛒

Grocery Store Customer Sentiment Analytics : End-to-end sentiment analysis and topic modeling of grocery store customer reviews (Trader Joe's/Whole Foods) using NLP and Deep Learning.

This project analyzes customer feedback from major grocery chains to identify key drivers of satisfaction and dissatisfaction. By applying Natural Language Processing (NLP) and Deep Learning techniques to unstructured review data, we generated actionable insights for improving the retail customer experience.

## 📄 Project Files
* **`1_Grocery_Data_EDA.ipynb`**: Exploratory Data Analysis. Visualizes rating distributions, review lengths, and word frequencies to understand the data landscape.
* **`2_Sentiment_Deep_Learning_Model.ipynb`**: The core modeling engine. Implements Deep Learning architectures (LSTM & RNN) to classify reviews as Positive, Negative, or Neutral.
* **`Executive_Presentation.pdf`**: High-level business summary, methodology, and strategic recommendations for stakeholders.

## 🔍 Business Insights & Methodology

### 1. The Challenge
Grocery retail is highly competitive. Understanding *why* customers leave 1-star vs. 5-star reviews is critical for retention. This project moved beyond simple star ratings to understand the context of the text.

### 2. Technical Approach & Processes
The solution follows a rigorous Deep Learning pipeline:

* **Data Preprocessing:**
    * **Text Cleaning:** Removal of non-alphabetic characters and conversion to lowercase.
    * **Tokenization:** Used `Keras Tokenizer` to convert text into numeric sequences (utilizing the top 5,000 most frequent words).
    * **Sequence Padding:** applied `pad_sequences` to ensure uniform input length for the neural networks.
    * **Label Encoding:** Transformed categorical sentiment labels into numeric targets using `LabelEncoder`.

* **Model Architecture:**
    * **Embedding Layer:** Mapped integer-encoded words to dense vectors to capture semantic meaning.
    * **Recurrent Layers:** Implemented and benchmarked **SimpleRNN** vs. **LSTM (Long Short-Term Memory)** layers to handle sequential text data.
    * **Dense Output:** Used a Softmax output layer for multi-class sentiment classification.

* **Evaluation:**
    * Optimized using `categorical_crossentropy` loss and the `Adam` optimizer.
    * Visualized performance using **Confusion Matrices** and Training/Validation Accuracy curves to detect overfitting.

## 🛠️ Tools & Libraries Used
* **Deep Learning:** `TensorFlow`, `Keras` (Sequential API, LSTM, Embedding, Dense).
* **Data Manipulation:** `Pandas`, `NumPy`.
* **Preprocessing:** `Scikit-learn` (LabelEncoder, train_test_split).
* **Visualization:** `Matplotlib` (Learning curves), `Seaborn` (Heatmaps).
