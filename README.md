📰 Fake News Detection

---

📌 Overview

This project focuses on building a Fake News Detection System using Natural Language Processing (NLP) and Deep Learning techniques.
The main objective is to classify news articles as Real or Fake by experimenting with multiple models, evaluating their performance, and deploying the final solution.


---

💡 Motivation

In today’s digital era, the spread of fake news through social media and online platforms has become a major global challenge.
Misinformation can affect public opinion, influence elections, and even create social unrest.

Our motivation behind this project was:

To apply NLP and Deep Learning in solving a real-world problem.

To build models capable of detecting misinformation effectively.

To gain hands-on experience with both traditional ML models and state-of-the-art transformers like BERT.

To demonstrate how AI can be applied for social good and promote trust in information sources.



---

📂 Dataset

We used a publicly available Fake News Detection dataset.

🔧 Preprocessing Steps

Text cleaning: removing punctuation, stopwords, and special characters.

Tokenization and lemmatization.

Vectorization using TF-IDF and word embeddings.

Splitting into training, validation, and testing sets.



---

⚙️ Methodology

1. Exploratory Data Analysis (EDA)

Analyzing data distribution, text length, and word frequencies.

Visualizing class balance/imbalance.



2. Preprocessing

Applying NLP preprocessing pipeline to prepare data for modeling.



3. Modeling
We implemented and compared multiple approaches:

BERT Fine-Tuning → Leveraging transformer-based contextual embeddings.

LSTM Model → Capturing sequential dependencies in text.

Classical ML baselines (e.g., Logistic Regression, Naive Bayes).



4. Evaluation Metrics

Accuracy

Precision, Recall, F1-Score

Confusion Matrix

ROC-AUC





---

🚀 Streamlit Web App Deployment

The final stage of this project will be deploying a Streamlit Web Application that provides an interactive interface for Fake News Detection.

Key features of the app will include:

A simple and user-friendly UI built with Streamlit.

Input field for users to paste a news headline or article.

Real-time predictions from trained models (BERT and LSTM).

Visualization of prediction confidence scores.

You can see the application using this [link](https://finetuning--roberta-base-for-fake-news-classifier-otpwywcmgxyu.streamlit.app/).



---

📊 Results

🧠 Model

The final model used was:

LSTM (Long Short-Term Memory) with Word Embeddings

Integrated with BERT embeddings for improved contextual understanding.


🎯 Performance Metrics

Precision

Class 0 (Fake News): 0.98

Class 1 (True News): 0.99


Recall

Class 0 (Fake News): 1.00

Class 1 (True News): 0.99


F1-Score

Class 0 (Fake News): 0.99

Class 1 (True News): 0.99



✅ These results indicate very strong performance with high precision, recall, and F1-scores across both classes.




---

🔮 Future Work

Explore larger transformer models (RoBERTa, XLNet, etc.).

Extend the system for multi-lingual fake news detection.

Experiment with ensemble methods to combine model strengths.



---

📂 Project Structure

GTC-Fake-News-Detection/
│
├── Bert_FineTuning_NLP.ipynb        # Notebook for fine-tuning BERT on Fake News dataset
├── model_trainnig_using_lstm.ipynb  # Notebook for building and training LSTM model
├── code.ipynb                       # General experiments and trials
│
├── Fake.csv                         # Dataset: Fake news samples
├── True.csv                         # Dataset: Real news samples
├── final_data_test.csv              # Dataset: Final test set
│
├── requirements.txt                 # Project dependencies
├── .gitignore                       # Files ignored by Git
└── README.md                        # Project documentation


---

👥 Contributors

This project was developed collaboratively by our team:

Mohammed Elhelaly → EDA & Preprocessing

Mohamed Mahmoud → EDA &
Preprocessing 

Ahmed Yusri → BERT Fine-tuning Model Development & Training & Deployment

MohamedBakr21 → LSTM Model Development & Training

mohamedsaid222 → Model Development & Training 

Fady Ramy → Deployment 
(Streamlit Web App) 

Seif Sameh → Deployment 
(Streamlit Web App) 



---

📜 License

This project is for educational purposes and developed as part of our team coursework/project.
  

