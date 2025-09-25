import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model as keras_load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os  

@st.cache_resource
def load_saved_objects():
    # Load vectorizer
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    # Load tokenizer
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

    # Load LSTM model
    model = keras_load_model("lstm_model.h5")

    return model, vectorizer, tokenizer

# Unpack all three
model, vectorizer, tokenizer = load_saved_objects()

@st.cache_data
def predict_news(text):
    if model is None or tokenizer is None:
        return "Error", 0.0
    
    try:
        # Convert text → sequence
        seq = tokenizer.texts_to_sequences([text])
        if not seq or len(seq[0]) == 0:
            return "Error", 0.0  # Text had no tokens
        
        # Pad sequence (make sure maxlen matches what you used in training, e.g., 400)
        padded = pad_sequences(seq, maxlen=400, padding="post", truncating="post")
        
        # Predict
        proba = model.predict(padded, verbose=0)[0]   # e.g. [0.78, 0.22]
        pred_idx = np.argmax(proba)                   # 0 or 1
        
        # Map to labels
        label = "Fake" if pred_idx == 1 else "Real"
        confidence = float(proba[pred_idx])
        
        return label, confidence
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return "Error", 0.0


# Custom CSS for styling (fixed missing semicolon)
st.markdown("""
<style>
    .stApp {
        background: #AD8B73;
        border-radius: 10px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    .title {
        color: #FFFBE9;
        text-align: center;
        font-size: 2.7rem;
        font-weight: bold;
        margin-bottom: 1.7rem;
    }
    .prediction-section {
        background: white;
        padding: 1.7rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.01); 
    }
    .real-news {
        background: green;
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .fake-news {
        background: red;
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .stButton>button {
        background: #CEAB93;
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        width: 100%;
        margin-top: 1rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 15px rgba(100, 120, 235, 0.5);
    background: #CEAB93;  
    color: white;         
    border: none;         
    }
    .info-box {
        background: #CEAB93;
        color: #FFFBE9;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3b82f6;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Confirm loading 
    if model is None or vectorizer is None:
        st.error("Failed to load model/vectorizer. Check files and restart the app.")
        st.stop()
    
    # The header
    st.markdown('<h1 class="title">Fake News Detector</h1>', unsafe_allow_html=True)
    
    # The info box
    st.markdown("""
    <div class="info-box">
        <strong>How it works:</strong> Enter news text below and our AI model will analyze 
        it for patterns commonly found in fake news articles.
    </div>
    """, unsafe_allow_html=True)
    
    # The input section
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.subheader("Enter News Text")
    news_text = st.text_area(
        "", 
        height=200,
        placeholder="Paste the news article text here...",
        help="Enter at least 50 characters for better accuracy"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # The predict button
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        predict_button = st.button("check validity", use_container_width=True)
    
    if predict_button:
        if not news_text.strip():
            st.error("Please enter some news text to analyze.")
        elif len(news_text.strip()) < 20:
            st.warning("For better accuracy, please enter at least 20 characters.")
        else:
            # Show loading spinner
            with st.spinner("Predicting..."):
                prediction, confidence = predict_news(news_text)
                
                if prediction == "Error":
                    st.stop()  
                
                # Display results
                st.markdown('<div class="prediction-section">', unsafe_allow_html=True)
                
                if prediction == "Real":
                    st.markdown('<div class="real-news">The input is REAL NEWS</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="fake-news">The input is FAKE NEWS</div>', unsafe_allow_html=True)
            
    # The footer
    st.markdown("""
    <br><br>
    <div style='text-align: center; color: #6b7280; font-size: 0.9rem;'>
        This tool is for educational purposes only. Always verify information from multiple reliable sources.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
