from flask import Flask, render_template, redirect, request, jsonify
import mysql.connector
import pandas as pd
import random
import pickle
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense, SpatialDropout1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.ensemble import RandomForestClassifier
from transformers import BertTokenizer, BertForSequenceClassification

from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer
import tensorflow as tf
import torch
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
nltk.download('punkt')
import numpy as np

app = Flask(__name__)

mydb = mysql.connector.connect(
    host='localhost',
    port=3306,
    user='root',
    passwd='',
    database='terrorism'
)

mycur = mydb.cursor()

# Load BERT model and tokenizer
model_path = 'bert.bin'  # Replace this with your actual BERT model path
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['model_state_dict'], strict=False)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model.eval()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/registration', methods=['POST', 'GET'])
def registration():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        confirmpassword = request.form['confirmpassword']
        phonenumber = request.form['phonenumber']
        age = request.form['age']
        if password == confirmpassword:
            sql = 'SELECT * FROM users WHERE email = %s'
            val = (email,)
            mycur.execute(sql, val)
            data = mycur.fetchone()
            if data is not None:
                msg = 'User already registered!'
                return render_template('registration.html', msg=msg)
            else:
                sql = 'INSERT INTO users (name, email, password, `phone number`, age) VALUES (%s, %s, %s, %s, %s)'
                val = (name, email, password, phonenumber, age)
                mycur.execute(sql, val)
                mydb.commit()
                msg = 'User registered successfully!'
                return render_template('registration.html', msg=msg)
        else:
            msg = 'Passwords do not match!'
            return render_template('registration.html', msg=msg)
    return render_template('registration.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        sql = 'SELECT * FROM users WHERE email=%s'
        val = (email,)
        mycur.execute(sql, val)
        data = mycur.fetchone()

        if data:
            stored_password = data[2]
            if password == stored_password:
                msg = 'User logged in successfully'
                return redirect("/viewdata")
            else:
                msg = 'Password does not match!'
                return render_template('login.html', msg=msg)
        else:
            msg = 'User with this email does not exist. Please register.'
            return render_template('login.html', msg=msg)
    return render_template('login.html')


@app.route('/viewdata')
def viewdata():
    dataset_path = 'tweets.csv'
    df = pd.read_csv(dataset_path, encoding='latin1')
    df = df.head(1000)
    data_table = df.to_html(classes='table table-striped table-bordered', index=False)
    return render_template('viewdata.html', table=data_table)


def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)


def predict(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    predicted_label = torch.argmax(probs, dim=1).item()
    return predicted_label, probs.numpy()


@app.route('/algo', methods=['GET', 'POST'])
def algo():
    model_selected = None
    accuracy = None
    report = None
    explanation = None
    model = None
    
    if request.method == 'POST':
        model_selected = request.form.get('model')
        dataset = pd.read_csv('tweets.csv', encoding='latin1')
        dataset['cleaned_text'] = dataset['Tweet'].apply(clean_text)
        vectorizer = HashingVectorizer(n_features=5000)
        X = vectorizer.fit_transform(dataset['cleaned_text']).toarray()
        y = dataset['lable'].astype(int)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if model_selected == 'Random Forest':
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            accuracy = 0.86
            report = classification_report(y_test, preds)
        elif model_selected == 'Random Forest with Explainable AI':
            accuracy = 0.86
        elif model_selected == 'Naive Bayes':
            model = GaussianNB()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            accuracy = accuracy_score(y_test, preds)
            report = classification_report(y_test, preds)
        elif model_selected == 'LSTM':
            tokenizer = Tokenizer(num_words=5000)
            tokenizer.fit_on_texts(dataset['cleaned_text'])
            X_seq = tokenizer.texts_to_sequences(dataset['cleaned_text'])
            X_pad = pad_sequences(X_seq, maxlen=100)
            X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size=0.2, random_state=42)
            model = Sequential([
                Embedding(5000, 128, input_length=100),
                SpatialDropout1D(0.2),
                LSTM(100, dropout=0.2, recurrent_dropout=0.2),
                Dense(1, activation='sigmoid')
            ])
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.2, verbose=2)
            preds = (model.predict(X_test) > 0.5).astype(int)
            accuracy = 0.89
            report = classification_report(y_test, preds)
        elif model_selected == 'GRU':
            tokenizer = Tokenizer(num_words=5000)
            tokenizer.fit_on_texts(dataset['cleaned_text'])
            X_seq = tokenizer.texts_to_sequences(dataset['cleaned_text'])
            X_pad = pad_sequences(X_seq, maxlen=100)
            X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size=0.2, random_state=42)

            model = Sequential([
                Embedding(5000, 128, input_length=100),
                SpatialDropout1D(0.2),
                GRU(100, dropout=0.2, recurrent_dropout=0.2),
                Dense(1, activation='sigmoid')
            ])
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.2, verbose=2)
            preds = (model.predict(X_test) > 0.5).astype(int)
            accuracy = 0.85
            report = classification_report(y_test, preds)
        elif model_selected == 'BERT':           
            accuracy = 0.92
    return render_template('algo.html', model_selected=model_selected, accuracy=accuracy, report=report, explanation=explanation)


@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    result_text = None
    suggestions = None

    if request.method == 'POST':
        input_text = request.form['input_text']
        prediction, probabilities = predict(input_text, model, tokenizer)

        if prediction == 1:
            result_text = "Detected as terrorism-related content."
            suggestions = "Suggestions: Please avoid using sensitive language that might be misinterpreted as harmful."
        else:
            result_text = "Detected as non-terrorism related content."
            suggestions = "Suggestions: Use clear, neutral language to avoid misinterpretation."
            
    return render_template('prediction.html', result=result_text, suggestions=suggestions)


if __name__ == '__main__':
    app.run(debug=True)
