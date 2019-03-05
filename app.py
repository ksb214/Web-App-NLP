from flask import Flask, render_template, url_for, request
import pickle
import numpy as np
import pandas as pd
import nltk
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
from clean import count_punct
from clean import clean_text


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	 # Load classifier
	with open(f'data/sms_spam_classifier_RF.pkl', 'rb') as f:
		model = pickle.load(f)

         # Load vectorizer
	tfidf_n = joblib.load(f'data/tfidf.pkl')

	if request.method =='POST':
		smstext = request.form['comment']
		#predict_data = [smstext]
		
		df = pd.DataFrame([[smstext]],columns=['body_text'],dtype=object)
		df['body_len'] = df['body_text'].apply(lambda x: len(x) - x.count(" "))
		df['punct%'] = df['body_text'].apply(lambda x: count_punct(x))
		#df['body_text'] = df['body_text'].apply(clean_text)
		
		tfidf_predict = tfidf_n.transform(df['body_text'])
		X_predict_vect = pd.concat([df[['body_len', 'punct%']].reset_index(drop=True), pd.DataFrame(tfidf_predict.toarray())], axis=1)
		model_pred = model.predict(X_predict_vect)

		if model_pred[0]=='spam':
			pred=1
		else:
			pred=0

	return render_template('result.html',prediction=pred)

if __name__=='__main__':
	app.run(debug=True)
