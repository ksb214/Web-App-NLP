# Web-App-NLP
This is a web app for detecting whether message is spam or ham. User inputs the message in the text box. 

Message text is transformed with TFIDF Vectorizer and then passed to the Random Forest model to predict the model is ham or spam. 

To Run the model: 
%python app.py

In browser open,
http://127.0.0.1:5000

<p><strong>Example of Ham SMS<p><strong>
<p><img src="https://github.com/ksb214/Web-App-NLP/blob/master/Ham_input.png" alt="" /></p>

<p><img src="https://github.com/ksb214/Web-App-NLP/blob/master/Ham_predict.png" alt="" /></p>

<p><strong>Example of Spam SMS<p><strong>

<p><img src="https://github.com/ksb214/Web-App-NLP/blob/master/Spam_input.png" alt="" /></p>

<p><img src="https://github.com/ksb214/Web-App-NLP/blob/master/Spam_predict.png" alt="" /></p>
