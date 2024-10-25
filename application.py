from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
from flask import Flask, render_template, request, jsonify, redirect, url_for

application = Flask(__name__)

# Load model and vectorizer
loaded_model = pickle.load(open('basic_classifier.pkl', 'rb'))
vectorizer = pickle.load(open('count_vectorizer.pkl', 'rb'))


@application.route('/predict', methods=['GET', 'POST'])
def predict():
    # Handle POST request
    if request.method == 'POST':
        if request.is_json:
            # extract 'text' from JSON
            data = request.json
            input_text = data.get('text', '').strip()
        else:
            # get 'text' from form data
            input_text = request.form.get('text', '').strip()
    else:
        # get 'text' from query parameters
        input_text = request.args.get('text', '').strip()

    if not input_text:
        return jsonify({'error': 'No text provided. Please enter some text to analyze.'}), 400

    prediction = loaded_model.predict(vectorizer.transform([input_text]))[0]

    return jsonify({'text': input_text, 'prediction': prediction})


@application.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form.get('text', '').strip()

        if not text:
            return render_template('index.html', error="Please enter some text to analyze.")

        # Use the predict endpoint
        response = predict()
        if response.status_code == 400:
            return render_template('index.html', error=response.json['error'])

        result = response.json['prediction']

        return redirect(url_for('index', text=text, result=result))

    text = request.args.get('text')
    result = request.args.get('result')

    return render_template('index.html', text=text, result=result)


if __name__ == '__main__':
    application.run(debug=True)
