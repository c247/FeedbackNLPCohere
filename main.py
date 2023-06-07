from flask import Flask, render_template, request
import cohere
from cohere.responses.classify import Example

co = cohere.Client('IgYIDVRHkDp4vev0qqqGIGpPONSupyW21XBQYqXN')


examples = [
    Example("The order came 5 days early", "positive"),
    Example("The item exceeded my expectations", "positive"),
    Example("The customer service was friendly and helpful", "positive"),
    Example("The website was easy to navigate", "positive"),
    Example("The product quality was top-notch", "positive"),
    Example("The delivery was delayed without any updates", "negative"),
    Example("The item arrived damaged", "negative"),
    Example("The customer support was unresponsive", "negative"),
    Example("The pricing was too high for the product", "negative"),
    Example("The product did not meet my expectations", "negative"),
]

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = request.form['feedback']
        response = analyze_sentiment(user_input)
        return render_template('index.html', user_input=user_input, response=response)
    return render_template('./index.html')


def analyze_sentiment(text):
    response = co.classify(
        model='large',
        inputs=[text],
        examples=examples,
    )
    classification = response.classifications[0]
    prediction = classification.prediction

    if prediction == 'negative':
        return "Sorry to hear that. Please reach out to v2ananth@uwaterloo.ca to address your concerns."
    else:
        return "Great! We're glad you enjoyed our service."

if __name__ == '__main__':
    app.run(debug=True)
