from flask import Flask, render_template, url_for, request
from mypredict import processing_predict
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    question = request.form.get('question')
    answers = processing_predict(question)
    print(question)
    return render_template('result.html', len=len(answers), Pokemons=answers, question = question)


if __name__ == "__main__":
    app.run(debug=True)
