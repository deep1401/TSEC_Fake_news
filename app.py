from flask import Flask,request
from flask_cors import CORS, cross_origin
from fastai.basic_train import load_learner
from fastai.text import *
from sklearn.metrics import f1_score

app = Flask(__name__)
CORS(app, support_credentials=True)


@np_func
def f1(inp, targ): return f1_score(targ, np.argmax(inp, axis=-1))


learn_title_cl = load_learner('./models',file='title_classifier.pkl')
learn_news_cl = load_learner('./models', file='news_classifier.pkl')


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/predict', methods=['POST'])
def predict_category():
    title = request.form.get('title')
    article = request.form.get('text')
    title_pred = learn_title_cl.predict(title)
    news_pred = learn_news_cl.predict(article)
    result = (title_pred[2] + news_pred[2])/2

    return {'Fake': float(result[0]),
            'True': float(result[1])}


if __name__ == '__main__':
    app.run(debug=True)
