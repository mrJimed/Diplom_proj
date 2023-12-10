from summarization.extractive.TfidfSummarizer import TfidfSummarizer
from summarization.abstractive.BartSummarizer import BartSummarizer
from flask import Flask, render_template, request

app = Flask(__name__)


@app.route('/', methods=["GET"])
def index():
    return render_template('html/index.html')


@app.route('/extractive', methods=["POST"])
def extractive_summarization():
    input_text = request.form['inputText']
    summarize = TfidfSummarizer.summarize_text(input_text)
    return render_template('html/index.html', summarization_text=summarize)


@app.route('/abstractive', methods=["POST"])
def abstractive_summarization():
    input_text = request.form['inputText']
    bart = BartSummarizer()
    summarize = bart.summarize_text(input_text)
    return render_template('html/index.html', summarization_text=summarize)


app.run()
