from flask import Blueprint, jsonify, request, render_template
from flask_login import current_user
from summarization.abstractive.BartSummarizer import BartSummarizer
from summarization.extractive.TfidfSummarizer import TfidfSummarizer

summarization_controller = Blueprint('summarization_controller', __name__)


@summarization_controller.route('/', methods=['GET'])
def index():
    if current_user.is_authenticated:
        return render_template('html/index.html', username=current_user.username)
    return render_template('html/index.html')


@summarization_controller.route('/transcribing', methods=['POST'])
def transcribing():  # TODO: ДОБАВИТЬ ЛЁХИН МЕТОД
    file = request.files['file']
    return jsonify({
        'transcribed_text': file.read().decode('utf-8')
    })


@summarization_controller.route('/extractive', methods=['POST'])
def extractive_summarization():
    data = request.get_json()
    text = data['text']
    summarized_text = TfidfSummarizer.summarize_text(text)
    return jsonify({
        'summarized_text': summarized_text
    })


@summarization_controller.route('/abstractive', methods=['POST'])
def abstractive_summarization():
    data = request.get_json()
    text = data['text']
    bart = BartSummarizer()
    summarized_text = bart.summarize_text(text)
    return jsonify({
        'summarized_text': summarized_text
    })
