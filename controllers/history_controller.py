from flask import Blueprint, render_template, make_response, request
from flask_login import current_user
from flask_login import login_required

from db_models import SummarizedText, db

history_controller = Blueprint('history_controller', __name__)


@login_required
@history_controller.route('/history', methods=['GET', 'POST'])
def history():
    if request.method == 'POST':
        if current_user.is_authenticated:
            data = request.get_json()
            new_summarized_text = SummarizedText(
                title=data['title'],
                text=data['text'],
                user_id=current_user.id
            )
            db.session.add(new_summarized_text)
            db.session.commit()
        return make_response('Ok', 200)
    return render_template(
        'html/history.html',
        history=sorted(current_user.summarized_texts, key=lambda x: x.create_ts, reverse=True)
    )
