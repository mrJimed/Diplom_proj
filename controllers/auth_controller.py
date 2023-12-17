from flask import Blueprint, request, redirect, jsonify, render_template
from flask_login import login_user, LoginManager, login_required, logout_user
from werkzeug.security import check_password_hash, generate_password_hash
from db_models import db, User

login_manager = LoginManager()
login_manager.user_loader(lambda user_id: User.query.get(user_id))
auth_controller = Blueprint('auth_controller', __name__)


@auth_controller.route('/authorization', methods=['GET', 'POST'])
def authorization():
    if request.method == 'POST':
        user = User.query.filter_by(
            email=request.form['email']
        ).first()
        if user is not None and check_password_hash(user.password, request.form['password']):
            login_user(user)
            return redirect('/')
        return jsonify({
            'error': 'Неверный email или пароль'
        }), 401
    return render_template('html/authorization.html')


@auth_controller.route('/registration', methods=['GET', 'POST'])
def registration():
    if request.method == 'POST':
        new_user = User(
            username=request.form['username'],
            email=request.form['email'],
            password=generate_password_hash(request.form['password'])
        )
        db.session.add(new_user)
        db.session.commit()
        login_user(new_user)
        return redirect('/')
    return render_template('html/registration.html')


@login_required
@auth_controller.route('/logout', methods=['GET'])
def logout():
    logout_user()
    return redirect('/')
