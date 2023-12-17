from flask import Flask
from controllers.auth_controller import auth_controller, login_manager
from controllers.summarization_controller import summarization_controller
from controllers.history_controller import history_controller
from db_models import db
from settings import DB_PASSWORD, DB_USERNAME, DB_NAME, DB_PORT, DB_HOST, SECRET_KEY

app = Flask(__name__)
app.secret_key = SECRET_KEY
app.config['SQLALCHEMY_DATABASE_URI'] = f'postgresql://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'

app.register_blueprint(auth_controller)
app.register_blueprint(summarization_controller)
app.register_blueprint(history_controller)

db.init_app(app)
login_manager.init_app(app)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
