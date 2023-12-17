import uuid
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.dialects.postgresql import UUID
from flask_login import UserMixin

db = SQLAlchemy()


class User(db.Model, UserMixin):
    id = db.Column(UUID(as_uuid=True), primary_key=True, default=lambda: uuid.uuid4())
    username = db.Column(db.String, nullable=False)
    email = db.Column(db.String, nullable=False, unique=True)
    password = db.Column(db.String, nullable=False)
    summarized_texts = db.relationship('SummarizedText', backref='user', lazy=True)


class SummarizedText(db.Model):
    id = db.Column(UUID(as_uuid=True), primary_key=True, default=lambda: uuid.uuid4())
    title = db.Column(db.Text, nullable=False)
    text = db.Column(db.Text, nullable=False)
    user_id = db.Column(UUID(as_uuid=True), db.ForeignKey('user.id'), nullable=False)
    create_ts = db.Column(db.TIMESTAMP, nullable=False, default=datetime.now)
