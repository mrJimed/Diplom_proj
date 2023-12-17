from dotenv import dotenv_values

env_vars = dotenv_values('.env')

DB_USERNAME = env_vars.get('DB_USERNAME')
DB_PASSWORD = env_vars.get('DB_PASSWORD')
DB_HOST = env_vars.get('DB_HOST')
DB_NAME = env_vars.get('DB_NAME')
DB_PORT = env_vars.get('DB_PORT')

SECRET_KEY = env_vars.get('SECRET_KEY')