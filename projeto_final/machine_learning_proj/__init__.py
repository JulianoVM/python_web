from flask import Flask

app = Flask(__name__)

from machine_learning_proj import routes
