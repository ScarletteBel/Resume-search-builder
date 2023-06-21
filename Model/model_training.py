#Flask test

from flask import Flask

app = Flask(__Resume_searcher_test__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

if __name__ == '__main__':
    app.run(debug=True)

export FLASK_APP=app
export FLASK_ENV=development  # enables debug mode
flask run