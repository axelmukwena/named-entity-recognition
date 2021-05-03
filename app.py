import flask
import run

app = flask.Flask(__name__)


@app.route('/')
def index():
    return flask.render_template('index.html', **locals())


@app.route('/predict', methods=['GET'])
def predict():
    name = flask.request.args.get('name')
    if name != "":
        text_pred = run.predict(name).replace("\n", "<br/>").replace(" ", "&nbsp;")
    return flask.render_template('index.html', **locals())


@app.route('/test')
def run_test():
    text_test = run.test().replace("\n", "<br/>")
    return flask.render_template('index.html', **locals())


@app.route('/show')
def run_show():
    text_show = run.show().replace("\n", "<br/>")
    text_show = text_show.replace(" ", "&nbsp;")
    return flask.render_template('index.html', **locals())


if __name__ == '__main__':
    app.run()
