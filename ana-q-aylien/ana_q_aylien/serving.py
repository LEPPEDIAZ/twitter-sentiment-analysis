from aylien_model_serving.app_factory import FlaskAppWrapper
from ana_q_aylien.model import ModelEvaluation


def run_app():
    
    predict = ModelEvaluation()

    def predict_text(text):
        nonlocal predict

        predict.define_and_load_model()

        predict.receive_and_cleanse_single_tweet(text)
        res = predict.predict()

        response = {"sentiment": res}
        return response


    def process_predict_text():
        return FlaskAppWrapper.process_json(predict_text)


    routes = [
        {
            "endpoint": "/predict",
            "callable": process_predict_text,
            "methods": ["POST"]
        }
    ]

    return FlaskAppWrapper.create_app(routes)
