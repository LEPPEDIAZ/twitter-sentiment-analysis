import streamlit as st
# importing tools from your library into the demo:
from ana_q_aylien.model import ModelEvaluation


page_config = st.set_page_config(
    page_title="Twitter Sentiment Analysis by Ana",
)


def main():
    predictor = ModelEvaluation(model_name='../../ana_q_aylien/resources/ana-q-aylien_trained.pt')

    st.write("# Twitter Sentiment Analysis")

    text = st.text_input("Enter Tweet")
    if st.button("Click to predict sentiment"):
        predictor.define_and_load_model()

        predictor.receive_and_cleanse_single_tweet(text)
        res = predictor.predict()

        st.write(res)


if __name__ == '__main__':
    main()
