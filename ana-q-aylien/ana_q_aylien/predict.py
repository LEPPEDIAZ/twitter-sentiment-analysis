import argparse
import warnings

from ana_q_aylien.model import ModelEvaluation

warnings.filterwarnings("ignore")

def main(input_file, output_file):
    evaluator = ModelEvaluation(model_name='resources/ana-q-aylien_trained.pt')

    evaluator.define_and_load_model()
    evaluator.load_tweets_from_file(input_file)

    results = evaluator.process_and_predict_tweets()

    evaluator.save_results_to_file(results, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict sentiments of tweets.")
    parser.add_argument('--input', type=str, required=True, help="Path to the input file containing tweets.")
    parser.add_argument('--output', type=str, required=True, help="Path to the output file where predictions will be saved.")

    args = parser.parse_args()

    main(args.input, args.output)
