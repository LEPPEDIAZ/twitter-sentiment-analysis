import re
import torch
import numpy as np
import torch.nn as nn

from ana_q_aylien.utils import vocabulary


class SelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(SelfAttention, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(self.device)


    def forward(self, x):
        N = x.shape[0]
        value_len, key_len, query_len = x.shape[1], x.shape[1], x.shape[1]
        values = self.values(x).view(N, value_len, self.num_heads, self.head_dim).to(self.device)
        keys = self.keys(x).view(N, key_len, self.num_heads, self.head_dim).to(self.device)
        queries = self.queries(x).view(N, query_len, self.num_heads, self.head_dim).to(self.device)
        values = values.permute(0, 2, 1, 3)
        keys = keys.permute(0, 2, 1, 3)
        queries = queries.permute(0, 2, 1, 3)
        energy = torch.matmul(queries, keys.permute(0, 1, 3, 2)) / self.scale
        attention = torch.nn.functional.softmax(energy, dim=3)
        out = torch.matmul(attention, values)
        out = out.permute(0, 2, 1, 3).contiguous().view(N, query_len, self.embed_size)
        return self.fc_out(out)


class TransformerLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, hidden_size, output_size, num_layers, dropout=0.5):
        super(TransformerLSTM, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.embedding = nn.Embedding(vocab_size, embed_size).to(self.device)
        
        # Layers
        self.conv1 = nn.Conv1d(in_channels=embed_size, out_channels=128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        self.self_attention = nn.ModuleList([SelfAttention(embed_size, num_heads) for _ in range(3)])  # Multiple Self-Attention Layers
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.permute(0, 2, 1)
        
        for layer in self.self_attention:
            x = layer(x)
        
        x, _ = self.lstm(x)
        return self.fc(self.dropout(x[:, -1, :]))


class ModelEvaluation:
    def __init__(self, model_name='ana_q_aylien/resources/ana-q-aylien_trained.pt'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.model = None
        self.single_tweet = None
        self.multiple_tweets = None
    

    def define_and_load_model(self):
        if not self.model:
            self.model = TransformerLSTM(
                vocab_size=len(vocabulary), embed_size=512, num_heads=8, hidden_size=512,
                output_size=3, num_layers=2, dropout=0.5
                ).to(self.device)

        self.model.load_state_dict(torch.load(self.model_name, map_location=self.device))
        self.model.eval()


    @staticmethod
    def _clean_single_tweet(tweet):
        tweet = tweet.lower()
        # Regex to remove User mentions, URLs, and special characters
        return re.sub(r'(@\w+|https?://\S+|[^\w\s])', '', tweet)
    

    @staticmethod
    def _encode_tweet(tweet):
        return [vocabulary.get(char, 0) for char in tweet]
    

    @staticmethod
    def _pad_sequences(tweet_sequence, max_len=100):
        # Initialize padded_sequences with zeros
        padded_sequences = np.zeros((len(tweet_sequence), max_len), dtype=int)
    
        for i, seq in enumerate(tweet_sequence):
            if len(seq) > 0:
                if len(seq) > max_len:
                    padded_sequences[i, :max_len] = seq[:max_len]
                else:
                    padded_sequences[i, -len(seq):] = seq
                
        return torch.tensor(padded_sequences)
    

    def receive_and_cleanse_single_tweet(self, tweet):
        clean_tweet = self._clean_single_tweet(tweet)
        encoded_tweet = self._encode_tweet(clean_tweet)
        padded_tweet = self._pad_sequences([encoded_tweet])

        self.single_tweet = padded_tweet.to(self.device)


    def predict(self):
        self.model.eval()

        with torch.no_grad():
            output = self.model(self.single_tweet)

        predicted_class_idx = torch.argmax(output, dim=1).item()
        class_labels = {0: 'Negative', 1: 'Positive', 2: 'Neutral'}

        return class_labels[predicted_class_idx]


    def load_tweets_from_file(self, file_path='ana_q_aylien/resources/test.txt'):
        with open(file_path, 'r') as file:
            self.multiple_tweets = file.readlines()
        
    def process_and_predict_tweets(self):
        results = []
        for tweet in self.multiple_tweets:
            self.receive_and_cleanse_single_tweet(tweet.strip())
            sentiment = self.predict()
            results.append(sentiment)
        return results
    
    def save_results_to_file(self, results, output_file_path='ana_q_aylien/resources/test_result.txt'):
        with open(output_file_path, 'w') as file:
            for sentiment in results:
                file.write(f"{sentiment}\n")