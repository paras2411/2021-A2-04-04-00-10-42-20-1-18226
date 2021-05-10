

import numpy as np
import ast
import sys
import pandas as pd
import tensorflow as tf
from tensorflow import keras

PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token
OOV_token = 3  # out of vocabulary token


batch_size = 64  # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 10000  # Number of samples to train on.


dim = 50


class Vocabulary:

    def __init__(self, name):

        self.name = name
        self.word2index = {"OOV": OOV_token, "PAD": PAD_token, "SOS": SOS_token, "EOS": EOS_token}
        self.word2count = {"OOV": 0}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS", OOV_token: "OOV"}
        self.num_words = 4
        self.longest_token_list = 0

    def add_word(self, word):

        if word not in self.word2index:
            # First entry of word into vocabulary
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1

        self.word2count[word] += 1

    def add_tokens(self, tokens):

        for token in tokens:
            self.add_word(token)

    def to_word(self, index):
        return self.index2word[index]

    def to_index(self, word):
        return self.word2index[word]


input_filename = sys.argv[1]
output_filename = sys.argv[2]
df = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/ASEML/demo/train.csv")

val_df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/ASEML/demo/'+ input_filename)

### Create vocabulary

input_tokens = []
target_tokens = []

vocab = Vocabulary("input")

for i in range(len(df)):
    sourceLineToken = ast.literal_eval(df['sourceLineTokens'][i])
    input_tokens.append(sourceLineToken)
    vocab.add_tokens(sourceLineToken)

for i in range(len(df)):
    targetLineToken = ast.literal_eval(df['targetLineTokens'][i])
    target_tokens.append(targetLineToken)

vocab.word2count = sorted(vocab.word2count.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)

valid_dict = dict(vocab.word2count[:300])

vocab = Vocabulary("output")

for key in valid_dict.keys():
    vocab.add_word(key)



# Define sampling models
# Restore the model and construct the encoder and decoder.
model = keras.models.load_model("/content/drive/MyDrive/Colab Notebooks/ASEML/demo/s2s")

encoder_inputs = model.input[0]  # input_1
encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output  # lstm_1
encoder_states = [state_h_enc, state_c_enc]
encoder_model = keras.Model(encoder_inputs, encoder_states)

decoder_inputs = model.input[1]  # input_2
decoder_state_input_h = keras.Input(shape=(latent_dim,), name="input_3")
decoder_state_input_c = keras.Input(shape=(latent_dim,), name="input_4")
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_lstm = model.layers[3]
decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs
)
decoder_states = [state_h_dec, state_c_dec]
decoder_dense = model.layers[4]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = keras.Model(
    [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
)



def correct_code(input_token):
    states_value = encoder_model.predict(input_token)
    target_seq = np.zeros((1, 1, vocab.num_words))
    target_seq[0, 0, SOS_token] = 1.0

    correct_code_list = []

    while True:

        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])

        # print(type(sampled_token_index))
        sampled_token = vocab.index2word[sampled_token_index]
        # print(sampled_token)
        correct_code_list.append(sampled_token)

        target_seq = np.zeros((1, 1, vocab.num_words))
        target_seq[0, 0, sampled_token_index] = 1.0

        states_value = [h, c]

        if sampled_token == "EOS" or len(correct_code_list) > dim:
            break

    return correct_code_list


df = val_df

input_tokens = []

for i in range(len(df)):
  sourceLineToken = ast.literal_eval(df['sourceLineTokens'][i])
  input_tokens.append(sourceLineToken)

for i in range(len(input_tokens)):
  for j in range(len(input_tokens[i])):
    if input_tokens[i][j] not in vocab.word2index.keys():
      input_tokens[i][j] = "OOV"

for i in range(len(input_tokens)):
  input_tokens[i] = input_tokens[i][:dim-1]


encoder_input_data = np.zeros(
    (len(input_tokens), dim + 1, vocab.num_words), dtype=float
)

for i, (input_token) in enumerate(input_tokens):

    # input_token = input_token[:dim-1]
    for t, token in enumerate(input_token):
        encoder_input_data[i, t, vocab.word2index[token]] = 1.0

    t = t + 1
    encoder_input_data[i, t, EOS_token] = 1.0

    encoder_input_data[i, t + 1: , PAD_token] = 1.0


corrected_code = []

ip = len(val_df)


for i in range(len(val_df)):
    sourceLineToken = ast.literal_eval(val_df['sourceLineTokens'][i])
    targetLineToken = ast.literal_eval(val_df['targetLineTokens'][i])

    cor_code = correct_code(encoder_input_data[i:i + 1])

    corrected_code.append(cor_code)

    for j, token in enumerate(targetLineToken):
        if token not in vocab.word2index.keys():
            targetLineToken[j] = 'OOV'

    end = min(len(targetLineToken), dim)
    for j in range(end):
        if targetLineToken[j] != cor_code[j]:
            ip -= 1
            break


### Ip will contain the number of correctly matched tokens

print("Accuracy: ", ip/len(val_df))

val_df['fixedTokens'] = corrected_code
val_df.to_csv('/content/drive/MyDrive/Colab Notebooks/ASEML/demo/' + output_filename)