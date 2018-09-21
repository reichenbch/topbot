# import req libraries
import pandas as pd
import numpy as np
import tensorflow as tf
import re
import time

# import data
lines = open('movie_lines.txt', encoding='utf-8', errors='ignore').read().split('\n')
conv_lines = open('movie_conversations.txt', encoding='utf-8', errors='ignore').read().split('\n')

# print(lines[:10])
# print(conv_lines[:10])

# mapping line to text
id2line = {}
for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line) == 5:
        id2line[_line[0]] = _line[4]

# list of convs
convs = []
for line in conv_lines[:-1]:
    _line = line.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ","")
    convs.append(_line.split(','))

# print(convs[:10])

# converting to inputs and output
questions = []
answers = []

for conv in convs:
    for i in range(len(conv) - 1):
        questions.append(id2line[conv[i]])
        answers.append(id2line[conv[i+1]])

print(len(questions))
print(len(answers))

# cleaning the data function
def clean_text(text):
    text = text.lower()

    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)

    return text

# cleaning the data
clean_questions = []
for question in questions:
    clean_questions.append(clean_text(question))

clean_answers = []
for answer in answers:
    clean_answers.append(clean_text(answer))

limit = 0
for i in range(limit, limit + 5):
    print(clean_questions[i])
    print(clean_answers[i])
    print()

lengths = []
for question in clean_questions:
    lengths.append(len(question.split()))
for answer in clean_answers:
    lengths.append(len(answer.split()))

lengths = pd.DataFrame(lengths, columns=['counts'])
print(lengths.describe())

# remove questions and answers that are shorter than 2 words and longer that 20 words
min_line_length = 2
max_line_length = 20

# Filter out the questions that are too short/long
short_questions_temp = []
short_answers_temp = []

i = 0
for question in clean_questions:
    if len(question.split()) >= min_line_length and len(question.split()) <= max_line_length:
        short_questions_temp.append(question)
        short_answers_temp.append(clean_answers[i])
    i += 1

# Filter out the answers that are too short/long
short_questions = []
short_answers = []

i = 0
for answer in short_answers_temp:
    if len(answer.split()) >= min_line_length and len(answer.split()) <= max_line_length:
        short_answers.append(answer)
        short_questions.append(short_questions_temp[i])
    i += 1

# print("# of questions:", len(short_questions))
# print("# of answers:", len(short_answers))
# print("% of data used: {}%".format(round(len(short_questions)/len(questions),4)*100))

# creating vocab
vocab = {}
for question in short_questions:
    for word in question.split():
        if word not in vocab:
            vocab[word] = 1
        else:
            vocab[word] += 1
for answer in short_answers:
    for word in answer.split():
        if word not in vocab:
            vocab[word] = 1
        else:
            vocab[word] += 1

threshold = 10
count = 0
for k,v in vocab.items():
    if v >= threshold:
        count += 1

print("Size of total vocab:", len(vocab))
print("Size of vocab we will use:", count)

questions_vocab_to_int = {}

word_num = 0
for word, count in vocab.items():
    if count >= threshold:
        questions_vocab_to_int[word] = word_num
        word_num += 1

answers_vocab_to_int = {}

word_num = 0
for word, count in vocab.items():
    if count >= threshold:
        answers_vocab_to_int[word] = word_num
        word_num += i

codes = ['<PAD>','<EOS>','<UNK>','<GO>']

for code in codes:
    questions_vocab_to_int[code] = len(questions_vocab_to_int) + 1

for code in codes:
    answers_vocab_to_int[code] = len(answers_vocab_to_int) + 1

questions_int_to_vocab = {v_i: v for v, v_i in questions_vocab_to_int.items()}
answers_int_to_vocab = {v_i: v for v, v_i in answers_vocab_to_int.items()}

print(len(questions_vocab_to_int))
print(len(questions_int_to_vocab))
print(len(answers_vocab_to_int))
print(len(answers_int_to_vocab))

# add <EOS> after the end of sentence
for i in range(len(short_answers)):
    short_answers[i] += '<EOS>'

# convert text to int
# replace unknown words to <UNK>
questions_int = []
for question in short_questions:
    ints = []
    for word in question.split():
        if word not in questions_vocab_to_int:
            ints.append(questions_vocab_to_int['<UNK>'])
        else:
            ints.append(questions_vocab_to_int[word])
    questions_int.append(ints)

answers_int = []
for answer in short_answers:
    ints = []
    for word in answer.split():
        if word not in answers_vocab_to_int:
            ints.append(answers_vocab_to_int['<UNK>'])
        else:
            ints.append(answers_vocab_to_int[word])
    answers_int.append(ints)

print(len(questions_int))
print(len(answers_int))

word_count = 0
unk_count = 0

for question in questions_int:
    for word in question:
        if word == questions_vocab_to_int["<UNK>"]:
            unk_count += 1
        word_count += 1

for answer in answers_int:
    for word in answer:
        if word == answers_vocab_to_int["<UNK>"]:
            unk_count += 1
        word_count += 1

unk_ratio = round(unk_count/word_count,4)*100

print("Total number of words:", word_count)
print("Number of times <UNK> is used:", unk_count)
print("Percent of words that are <UNK>: {}%".format(round(unk_ratio,3)))

sorted_questions = []
sorted_answers = []

for length in range(1, max_line_length+1):
    for i in enumerate(questions_int):
        if len(i[1]) == length:
            sorted_questions.append(questions_int[i[0]])
            sorted_answers.append(answers_int[i[0]])

print(len(sorted_questions))
print(len(sorted_answers))
print()
for i in range(3):
    print(sorted_questions[i])
    print(sorted_answers[i])
    print()

def model_inputs():
    input_data = tf.placeholder(tf.int32, [None, None], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    lr = tf.placeholder(tf.float32, name='learning_rate')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    return input_data, targets, lr, keep_prob

def process_encoding_input(target_data, vocab_to_int, batch_size):
    ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
    dec_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<GO>']), ending], 1)

    return dec_input

def encoding_layer(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    drop = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
    enc_cell = tf.contrib.rnn.MultiRNNCell([drop] * num_layers)
    _, enc_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = enc_cell,
                                                   cell_bw = enc_cell,
                                                   sequence_length = sequence_length,
                                                   inputs = rnn_inputs,
                                                   dtype = tf.float32)
    return enc_state

def decoding_layer_train(encoder_state, dec_cell, dec_embed_input, sequence_length, decoding_scope, output_fn, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, dec_cell.output_size])
    att_keys, att_vals, att_score_fn, att_construct_fn = tf.contrib.seq2seq.prepare_attention(attention_states,
                                                                                              attention_option="bahdanau",
                                                                                              num_units=dec_cell.output_size)
    train_decoder_fn = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],
                                                                     att_keys,
                                                                     att_vals,
                                                                     att_score_fn,
                                                                     att_construct_fn,
                                                                     name="attn_dec_train")
    train_pred, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(dec_cell,
                                                              train_decoder_fn,
                                                              sequence_length,
                                                              scope=decoding_scope)
    train_pred_drop = tf.nn.dropout(train_pred, keep_prob)
    return output_fn(train_pred_drop)

def decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, start_of_sequence_id, end_of_sequence_id, maximum_length, vocab_size, decoding_scope, output_fn, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, dec_cell.output_size])
    att_keys, att_vals, att_score_fn, att_construct_fn = tf.contrib.seq2seq.prepare_attention(attention_states,
                                                                                              attention_option="bahdanau",
                                                                                              num_units=dec_cell.output_size)
    infer_decoder_fn = tf.contrib.seq2seq.attention_decoder_fn_inference(output_fn,
                                                                         encoder_state[0],
                                                                         att_keys,
                                                                         att_vals,
                                                                         att_score_fn,
                                                                         att_construct_fn,
                                                                         dec_embeddings,
                                                                         start_of_sequence_id,
                                                                         end_of_sequence_id,
                                                                         maximum_length,
                                                                         vocab_size,
                                                                         name = "att_dec_inf")
    infer_logits, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(dec_cell, infer_decoder_fn, scope=decoding_scope)
    return infer_logits

def decoding_layer(dec_embed_input, dec_embeddings, encoder_state, vocab_size, sequence_length, rnn_size, num_layers, vocab_to_int, keep_prob, batch_size):
    with tf.variable_scope("decoding") as decoding_scope:
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        drop = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
        dec_cell = tf.contrib.MultiRNNCell([drop] * num_layers)
        weights = tf.truncated_normal_initializer(stddev=0.1)
        biases = tf.zeros.initalizer()
        output_fn = lambda x: tf.contrib.layers.fully_connected(x,
                                                                vocab_size,
                                                                None,
                                                                scope=decoding_scope,
                                                                weights_initilizer = weights,
                                                                biases_initializer = biases)
        train_logits = decoding_layer_train(encoder_state,
                                            dec_cell,
                                            dec_embed_input,
                                            sequence_length,
                                            decoding_scope,
                                            output_fn,
                                            keep_prob,
                                            batch_size)
        decoding_scope.reuse_variables()
        infer_logits = decoding_layer_infer(encoder_state,
                                            dec_cell,
                                            dec_embeddings,
                                            vocab_to_int['<GO>'],
                                            vocab_to_int['<EOS>'],
                                            sequence_length - 1,
                                            vocab_size,
                                            decoding_scope,
                                            output_fn,
                                            keep_prob,
                                            batch_size)
        return train_logits, infer_logits

def seq2seq(input_data, target_data, keep_prob, batch_size, sequence_length, answers_vocab_size, questions_vocab_size, enc_embedding_size, dec_embedding_size, rnn_size, num_layers, questions_vocab_to_int):
    enc_embed_input = tf.contrib.alyers.embed_sequence(input_data,
                                                       answers_vocab_size+1,
                                                       enc_embedding_size,
                                                       initializer = tf.random_uniform_initializer(-1, 1))
    enc_state = encoding_layer(enc_embed_input, rnn_size, num_layers, keep_prob, sequence_length)
    dec_input = process_encoding_input(target_data, questions_vocab_to_int, batch_size)
    dec_embeddings = tf.Variable(tf.random_uniform([questions_vocab_size+1, dec_embedding_size], -1, 1))
    dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)
    train_logits, infer_logits = decoding_layer(dec_embed_input,
                                                dec_embeddings,
                                                enc_state,
                                                questions_vocab_size,
                                                sequence_length,
                                                rnn_size,
                                                num_layers,
                                                questions_vocab_to_int,
                                                keep_prob,
                                                batch_size)
    return train_logits, infer_logits
