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

short_questions_temp = []
short_answers_temp = []

i = 0
for question in clean_questions:
    if len(question.split()) >= min_line_length and len(question.split()) <= max_line_length:
        short_questions_temp.append(question)
        short_answers_temp.append(clean_answers[i])
    i += 1

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
