# coding=utf-8
import nltk
import numpy as np

def chinese_process(filein, fileout):
    with open(filein, 'r') as infile:
        with open(fileout, 'w') as outfile:
            for line in infile:
                output = list()
                line = nltk.word_tokenize(line)[0]
                for char in line:
                    output.append(char)
                    output.append(' ')
                output.append('\n')
                output = ''.join(output)
                outfile.write(output)


def text_to_code(tokens, dictionary, seq_len):
    code_str = ""
    eof_code = len(dictionary)
    for sentence in tokens:
        index = 0
        for word in sentence:
            code_str += (str(dictionary[word]) + ' ')
            index += 1
        while index < seq_len:
            code_str += (str(eof_code) + ' ')
            index += 1
        code_str += '\n'
    return code_str

def text_to_code_with_labels(tokens, dictionary, seq_len, labels):
    code_str = ""
    eof_code = len(dictionary)
    print('eof_code: ', eof_code)
    for label_index, sentence in enumerate(tokens):
        index = 0
        for word in sentence:
            code_str += (str(dictionary[word]) + ' ')
            index += 1
        while index < seq_len:
            code_str += (str(eof_code) + ' ')
            index += 1
        code_str += (str(labels[label_index]))
        code_str += '\n'

    return code_str


def code_to_text(codes, dictionary):
    paras = ""
    eof_code = len(dictionary)
    for line in codes:
        numbers = map(int, line)
        for number in numbers:
            if number == eof_code:
                continue
            paras += (dictionary[str(number)] + ' ')
        paras += '\n'
    return paras

def code_to_text_with_labels(codes, dictionary, labels):
    paras = ""
    eof_code = len(dictionary)

    print('eof_code: ', eof_code)
    for idx, line in enumerate(codes):
        label = labels[idx]
        numbers = map(int, line)
        numbers = list(numbers)
        for i,number in enumerate(numbers):
            if number == eof_code:
                continue
            paras += (dictionary[str(number)] + ' ') if i < len(numbers)-1 else (dictionary[str(number)])
        paras += (',' + str(label))
        paras += '\n'
    return paras


def get_tokenlized(file):
    tokenlized = list()
    with open(file) as raw:
        for text in raw:
            text = nltk.word_tokenize(text.lower())
            tokenlized.append(text)
    return tokenlized

def get_csv_tokenlized(file):
    tokenlized = list()
    labels = list()

    with open(file) as raw_csv:
        for text in raw_csv:
            text = text.strip().rpartition(',')
            label = text[-1]
            text = text[0]
            labels.append(label)
            text = nltk.word_tokenize(text.lower())
            tokenlized.append(text)
    return tokenlized, labels

def parse_generator_file(file):
    tokenlized = list()
    labels = list()
    with open(file) as generated_seq:
        for text in generated_seq:
            text = text.strip()
            label = text[-1]
            text = text[:-1].strip()
            labels.append(label)
            text = nltk.word_tokenize(text.lower())
            tokenlized.append(text)
    return tokenlized, labels   # tokenlized = list of list of sequence codes; labels = list of label codes

def get_word_list(tokens):
    word_set = list()
    for sentence in tokens:
        for word in sentence:
            word_set.append(word)
    return list(set(word_set))

def get_word_list_conditional(tokens, labels):
    word_set = list()
    label_words_dict = {}

    for index, sentence in enumerate(tokens):
        label = labels[index]
        if (label not in label_words_dict):
            label_words_dict[label] = set()
        for word in sentence:
            label_words_dict[label].add(word)
            word_set.append(word)
    return list(set(word_set)), label_words_dict

def get_dict(word_set):
    word_index_dict = dict()
    index_word_dict = dict()
    index = 0
    for word in word_set:
        word_index_dict[word] = str(index)
        index_word_dict[str(index)] = word
        index += 1
    return word_index_dict, index_word_dict

def text_precess(train_text_loc, test_text_loc=None):
    train_tokens = get_tokenlized(train_text_loc)
    if test_text_loc is None:
        test_tokens = list()
    else:
        test_tokens = get_tokenlized(test_text_loc)
    word_set = get_word_list(train_tokens + test_tokens)
    [word_index_dict, index_word_dict] = get_dict(word_set)

    if test_text_loc is None:
        sequence_len = len(max(train_tokens, key=len))
    else:
        sequence_len = max(len(max(train_tokens, key=len)), len(max(test_tokens, key=len)))
    with open('save/eval_data_g1d1.txt', 'w') as outfile:
        outfile.write(text_to_code(test_tokens, word_index_dict, sequence_len))

    return sequence_len, len(word_index_dict) + 1

#KratikaA: added to process labelled input real data stored in the csv format
def csv_text_precess(train_text_loc, test_text_loc=None):
    train_tokens, train_labels = get_csv_tokenlized(train_text_loc) # train_tokens is a list of list, train_labels is a list
    unique_labels = list(set(train_labels)) # get all unique labels
    if test_text_loc is None:
        test_tokens = list()
    else:
        test_tokens, test_labels = get_csv_tokenlized(test_text_loc)
    word_set, label_words_dict = get_word_list_conditional(train_tokens, train_labels)
    [word_index_dict, index_word_dict] = get_dict(word_set)

    if test_text_loc is None:
        sequence_len = len(max(train_tokens, key=len))
    else:
        sequence_len = max(len(max(train_tokens, key=len)), len(max(test_tokens, key=len)))
    with open('save/eval_data_g1d1.txt', 'w') as outfile:
        outfile.write(text_to_code_with_labels(train_tokens, word_index_dict, sequence_len, train_labels)) #, label_index_dict))

    return sequence_len, len(word_index_dict) + 1, train_labels
