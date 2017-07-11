import json
import os
import logging
import csv
import numpy as np
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

def save_extrasentence(preds, ids, opt):
    ids = np.array(ids)
    results = np.concatenate((np.expand_dims(ids, axis=1), preds), axis=1)
    fn_sentence = "{}{}_sentence.txt".format(opt.log_dir, opt.model_name)
    with open(fn_sentence, 'wb') as txtfile:
        for result in results:
            txtfile.write('{}\t'.format(result[0]))
            txtfile.write('p\t')
            for value in result[1:601]:
                txtfile.write('{0:.5f} '.format(float(value)))
            txtfile.write('\n')


def save_test_matched(preds, ids, opt):
    ids = np.array(ids)
    results = np.concatenate((np.expand_dims(ids, axis=1), preds), axis=1)
    fn_sentence = "{}{}_matched.txt".format(opt.log_dir, opt.model_name)
    with open(fn_sentence, 'wb') as txtfile:
        for result in results:
            txtfile.write('{}\t'.format(result[0]))
            txtfile.write('p\t')
            for value in result[1:601]:
                txtfile.write('{0:.5f} '.format(float(value)))
            txtfile.write('\n')
            txtfile.write('{}\t'.format(result[0]))
            txtfile.write('h\t')
            for value in result[602:1202]:
                txtfile.write('{0:.5f} '.format(float(value)))
            txtfile.write('\n')


def save_test_mismatched(preds, ids, opt):
    ids = np.array(ids)
    results = np.concatenate((np.expand_dims(ids, axis=1), preds), axis=1)
    fn_sentence = "{}{}_mismatched.txt".format(opt.log_dir, opt.model_name)
    with open(fn_sentence, 'wb') as txtfile:
        for result in results:
            txtfile.write('{}\t'.format(result[0]))
            txtfile.write('p\t')
            for value in result[1:601]:
                txtfile.write('{0:.5f} '.format(float(value)))
            txtfile.write('\n')
            txtfile.write('{}\t'.format(result[0]))
            txtfile.write('h\t')
            for value in result[602:1202]:
                txtfile.write('{0:.5f} '.format(float(value)))
            txtfile.write('\n')

def save_preds_matched_to_csv(preds, ids, opt):
    ids = np.array(ids)
    results = np.concatenate((np.expand_dims(ids, axis=1), preds), axis=1)

    LABELS_DIC = {0: 'contradiction', 1: 'neutral', 2: 'entailment'}
    labels = np.argmax(preds, axis=1)
    gold_labels = np.array([LABELS_DIC[label] for label in labels])
    uploads = np.concatenate((np.expand_dims(ids, axis=1), np.expand_dims(gold_labels, axis=1)), axis=1)

    fn_results = "{}{}_matched_results.csv".format(opt.log_dir, opt.model_name)
    fn_upload = "{}{}_matched_upload.csv".format(opt.log_dir, opt.model_name)
    with open(fn_results, 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        spamwriter.writerow(['pairID', 'contradiction', 'neutral', 'entailment'])
        for result in results:
            spamwriter.writerow([result[0], result[1], result[2], result[3]])
    with open(fn_upload, 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        spamwriter.writerow(['pairID', 'gold_label'])
        for upload in uploads:
            spamwriter.writerow([upload[0], upload[1]])

def save_preds_mismatched_to_csv(preds, ids, opt):
    ids = np.array(ids)
    results = np.concatenate((np.expand_dims(ids, axis=1), preds), axis=1)

    LABELS_DIC = {0: 'contradiction', 1: 'neutral', 2: 'entailment'}
    labels = np.argmax(preds, axis=1)
    gold_labels = np.array([LABELS_DIC[label] for label in labels])
    uploads = np.concatenate((np.expand_dims(ids, axis=1), np.expand_dims(gold_labels, axis=1)), axis=1)

    fn_results = "{}{}_mismatched_results.csv".format(opt.log_dir, opt.model_name)
    fn_upload = "{}{}_mismatched_upload.csv".format(opt.log_dir, opt.model_name)
    with open(fn_results, 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        spamwriter.writerow(['pairID', 'contradiction', 'neutral', 'entailment'])
        for result in results:
            spamwriter.writerow([result[0], result[1], result[2], result[3]])
    with open(fn_upload, 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        spamwriter.writerow(['pairID', 'gold_label'])
        for upload in uploads:
            spamwriter.writerow([upload[0], upload[1]])

def initial_logger(opt):
    # create logger
    logger = logging.getLogger(opt.model_name)
    logger.setLevel(logging.DEBUG)
    log_path = "{}{}.log".format(opt.log_dir, opt.model_name)
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fmt = "%(asctime)s | %(levelname)s | %(filename)s LINE-%(lineno)d | PROCESS-%(process)d | %(message)s"
    formatter = logging.Formatter(fmt)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

def extract_tokens_from_binary_parse(parse):
    return parse.replace('(', ' ').replace(')', ' ').replace('-LRB-', '(').replace('-RRB-', ')').split()

def yield_examples(fn, skip_no_majority=True, limit=None):
    for i, line in enumerate(open(fn)):
        if limit and i > limit:
            break
        data = json.loads(line)
        label = data['gold_label']
        s1 = ' '.join(extract_tokens_from_binary_parse(data['sentence1_binary_parse']))
        s2 = ' '.join(extract_tokens_from_binary_parse(data['sentence2_binary_parse']))
        if skip_no_majority and label == '-':
            continue
        yield (label, s1, s2)

def yield_examples_test(fn, skip_no_majority=True, limit=None):
    for i, line in enumerate(open(fn)):
        if limit and i > limit:
            break
        data = json.loads(line)
        label = data['pairID']
        s1 = ' '.join(extract_tokens_from_binary_parse(data['sentence1_binary_parse']))
        s2 = ' '.join(extract_tokens_from_binary_parse(data['sentence2_binary_parse']))
        yield (label, s1, s2)

def get_data(fn, limit=None):
    raw_data = list(yield_examples(fn=fn, limit=limit))
    left = [s1 for _, s1, s2 in raw_data]
    right = [s2 for _, s1, s2 in raw_data]

    LABELS = {'contradiction': 0, 'neutral': 1, 'entailment': 2}
    Y = np.array([LABELS[label] for label, s1, s2 in raw_data])
    Y = np_utils.to_categorical(Y, len(LABELS))

    return left, right, Y

def get_test_data(fn, limit=None):
    raw_data = list(yield_examples_test(fn=fn, limit=limit))
    left = [s1 for _, s1, s2 in raw_data]
    right = [s2 for _, s1, s2 in raw_data]
    ids = [idx for idx, _, _ in raw_data]
    return left, right, ids

def load_data(opt):

    training = get_data('{}multinli_0.9_train.jsonl'.format(opt.data_dir))
    training_snli = get_data('{}snli_1.0_train.jsonl'.format(opt.data_dir))
    validation = get_data('{}multinli_0.9_dev_matched.jsonl'.format(opt.data_dir))
    test_matched = get_test_data('{}multinli_0.9_test_matched.jsonl'.format(opt.data_dir))
    test_mismatched = get_test_data('{}multinli_0.9_test_mismatched.jsonl'.format(opt.data_dir))

    opt.labels = {'contradiction': 0, 'neutral': 1, 'entailment': 2}
    print('finsh load data')

    tokenizer_char = Tokenizer(lower=True, filters='')
    chars = sorted(list(set(opt.alphabet)))
    tokenizer_char.fit_on_texts(chars)
    # Lowest index from the tokenizer is 1 - we need to include 0 in our vocab count
    opt.vocab_char = len(tokenizer_char.word_counts) + 1
    print('finish tokenization')

    if not (os.path.exists(opt.data_dir + 'precomputed_training_char.npy') or
                os.path.exists(opt.data_dir + 'precomputed_validation_char.npy') or
                os.path.exists(opt.data_dir + 'precomputed_test_matched_char.npy') or
                os.path.exists(opt.data_dir + 'precomputed_test_mismatched_char.npy')):
        flatten_char = lambda list_char: [item for sublist in list_char for item in sublist]
        pick_words = lambda seq: [i for i in seq.split(' ') if i]
        def word_to_char(str_text):
            chars_array = np.zeros((len(str_text), opt.max_seq_len, opt.max_word_len), dtype='int32')
            for idx_seq, str_sequence in enumerate(str_text):
                list_str_sequence = pick_words(str_sequence)
                trunc = list_str_sequence[-opt.max_seq_len:]
                str_seq_maxlen = ["" for x in range(opt.max_seq_len)]
                str_seq_maxlen[:len(trunc)] = trunc
                for idx_word, str_word in enumerate(str_seq_maxlen):
                    if str_word == '':
                        chars_array[idx_seq][idx_word] = np.zeros((opt.max_word_len), dtype='int32')
                    else:
                        str_char = list(str_word)
                        tok_char = tokenizer_char.texts_to_sequences(str_char)
                        tok_list = flatten_char(tok_char)
                        chars_array[idx_seq][idx_word] = pad_sequences([tok_list], maxlen=opt.max_word_len, padding='post')[0]
            return chars_array
        prepare_data_char = lambda data: np.concatenate((np.expand_dims(word_to_char(data[0]), axis=0), np.expand_dims(word_to_char(data[1]), axis=0)), axis=0)
        training_char = prepare_data_char(training)
        training_snli_char = prepare_data_char(training_snli)
        validation_char = prepare_data_char(validation)
        test_matched_char = prepare_data_char(test_matched)
        test_mismatched_char = prepare_data_char(test_mismatched)

        np.save(opt.data_dir + 'precomputed_training_char', training_char)
        np.save(opt.data_dir + 'precomputed_training_snli_char', training_snli_char)
        np.save(opt.data_dir + 'precomputed_validation_char', validation_char)
        np.save(opt.data_dir + 'precomputed_test_matched_char', test_matched_char)
        np.save(opt.data_dir + 'precomputed_test_mismatched_char', test_mismatched_char)
    else:
        training_char = np.load(opt.data_dir + 'precomputed_training_char.npy')
        training_snli_char = np.load(opt.data_dir + 'precomputed_training_snli_char.npy')
        validation_char = np.load(opt.data_dir + 'precomputed_validation_char.npy')
        test_matched_char = np.load(opt.data_dir + 'precomputed_test_matched_char.npy')
        test_mismatched_char = np.load(opt.data_dir + 'precomputed_test_mismatched_char.npy')

    training_data = [training_char[0], training_char[1], training[2]]
    training_snli_data = [training_snli_char[0], training_snli_char[1], training_snli[2]]
    validation_data = [validation_char[0], validation_char[1], validation[2]]
    test_matched_data = [test_matched_char[0], test_matched_char[1], test_matched[2]]
    test_mismatched_data = [test_mismatched_char[0], test_mismatched_char[1], test_mismatched[2]]

    print('finish prepare training, validation, test data')
    return opt, training_data, training_snli_data, validation_data, test_matched_data, test_mismatched_data
