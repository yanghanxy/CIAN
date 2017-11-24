import argparse
import numpy as np
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint

from model_builder import build_model, save_model_local, load_model_local
from utils import load_data, initial_logger, save_preds_matched_to_csv, save_preds_mismatched_to_csv, save_extrasentence

def train_model(opt, logger):
    logger.info('---START---')
    # initialize for reproduce
    np.random.seed(opt.seed)

    # load data
    logger.info('---LOAD DATA---')
    opt, training, training_snli, validation, test_matched, test_mismatched = load_data(opt)

    if not opt.skip_train:
        logger.info('---TRAIN MODEL---')
        for train_counter in range(opt.max_epochs):
            if train_counter == 0:
                model = build_model(opt)
            else:
                model = load_model_local(opt)
            np.random.seed(train_counter)
            lens = len(training_snli[-1])
            perm = np.random.permutation(lens)
            idx = perm[:int(lens * 0.2)]
            train_data = [np.concatenate((training[0], training_snli[0][idx])),
                          np.concatenate((training[1], training_snli[1][idx])),
                          np.concatenate((training[2], training_snli[2][idx]))]
            csv_logger = CSVLogger('{}{}.csv'.format(opt.log_dir, opt.model_name), append=True)
            cp_filepath = opt.save_dir + "cp-" + opt.model_name + "-" + str(train_counter) + "-{val_acc:.2f}.h5"
            cp = ModelCheckpoint(cp_filepath, monitor='val_acc', save_best_only=True, save_weights_only=True)
            callbacks = [cp, csv_logger]
            model.fit(train_data[:-1], train_data[-1], batch_size=opt.batch_size, epochs=1, validation_data=(validation[:-1], validation[-1]), callbacks=callbacks)
            save_model_local(opt, model)
    else:
        logger.info('---LOAD MODEL---')
        model = load_model_local(opt)

    # predict
    logger.info('---TEST MODEL---')
    preds_matched = model.predict(test_matched[:-1], batch_size=128, verbose=1)
    preds_mismatched = model.predict(test_mismatched[:-1], batch_size=128, verbose=1)

    save_preds_matched_to_csv(preds_matched, test_mismatched[-1], opt)
    save_preds_mismatched_to_csv(preds_mismatched, test_mismatched[-1], opt)

if __name__ == '__main__':
    # initialize
    parser = argparse.ArgumentParser(description='Train a word+character-level Textural Entailment model')
    parser.add_argument('--model_name', type=str, default='model_CNN_BILSTMDP_ATT', help='name of model')
    # data
    parser.add_argument('--data_dir', type=str, default='.//data//', help='data directory. Should contain train.txt/valid.txt/test.jsonl with input data')
    parser.add_argument('--log_dir', type=str, default='.//log//', help='log file directory')
    parser.add_argument('--alphabet', type=str, default="abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}", help='characters to be tokenized')
    parser.add_argument('--max_seq_len', type=int, default=50, help='number of timesteps to unroll for')
    parser.add_argument('--max_word_len', type=int, default=15, help='maximum word length')
    parser.add_argument('--vocab_word', type=int, default=10000, help='max number of words in model')
    parser.add_argument('--vocab_char', type=int, default=60, help='max number of char in model')
    # model params
    parser.add_argument('--use_char', default=False, help='use characters')
    parser.add_argument('--word_embed_size', type=int, default=300, help='dimensionality of word embeddings')
    parser.add_argument('--char_embed_size', type=int, default=15, help='dimensionality of character embeddings')
    parser.add_argument('--cnn_feature_maps', type=int, nargs='+', default=[50,100,150,200,200,200,200], help='number of feature maps in the CNN')
    parser.add_argument('--cnn_kernels', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6, 7], help='conv net kernel widths')
    parser.add_argument('--rnn_size', type=int, default=300, help='size of rnn internal state')
    parser.add_argument('--use_highway', default=False, help='size of rnn internal state')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout. 0 = no dropout')
    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='starting learning rate')
    parser.add_argument('--batch_size', type=int, default=1, help='number of sequences to train on in parallel')
    parser.add_argument('--max_epochs', type=int, default=10, help='number of full passes through the training data')
    parser.add_argument('--patience', type=int, default=4, help='early stopping after this epochs')
    parser.add_argument('--learning_rate_decay', type=float, default=0.5, help='learning rate decay')
    parser.add_argument('--decay_when', type=float, default=1, help='decay if validation perplexity does not improve by more than this much')
    parser.add_argument('--max_grad_norm', type=float, default=5, help='normalize gradients at')
    parser.add_argument('--activation', type=str, default='relu', help='activation funcation')
    # bookkeeping
    parser.add_argument('--seed', type=int, default=1337, help='manual random number generator seed')
    parser.add_argument('--save_dir', type=str, default='.//trained//', help='output directory where trained model get written')
    parser.add_argument('--skip_train', default=False, help='skip training', action='store_true')

    # parse input params and initial logger
    params = parser.parse_args()
    logger = initial_logger(params)

    # train and predict
    try:
        train_model(params, logger)
    except Exception as e:
        logger.exception(e)
    finally:
        # optional clean up code
        logger.info('---END---')
