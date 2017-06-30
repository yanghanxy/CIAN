from keras import backend as K
from keras.optimizers import Adam
from keras.layers import recurrent, Input, TimeDistributed, Dense, Dropout, Reshape, Concatenate, Conv2D, MaxPooling2D, Lambda, \
    merge
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import Bidirectional
from keras.layers.merge import Multiply
from keras.regularizers import l2
from keras.models import Model, Sequential, load_model
from keras.models import model_from_json

from model_library import Highway, AttentionWithContext

def save_model_local(opt, model):
    file_path = opt.save_dir + opt.model_name
    model.save_weights('{}.h5'.format(file_path))

def load_model_local(opt):
    file_path = opt.save_dir + opt.model_name
    model = build_model(opt)
    model.load_weights('{}.h5'.format(file_path))
    optm = Adam(lr=opt.learning_rate)
    model.compile(optimizer=optm, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_model(opt):
    print('Build model...')

    '''
    Input
    '''
    prem_inputs = Input(shape=(opt.max_seq_len, opt.max_word_len,), dtype='int32')
    hypo_inputs = Input(shape=(opt.max_seq_len, opt.max_word_len,), dtype='int32')

    '''
    CHARACTER EMBEDDING LAYER
    '''
    char_embedder = TimeDistributed(Embedding(opt.vocab_char, opt.char_embed_size, input_length=opt.max_word_len))
    prem_char_embed = char_embedder(prem_inputs)
    hypo_char_embed = char_embedder(hypo_inputs)

    prem_cnn_con = []
    hypo_cnn_con = []
    for feature_map, kernel in zip(opt.cnn_feature_maps, opt.cnn_kernels):
        reduced_l = opt.max_word_len - kernel + 1
        conv = Conv2D(feature_map, (1, kernel), activation='tanh', data_format="channels_last")
        maxp = MaxPooling2D(pool_size=(1, reduced_l), data_format="channels_last")
        prem_cnn_embed = maxp(conv(prem_char_embed))
        hypo_cnn_embed = maxp(conv(hypo_char_embed))
        prem_cnn_con.append(prem_cnn_embed)
        hypo_cnn_con.append(hypo_cnn_embed)
    prem_char_embed = Concatenate()(prem_cnn_con)
    hypo_char_embed = Concatenate()(hypo_cnn_con)
    reshape = Reshape((opt.max_seq_len, sum(opt.cnn_feature_maps)))
    prem_embed = reshape(prem_char_embed)
    hypo_embed = reshape(hypo_char_embed)
    prem_embed = BatchNormalization()(prem_embed)
    hypo_embed = BatchNormalization()(hypo_embed)

    '''
    HIGHWAY LAYER
    '''
    highway_first = TimeDistributed(Highway(activation=opt.activation))
    highway_second = TimeDistributed(Highway(activation=opt.activation))
    prem_embed = highway_first(prem_embed)
    prem_embed = highway_second(prem_embed)
    hypo_embed = highway_first(hypo_embed)
    hypo_embed = highway_second(hypo_embed)


    '''
    ENCODER LAYER
    '''
    encoder_forward = recurrent.LSTM(units=opt.rnn_size, dropout=opt.dropout, recurrent_dropout=opt.dropout, implementation=2, return_sequences=True)
    encoder_bacward = recurrent.LSTM(units=opt.rnn_size, dropout=opt.dropout, recurrent_dropout=opt.dropout, implementation=2, return_sequences=True, go_backwards=True)
    attention_forward = AttentionWithContext(W_dropout=opt.dropout)
    attention_bacward = AttentionWithContext(W_dropout=opt.dropout)

    prem_encoded_forward = encoder_forward(prem_embed)
    prem_encoded_forward = attention_forward(prem_encoded_forward)
    prem_encoded_bacward = encoder_bacward(prem_embed)
    prem_encoded_bacward = attention_bacward(prem_encoded_bacward)
    prem_encoded = Concatenate()([prem_encoded_forward, prem_encoded_bacward])
    prem_encoded = BatchNormalization()(prem_encoded)

    hypo_encoded_forward = encoder_forward(hypo_embed)
    hypo_encoded_forward = attention_forward(hypo_encoded_forward)
    hypo_encoded_bacward = encoder_bacward(hypo_embed)
    hypo_encoded_bacward = attention_bacward(hypo_encoded_bacward)
    hypo_encoded = Concatenate()([hypo_encoded_forward, hypo_encoded_bacward])
    hypo_encoded = BatchNormalization()(hypo_encoded)

    prem = Dropout(opt.dropout)(prem_encoded)
    hypo = Dropout(opt.dropout)(hypo_encoded)
    joint = Concatenate()([prem, hypo])

    '''
    ReLU LAYER
    '''
    for i in range(3):
        joint = Dense(2 * opt.rnn_size, activation=opt.activation, kernel_regularizer=l2(opt.learning_rate))(joint)
        joint = Dropout(opt.dropout)(joint)
        joint = BatchNormalization()(joint)

    '''
    3-WAY SOFTMAX
    '''
    pred = Dense(len(opt.labels), activation='softmax')(joint)

    model = Model(inputs=[prem_inputs, hypo_inputs], outputs=pred)
    optm = Adam(lr=opt.learning_rate)
    model.compile(optimizer=optm, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    return model
