# -*- coding: utf-8 -*-
"""

This Note book is adapted from [WuJiaocan's Github](https://github.com/WuJiaocan/tensorflow)

## NOTE: 

To Use This .py, need to also load following files:

1. test.en
2. test.fr
3. vocab.en
4. vocab.fr

"""

import tensorflow as tf
import codecs
import sys
import numpy as np
from nltk.translate.bleu_score import sentence_bleu


class NMTModel(object):

    def __init__(self):
            
        # Model HyperParameters, need to keep consistent with trained model
        self.HIDDEN_SIZE = 256
        self.DECODER_LAYERS = 2
        self.SRC_VOCAB_SIZE = 40396
        self.TRG_VOCAB_SIZE = 49303
        SHARE_EMB_AND_SOFTMAX = True

        # ID for <sos> and <eos> in vocab
        self.SOS_ID = 1
        self.EOS_ID = 2
        
        # Define RNN CELL for Encoder and Decoder
        self.enc_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(self.HIDDEN_SIZE)
        self.enc_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(self.HIDDEN_SIZE)
        self.dec_cell = tf.nn.rnn_cell.MultiRNNCell(
          [tf.nn.rnn_cell.BasicLSTMCell(self.HIDDEN_SIZE) 
           for _ in range(self.DECODER_LAYERS)])

        # Define EMBEDDING LAYER for Source/Target Language
        self.src_embedding = tf.get_variable(
            "src_emb", [self.SRC_VOCAB_SIZE, self.HIDDEN_SIZE])
        self.trg_embedding = tf.get_variable(
            "trg_emb", [self.TRG_VOCAB_SIZE, self.HIDDEN_SIZE])

        # Define SOFTMAX LAYER (share with Embedding or not share)
        if SHARE_EMB_AND_SOFTMAX:
            self.softmax_weight = tf.transpose(self.trg_embedding)
        else:
            self.softmax_weight = tf.get_variable(
               "weight", [self.HIDDEN_SIZE, self.TRG_VOCAB_SIZE])
        self.softmax_bias = tf.get_variable(
            "softmax_bias", [self.TRG_VOCAB_SIZE])

    def inference(self, src_input):
        # As dynamic rnn requires batch input, 
        # to infer one single sentence, we set batch = 1
        src_size = tf.convert_to_tensor([len(src_input)], dtype=tf.int32)
        src_input = tf.convert_to_tensor([src_input], dtype=tf.int32)
        src_emb = tf.nn.embedding_lookup(self.src_embedding, src_input)

        with tf.variable_scope("encoder"):
            # Use bidirectional_dynamic_rnn for encoder as in training step
            enc_outputs, enc_state = tf.nn.bidirectional_dynamic_rnn(
                self.enc_cell_fw, self.enc_cell_bw, src_emb, src_size, 
                dtype=tf.float32)
            # concatenate two outputs of LSTM cell as 1 tensor
            # enc_outputs = (output_fw, output_bw)
            enc_outputs = tf.concat([enc_outputs[0], enc_outputs[1]], -1)    
        
        with tf.variable_scope("decoder"):
            # define ATTENTION MECHANISM for decoder
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                                    self.HIDDEN_SIZE, enc_outputs,
                                    memory_sequence_length=src_size)

            # wrap decoder cell and attention mechanism as an ATTENTION CELL
            attention_cell = tf.contrib.seq2seq.AttentionWrapper(
                                    self.dec_cell, attention_mechanism,
                                    attention_layer_size=self.HIDDEN_SIZE)
   
        # Set the max step for decoder
        MAX_DEC_LEN=100

        with tf.variable_scope("decoder/rnn/attention_wrapper"):
            # use TensorArray store produced sentence
            init_array = tf.TensorArray(dtype=tf.int32, size=0,
                                        dynamic_size=True, clear_after_read=False)

            # start input for decoder with <sos> tag
            init_array = init_array.write(0, self.SOS_ID)

            # initilize attention cell
            init_loop_var = (
                attention_cell.zero_state(batch_size=1, dtype=tf.float32),
                init_array, 0)

            # condition for tf.while_loop：
            # loop until decoder outputs <eos> or hit max length setting
            def continue_loop_condition(state, trg_ids, step):
                return tf.reduce_all(tf.logical_and(
                    tf.not_equal(trg_ids.read(step), self.EOS_ID),
                    tf.less(step, MAX_DEC_LEN-1)))

            def loop_body(state, trg_ids, step):
                # read last step output as input for attention cell
                trg_input = [trg_ids.read(step)]
                trg_emb = tf.nn.embedding_lookup(self.trg_embedding,
                                                 trg_input)
                # FORWARD calc ATTENTION CELL and output decoder outputs/state
                dec_outputs, next_state = attention_cell.call(
                    state=state, inputs=trg_emb)
       
                # calc logit for each candidate target vocab
                # select the max as output
                output = tf.reshape(dec_outputs, [-1, self.HIDDEN_SIZE])
                logits = (tf.matmul(output, self.softmax_weight)
                          + self.softmax_bias)
                next_id = tf.argmax(logits, axis=1, output_type=tf.int32)
                # write output word into trg_ids
                trg_ids = trg_ids.write(step+1, next_id[0])
                return next_state, trg_ids, step+1

            # execute tf.while_loop return trg_ides
            state, trg_ids, step = tf.while_loop(
                continue_loop_condition, loop_body, init_loop_var)
            return trg_ids.stack()


def levenshtein(seq1, seq2):  
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y))
    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix[x, y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix[x, y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1] + 1,
                    matrix[x, y-1] + 1
                )

    return matrix[size_x - 1, size_y - 1]


def translate_en_fr(src_sent):
    
    # read checkpoint path, number indicates the latest step
    CHECKPOINT_PATH = "INFO7374-12200"
       
    tf.reset_default_graph()
        
    # define the trained model
    with tf.variable_scope("nmt_model", reuse=None):
        model = NMTModel()

    # sentence for testing
    test_en_text = src_sent
    
    # file for vocab
    SRC_VOCAB = "vocab.en"
    TRG_VOCAB = "vocab.fr"
    
    # convert sentence to word_index according to vocab
    with codecs.open(SRC_VOCAB, "r", "utf-8") as f_vocab:
        src_vocab = [w.strip() for w in f_vocab.readlines()]
        src_id_dict = dict((src_vocab[x], x) for x in range(len(src_vocab)))
    test_en_ids = [(src_id_dict[token] if token in src_id_dict else src_id_dict['<unk>'])
                   for token in test_en_text.split()]

    # build inference based on saved model weights
    output_op = model.inference(test_en_ids)
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, CHECKPOINT_PATH)

    # read translation output
    output_ids = sess.run(output_op)
    
    # convert translation idx into word
    with codecs.open(TRG_VOCAB, "r", "utf-8") as f_vocab:
        trg_vocab = [w.strip() for w in f_vocab.readlines()]
    output_text = ' '.join([trg_vocab[x] for x in output_ids])
    
    # output translation
    final_output_text = output_text.encode('utf8').decode(sys.stdout.encoding).strip('<eos>')
    
    # load test_set - size: 100

    src_test = []
    with open('test.en', 'r', encoding='utf-8') as f:
        for line in f:
            src_test.append(line.strip())

    tgt_test = []
    with open('test.fr', 'r', encoding='utf-8') as f:
        for line in f:
            tgt_test.append(line.strip())
        
    if src_sent in src_test:
        idx = src_test.index(src_sent)
        trgt_sent = tgt_test[idx]
        bleu = sentence_bleu(trgt_sent, final_output_text)
        lst = levenshtein(trgt_sent, final_output_text)
    else:
        trgt_sent = 'Not Available In App Test Set'
        bleu = 'NA'
        lst = 'NA'
    
    return output_text[6:-7], trgt_sent, bleu, lst
    sess.close()


""" 
## CALL THE FUNCTION 

"""

#output, tgt_sent, bleu, lst = translate_en_fr("morphological variations of mandibular architecture of mammals")
#print(output)
#print(tgt_sent)
#print(bleu)
#print(lst)