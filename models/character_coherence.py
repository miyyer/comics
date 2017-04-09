import theano, cPickle, lasagne, random, csv, gzip, time, argparse, sys
import numpy as np
import h5py as h5
import theano.tensor as T   
from collections import Counter
from preprocess.character_coherence_minibatching import *
from layers_and_utils import *

# compute accuracy over a fold
def validate(fold_name, fold_data, fold_file, val_batch_size=1024):
    batches = [(x, x + val_batch_size) for x in range(0, len(fold_data[0]), val_batch_size)]
    correct = 0.
    total = 0.
    for start, end in batches:
        for batch in generate_minibatches_from_megabatch(fold_data, vdict, start, end, fold_dict=read_fold(fold_file, vdict), shuffle_candidates=True):
            prods = pred_fn(*batch[1:-1])
            labels = np.argmax(batch[-1], axis=-1)
            max_prods = np.argmax(prods, axis=-1)

            for i in range(prods.shape[0]):
                if max_prods[i] == labels[i]:
                    correct += 1
                
                total += 1

    return 'fold %s: got %d out of %d correct for %f accuracy' % (fold_name, correct, total, correct/total)


'''NETWORK ASSEMBLY'''
def build_text_only_network(d_word, d_hidden, lr, eps=1e-6):
    # input theano vars
    in_context_fc7 = T.tensor3(name='context_images')
    in_context_bb = T.tensor4(name='context_bb')
    in_bbmask = T.tensor3(name='bounding_box_mask')
    in_context = T.itensor4(name='context')
    in_cmask = T.tensor4(name='context_mask')
    in_answer_fc7 = T.matrix(name='answer_images')
    in_answer_bb = T.tensor3(name='answer_bb')
    in_ans1 = T.itensor3(name='answers')
    in_amask1 = T.tensor3(name='answer_mask')
    in_ans2 = T.itensor3(name='answers')
    in_amask2 = T.tensor3(name='answer_mask')
    in_labels = T.imatrix(name='labels')

    # define network
    l_context = lasagne.layers.InputLayer(shape=(None, max_panels, max_boxes, max_words), 
        input_var=in_context)
    l_ans1 = lasagne.layers.InputLayer(shape=(None, 2, max_words), input_var=in_ans1)
    l_ans2 = lasagne.layers.InputLayer(shape=(None, 2, max_words), input_var=in_ans2)

    l_cmask = lasagne.layers.InputLayer(shape=l_context.shape, input_var=in_cmask)
    l_amask1 = lasagne.layers.InputLayer(shape=l_ans1.shape, input_var=in_amask1)
    l_amask2 = lasagne.layers.InputLayer(shape=l_ans2.shape, input_var=in_amask2)
    l_bbmask = lasagne.layers.InputLayer(shape=(None, 3, max_boxes), input_var=in_bbmask)

    # contexts and answers should share embeddings
    l_context_emb = lasagne.layers.EmbeddingLayer(l_context, len_voc, 
        d_word, name='word_emb')
    l_ans1_emb = lasagne.layers.EmbeddingLayer(l_ans1, len_voc, 
        d_word, W=l_context_emb.W)
    l_ans2_emb = lasagne.layers.EmbeddingLayer(l_ans2, len_voc, 
        d_word, W=l_context_emb.W)

    l_context_box_reps = SumAverageLayer([l_context_emb, l_cmask], compute_sum=True, num_dims=4)
    l_box_reshape = lasagne.layers.ReshapeLayer(l_context_box_reps, (-1, max_boxes, d_word))
    l_bbmask_reshape = lasagne.layers.ReshapeLayer(l_bbmask, (-1, max_boxes))
    l_box_lstm = lasagne.layers.LSTMLayer(l_box_reshape, num_units=d_word, mask_input=l_bbmask_reshape, only_return_final=True)
    l_context_panel_reps = lasagne.layers.ReshapeLayer(l_box_lstm, (-1, 3, d_word))
    l_context_final_reps = lasagne.layers.LSTMLayer(l_context_panel_reps, num_units=d_hidden, only_return_final=True)

    l_ans1_reps = SumAverageLayer([l_ans1_emb, l_amask1], compute_sum=True, num_dims=3)
    l_ans1_panel_reps = lasagne.layers.LSTMLayer(l_ans1_reps, num_units=d_hidden, only_return_final=True,
          ingate=lasagne.layers.Gate(W_in=l_box_lstm.W_in_to_ingate,
                                     W_hid=l_box_lstm.W_hid_to_ingate,
                                     W_cell=l_box_lstm.W_cell_to_ingate,
                                     b=l_box_lstm.b_ingate),
          outgate=lasagne.layers.Gate(W_in=l_box_lstm.W_in_to_outgate,
                                     W_hid=l_box_lstm.W_hid_to_outgate,
                                     W_cell=l_box_lstm.W_cell_to_outgate,
                                     b=l_box_lstm.b_outgate),
           forgetgate=lasagne.layers.Gate(W_in=l_box_lstm.W_in_to_forgetgate,
                                     W_hid=l_box_lstm.W_hid_to_forgetgate,
                                     W_cell=l_box_lstm.W_cell_to_forgetgate,
                                     b=l_box_lstm.b_forgetgate),
           cell=lasagne.layers.Gate(W_in=l_box_lstm.W_in_to_cell,
                                     W_hid=l_box_lstm.W_hid_to_cell,
                                     W_cell=None,
                                     b=l_box_lstm.b_cell) ) 

    l_ans2_reps = SumAverageLayer([l_ans2_emb, l_amask2], compute_sum=True, num_dims=3)
    l_ans2_panel_reps = lasagne.layers.LSTMLayer(l_ans2_reps, num_units=d_hidden, only_return_final=True,
          ingate=lasagne.layers.Gate(W_in=l_box_lstm.W_in_to_ingate,
                                     W_hid=l_box_lstm.W_hid_to_ingate,
                                     W_cell=l_box_lstm.W_cell_to_ingate,
                                     b=l_box_lstm.b_ingate),
          outgate=lasagne.layers.Gate(W_in=l_box_lstm.W_in_to_outgate,
                                     W_hid=l_box_lstm.W_hid_to_outgate,
                                     W_cell=l_box_lstm.W_cell_to_outgate,
                                     b=l_box_lstm.b_outgate),
           forgetgate=lasagne.layers.Gate(W_in=l_box_lstm.W_in_to_forgetgate,
                                     W_hid=l_box_lstm.W_hid_to_forgetgate,
                                     W_cell=l_box_lstm.W_cell_to_forgetgate,
                                     b=l_box_lstm.b_forgetgate),
           cell=lasagne.layers.Gate(W_in=l_box_lstm.W_in_to_cell,
                                     W_hid=l_box_lstm.W_hid_to_cell,
                                     W_cell=None,
                                     b=l_box_lstm.b_cell) ) 

    l_scores1 = InnerProductLayer([l_context_final_reps, l_ans1_panel_reps], is_cc=True)
    l_scores2 = InnerProductLayer([l_context_final_reps, l_ans2_panel_reps], is_cc=True)
    l_scores = lasagne.layers.concat([l_scores1, l_scores2], axis=-1)
    l_scores = lasagne.layers.NonlinearityLayer(l_scores, nonlinearity=lasagne.nonlinearities.softmax)

    preds = lasagne.layers.get_output(l_scores)
    loss = T.mean(lasagne.objectives.categorical_crossentropy(preds, in_labels))

    all_params = lasagne.layers.get_all_params(l_scores, trainable=True)
    updates = lasagne.updates.adam(loss, all_params, learning_rate=lr)
    train_fn = theano.function([in_context_fc7, in_context_bb, in_bbmask, in_context, in_cmask, 
        in_answer_fc7, in_answer_bb, in_ans1, in_amask1, in_ans2, in_amask2, in_labels], 
        loss, updates=updates, on_unused_input='warn')
    pred_fn = theano.function([in_context_fc7, in_context_bb, in_bbmask, in_context, in_cmask, 
        in_answer_fc7, in_answer_bb, in_ans1, in_amask1, in_ans2, in_amask2], 
        preds, on_unused_input='warn')
    return train_fn, pred_fn, l_scores


def build_image_only_network(d_word, d_hidden, lr, eps=1e-6):

    # input theano vars
    in_context_fc7 = T.tensor3(name='context_images')
    in_context_bb = T.tensor4(name='context_bb')
    in_bbmask = T.tensor3(name='bounding_box_mask')
    in_context = T.itensor4(name='context')
    in_cmask = T.tensor4(name='context_mask')
    in_answer_fc7 = T.matrix(name='answer_images')
    in_answer_bb = T.tensor3(name='answer_bb')
    in_ans1 = T.itensor3(name='answers')
    in_amask1 = T.tensor3(name='answer_mask')
    in_ans2 = T.itensor3(name='answers')
    in_amask2 = T.tensor3(name='answer_mask')
    in_labels = T.imatrix(name='labels')

    # define network
    l_context_fc7 = lasagne.layers.InputLayer(shape=(None, 3, 4096), input_var=in_context_fc7)
    l_answer_fc7 = lasagne.layers.InputLayer(shape=(None, 4096), input_var=in_answer_fc7)

    l_ans1 = lasagne.layers.InputLayer(shape=(None, 2, max_words), input_var=in_ans1)
    l_ans2 = lasagne.layers.InputLayer(shape=(None, 2, max_words), input_var=in_ans2)

    l_amask1 = lasagne.layers.InputLayer(shape=l_ans1.shape, input_var=in_amask1)
    l_amask2 = lasagne.layers.InputLayer(shape=l_ans2.shape, input_var=in_amask2)
    l_bbmask = lasagne.layers.InputLayer(shape=(None, 3, max_boxes), input_var=in_bbmask)

    # contexts and answers should share embeddings
    l_ans1_emb = lasagne.layers.EmbeddingLayer(l_ans1, len_voc, 
        d_word)
    l_ans2_emb = lasagne.layers.EmbeddingLayer(l_ans2, len_voc, 
        d_word, W=l_ans1_emb.W)

    l_context_proj = lasagne.layers.DenseLayer(l_context_fc7, num_units=d_word, nonlinearity=lasagne.nonlinearities.rectify, num_leading_axes=2)
    l_context_final_reps = lasagne.layers.LSTMLayer(l_context_proj, num_units=d_hidden, only_return_final=True)

    l_ans1_reps = SumAverageLayer([l_ans1_emb, l_amask1], compute_sum=True, num_dims=3)
    l_ans1_panel_reps = lasagne.layers.LSTMLayer(l_ans1_reps, num_units=d_word, only_return_final=True) 
    l_ans1_concat = MyConcatLayer([l_ans1_panel_reps, l_answer_fc7], axis=-1)
    l_ans1_proj = lasagne.layers.DenseLayer(l_ans1_concat, num_units=d_hidden, nonlinearity=lasagne.nonlinearities.rectify)

    l_ans2_reps = SumAverageLayer([l_ans2_emb, l_amask2], compute_sum=True, num_dims=3)
    l_ans2_panel_reps = lasagne.layers.LSTMLayer(l_ans2_reps, num_units=d_word, only_return_final=True,
          ingate=lasagne.layers.Gate(W_in=l_ans1_panel_reps.W_in_to_ingate,
                                     W_hid=l_ans1_panel_reps.W_hid_to_ingate,
                                     W_cell=l_ans1_panel_reps.W_cell_to_ingate,
                                     b=l_ans1_panel_reps.b_ingate),
          outgate=lasagne.layers.Gate(W_in=l_ans1_panel_reps.W_in_to_outgate,
                                     W_hid=l_ans1_panel_reps.W_hid_to_outgate,
                                     W_cell=l_ans1_panel_reps.W_cell_to_outgate,
                                     b=l_ans1_panel_reps.b_outgate),
           forgetgate=lasagne.layers.Gate(W_in=l_ans1_panel_reps.W_in_to_forgetgate,
                                     W_hid=l_ans1_panel_reps.W_hid_to_forgetgate,
                                     W_cell=l_ans1_panel_reps.W_cell_to_forgetgate,
                                     b=l_ans1_panel_reps.b_forgetgate),
           cell=lasagne.layers.Gate(W_in=l_ans1_panel_reps.W_in_to_cell,
                                     W_hid=l_ans1_panel_reps.W_hid_to_cell,
                                     W_cell=None,
                                     b=l_ans1_panel_reps.b_cell) ) 
    l_ans2_concat = MyConcatLayer([l_ans2_panel_reps, l_answer_fc7], axis=-1)
    l_ans2_proj = lasagne.layers.DenseLayer(l_ans2_concat, num_units=d_hidden, nonlinearity=lasagne.nonlinearities.rectify, 
        W=l_ans1_proj.W, b=l_ans1_proj.b)

    l_scores1 = InnerProductLayer([l_context_final_reps, l_ans1_proj], is_cc=True)
    l_scores2 = InnerProductLayer([l_context_final_reps, l_ans2_proj], is_cc=True)
    l_scores = lasagne.layers.concat([l_scores1, l_scores2], axis=-1)
    l_scores = lasagne.layers.NonlinearityLayer(l_scores, nonlinearity=lasagne.nonlinearities.softmax)

    preds = lasagne.layers.get_output(l_scores)
    loss = T.mean(lasagne.objectives.categorical_crossentropy(preds, in_labels))

    all_params = lasagne.layers.get_all_params(l_scores, trainable=True)
    updates = lasagne.updates.adam(loss, all_params, learning_rate=lr)
    train_fn = theano.function([in_context_fc7, in_context_bb, in_bbmask, in_context, in_cmask, 
        in_answer_fc7, in_answer_bb, in_ans1, in_amask1, in_ans2, in_amask2, in_labels], 
        loss, updates=updates, on_unused_input='warn')
    pred_fn = theano.function([in_context_fc7, in_context_bb, in_bbmask, in_context, in_cmask, 
        in_answer_fc7, in_answer_bb, in_ans1, in_amask1, in_ans2, in_amask2], 
        preds, on_unused_input='warn')
    return train_fn, pred_fn, l_scores


def build_image_text_network(d_word, d_hidden, lr, eps=1e-6):

    # input theano vars
    in_context_fc7 = T.tensor3(name='context_images')
    in_context_bb = T.tensor4(name='context_bb')
    in_bbmask = T.tensor3(name='bounding_box_mask')
    in_context = T.itensor4(name='context')
    in_cmask = T.tensor4(name='context_mask')
    in_answer_fc7 = T.matrix(name='answer_images')
    in_answer_bb = T.tensor3(name='answer_bb')
    in_ans1 = T.itensor3(name='answers')
    in_amask1 = T.tensor3(name='answer_mask')
    in_ans2 = T.itensor3(name='answers')
    in_amask2 = T.tensor3(name='answer_mask')
    in_labels = T.imatrix(name='labels')

    # define network
    l_context_fc7 = lasagne.layers.InputLayer(shape=(None, 3, 4096), input_var=in_context_fc7)
    l_answer_fc7 = lasagne.layers.InputLayer(shape=(None, 4096), input_var=in_answer_fc7)

    l_context = lasagne.layers.InputLayer(shape=(None, max_panels, max_boxes, max_words), 
        input_var=in_context)
    l_ans1 = lasagne.layers.InputLayer(shape=(None, 2, max_words), input_var=in_ans1)
    l_ans2 = lasagne.layers.InputLayer(shape=(None, 2, max_words), input_var=in_ans2)

    l_cmask = lasagne.layers.InputLayer(shape=l_context.shape, input_var=in_cmask)
    l_amask1 = lasagne.layers.InputLayer(shape=l_ans1.shape, input_var=in_amask1)
    l_amask2 = lasagne.layers.InputLayer(shape=l_ans2.shape, input_var=in_amask2)
    l_bbmask = lasagne.layers.InputLayer(shape=(None, 3, max_boxes), input_var=in_bbmask)

    # contexts and answers should share embeddings
    l_context_emb = lasagne.layers.EmbeddingLayer(l_context, len_voc, 
        d_word, name='word_emb')
    l_ans1_emb = lasagne.layers.EmbeddingLayer(l_ans1, len_voc, 
        d_word, W=l_context_emb.W)
    l_ans2_emb = lasagne.layers.EmbeddingLayer(l_ans2, len_voc, 
        d_word, W=l_context_emb.W)

    l_context_box_reps = SumAverageLayer([l_context_emb, l_cmask], compute_sum=True, num_dims=4)
    l_box_reshape = lasagne.layers.ReshapeLayer(l_context_box_reps, (-1, max_boxes, d_word))
    l_bbmask_reshape = lasagne.layers.ReshapeLayer(l_bbmask, (-1, max_boxes))
    l_box_lstm = lasagne.layers.LSTMLayer(l_box_reshape, num_units=d_word, mask_input=l_bbmask_reshape, only_return_final=True)
    l_context_panel_reps = lasagne.layers.ReshapeLayer(l_box_lstm, (-1, 3, d_word))
    l_context_concat = MyConcatLayer([l_context_panel_reps, l_context_fc7], axis=-1)
    l_context_proj = lasagne.layers.DenseLayer(l_context_concat, num_units=d_word, nonlinearity=lasagne.nonlinearities.rectify, num_leading_axes=2)
    l_context_final_reps = lasagne.layers.LSTMLayer(l_context_proj, num_units=d_hidden, only_return_final=True)

    l_ans1_reps = SumAverageLayer([l_ans1_emb, l_amask1], compute_sum=True, num_dims=3)
    l_ans1_panel_reps = lasagne.layers.LSTMLayer(l_ans1_reps, num_units=d_word, only_return_final=True,
          ingate=lasagne.layers.Gate(W_in=l_box_lstm.W_in_to_ingate,
                                     W_hid=l_box_lstm.W_hid_to_ingate,
                                     W_cell=l_box_lstm.W_cell_to_ingate,
                                     b=l_box_lstm.b_ingate),
          outgate=lasagne.layers.Gate(W_in=l_box_lstm.W_in_to_outgate,
                                     W_hid=l_box_lstm.W_hid_to_outgate,
                                     W_cell=l_box_lstm.W_cell_to_outgate,
                                     b=l_box_lstm.b_outgate),
           forgetgate=lasagne.layers.Gate(W_in=l_box_lstm.W_in_to_forgetgate,
                                     W_hid=l_box_lstm.W_hid_to_forgetgate,
                                     W_cell=l_box_lstm.W_cell_to_forgetgate,
                                     b=l_box_lstm.b_forgetgate),
           cell=lasagne.layers.Gate(W_in=l_box_lstm.W_in_to_cell,
                                     W_hid=l_box_lstm.W_hid_to_cell,
                                     W_cell=None,
                                     b=l_box_lstm.b_cell) ) 
    l_ans1_concat = MyConcatLayer([l_ans1_panel_reps, l_answer_fc7], axis=-1)
    l_ans1_proj = lasagne.layers.DenseLayer(l_ans1_concat, num_units=d_hidden, nonlinearity=lasagne.nonlinearities.rectify, 
        W=l_context_proj.W, b=l_context_proj.b)

    l_ans2_reps = SumAverageLayer([l_ans2_emb, l_amask2], compute_sum=True, num_dims=3)
    l_ans2_panel_reps = lasagne.layers.LSTMLayer(l_ans2_reps, num_units=d_word, only_return_final=True,
          ingate=lasagne.layers.Gate(W_in=l_box_lstm.W_in_to_ingate,
                                     W_hid=l_box_lstm.W_hid_to_ingate,
                                     W_cell=l_box_lstm.W_cell_to_ingate,
                                     b=l_box_lstm.b_ingate),
          outgate=lasagne.layers.Gate(W_in=l_box_lstm.W_in_to_outgate,
                                     W_hid=l_box_lstm.W_hid_to_outgate,
                                     W_cell=l_box_lstm.W_cell_to_outgate,
                                     b=l_box_lstm.b_outgate),
           forgetgate=lasagne.layers.Gate(W_in=l_box_lstm.W_in_to_forgetgate,
                                     W_hid=l_box_lstm.W_hid_to_forgetgate,
                                     W_cell=l_box_lstm.W_cell_to_forgetgate,
                                     b=l_box_lstm.b_forgetgate),
           cell=lasagne.layers.Gate(W_in=l_box_lstm.W_in_to_cell,
                                     W_hid=l_box_lstm.W_hid_to_cell,
                                     W_cell=None,
                                     b=l_box_lstm.b_cell) ) 
    l_ans2_concat = MyConcatLayer([l_ans2_panel_reps, l_answer_fc7], axis=-1)
    l_ans2_proj = lasagne.layers.DenseLayer(l_ans2_concat, num_units=d_hidden, nonlinearity=lasagne.nonlinearities.rectify, 
        W=l_context_proj.W, b=l_context_proj.b)

    l_scores1 = InnerProductLayer([l_context_final_reps, l_ans1_proj], is_cc=True)
    l_scores2 = InnerProductLayer([l_context_final_reps, l_ans2_proj], is_cc=True)
    l_scores = lasagne.layers.concat([l_scores1, l_scores2], axis=-1)
    l_scores = lasagne.layers.NonlinearityLayer(l_scores, nonlinearity=lasagne.nonlinearities.softmax)

    preds = lasagne.layers.get_output(l_scores)
    loss = T.mean(lasagne.objectives.categorical_crossentropy(preds, in_labels))

    all_params = lasagne.layers.get_all_params(l_scores, trainable=True)
    updates = lasagne.updates.adam(loss, all_params, learning_rate=lr)
    train_fn = theano.function([in_context_fc7, in_context_bb, in_bbmask, in_context, in_cmask, 
        in_answer_fc7, in_answer_bb, in_ans1, in_amask1, in_ans2, in_amask2, in_labels], 
        loss, updates=updates, on_unused_input='warn')
    pred_fn = theano.function([in_context_fc7, in_context_bb, in_bbmask, in_context, in_cmask, 
        in_answer_fc7, in_answer_bb, in_ans1, in_amask1, in_ans2, in_amask2], 
        preds, on_unused_input='warn')
    return train_fn, pred_fn, l_scores


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='character coherence models')
    parser.add_argument('-data', default='data/comics.h5')
    parser.add_argument('-vocab', default='data/comics_vocab.p')
    parser.add_argument('-model', default='image_only', 
        help='image_text, image_only, or text_only')
    parser.add_argument('-vgg_feats', default='data/vgg_features.h5')
    parser.add_argument('-d_word', default=256, type=int)
    parser.add_argument('-d_hidden', default=256, type=int)
    parser.add_argument('-lr', default=0.001, type=float)
    parser.add_argument('-n_epochs', default=10, type=int)
    parser.add_argument('-megabatch_size', default=512, type=int)
    parser.add_argument('-batch_size', default=64, type=int)
    args = parser.parse_args()

    print 'loading data...'
    vdict, rvdict = cPickle.load(open(args.vocab, 'rb'))
    comics_data = h5.File(args.data, 'r')
    all_vggs = h5.File(args.vgg_feats)
    train_data = load_hdf5(comics_data['train'], all_vggs['train'])
    dev_data = load_hdf5(comics_data['dev'], all_vggs['dev'])
    test_data = load_hdf5(comics_data['test'], all_vggs['test'])

    print 'training %s model for character_coherence with d_word=%d, d_hidden=%d' %\
        (args.model, args.d_word, args.d_hidden)
    log = open('logs/%s_%s_%ddword_%ddhidden.log' % (args.model, 'character_coherence', \
        args.d_word, args.d_hidden), 'w')

    # predefined parameters
    total_pages, max_panels, max_boxes, max_words = comics_data['train']['words'].shape
    len_voc = len(vdict)

    log = open('logs/%s_%s_%ddword_%ddhidden.log' % (args.model, 'character_coherence', \
        args.d_word, args.d_hidden), 'w')
    dev_fold = 'folds/%s_%s.csv' % ('char_coherence', 'dev')
    test_fold = 'folds/%s_%s.csv' % ('char_coherence', 'test')

    build_dict = {'image_text': build_image_text_network, 
        'image_only': build_image_only_network, 
        'text_only': build_text_only_network}

    build_fn = build_dict[args.model]
    print 'compiling'
    train_fn, pred_fn, final_layer = build_fn(args.d_word, args.d_hidden, args.lr)
    print 'done compiling'

    # generate train minibatches
    train_batches = [(x, x + args.megabatch_size) for x in range(0, total_pages, args.megabatch_size)]

    print 'training...'
    for epoch in range(args.n_epochs):
        epoch_loss = 0.
        start_time = time.time()
        for start, end in train_batches:
            for batch in generate_minibatches_from_megabatch(train_data, vdict,
                start, end, context_size=3, shuffle_candidates=True):

                batch_loss = train_fn(*batch[1:])
                epoch_loss += batch_loss

        epoch_log = 'done with epoch %d in %d seconds, loss is %f' % \
            (epoch, time.time() - start_time, epoch_loss / len(train_batches))
        log.write(epoch_log + '\n')
        print epoch_log

        dev_val = validate('dev', dev_data, dev_fold)
        test_val = validate('test', test_data, test_fold)

        log.write(dev_val + '\n')
        log.write(test_val + '\n\n')
        print dev_val
        print test_val
        log.flush()
