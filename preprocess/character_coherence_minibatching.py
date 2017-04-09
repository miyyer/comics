import time, sys, csv
import h5py as h5
from os.path import join
import cPickle as pickle
import numpy as np
import random

# read csv, extract answer candidates and label, and store as dict
def read_fold(csv_file, vdict, max_len=30):
    reader = csv.DictReader(open(csv_file, 'r'))
    fold_dict = {}
    for row in reader:
        key = '%s_%s_%s' % (row['book_id'], row['page_id'], row['answer_panel_id'])
        fold_dict[key] = []
        cand1 = np.zeros((2, max_len)).astype('int32')
        cand1_masks = np.zeros((2, max_len)).astype('float32')
        cand2 = np.zeros((2, max_len)).astype('int32')
        cand2_masks = np.zeros((2, max_len)).astype('float32')
        label = [0,0]
        corr_answer = int(row['correct_answer'])
        label[corr_answer] = 1
        for a in range(2):
            text = row['answer_candidate_%d_box_%d' % (0,a)].split()
            cand1[a,:len(text)] = [vdict[w] for w in text]
            cand1_masks[a, :len(text)] = 1.

        for a in range(2):
            text = row['answer_candidate_%d_box_%d' % (1,a)].split()
            cand2[a,:len(text)] = [vdict[w] for w in text]
            cand2_masks[a, :len(text)] = 1.

        fold_dict[key] = [cand1, cand1_masks, cand2, cand2_masks, label]

    return fold_dict

# save folds to a csv file
def save_fold(data, vdict, fold_name, num_pages, batch_size=1024, 
        fold_dict=None):

    fields = ['book_id', 'page_id', 'answer_panel_id', 'context_panel_0_id', 
        'context_panel_1_id', 'context_panel_2_id']

    for a in range(3):
        for b in range(3):
            fields.append('context_text_%d_%d' % (a,b))

    for a in range(2):
        for b in range(2):
            fields += ['answer_candidate_%d_box_%d' % (a,b)]

    fields += ['correct_answer']

    out_file = csv.DictWriter(open('folds/char_coherence_%s.csv' % (fold_name,), 'w'), fieldnames=fields)
    out_file.writerow( dict((f,f) for f in out_file.fieldnames))
    
    # read folds in megabatches
    mbs = [(x, x+batch_size )for x in range(0, num_pages, batch_size)]
    mbs[-1] = (mbs[-1][0], min(num_pages, mbs[-1][-1]))
    total_sum = 0
    rows = []
    for mb_start, mb_end in mbs:

        for batch in generate_minibatches_from_megabatch(data, vdict, mb_start, mb_end, 
            shuffle_candidates=True, fold_dict=None):
            
            a_id, c_fc7, c_bb, c_bbm, c_w, c_wm, a_img, a_bb, a_w, a_wm, r_aw, r_awm, labels = batch
            for i in range(len(a_id)):
                book, page, panel = a_id[i]
                prev_pages = []
                instance = {}
                instance['book_id'] = book
                instance['page_id'] = page
                instance['answer_panel_id'] = panel
                for j, context_panels in enumerate(range(panel-3, panel)):
                    instance['context_panel_%d_id'%j] = context_panels
                    for box in range(3):
                        if c_bbm[i,j,box] == 1:
                            text = ' '.join([rvdict[y] for r,y in enumerate(c_w[i, j, box]) \
                                if c_wm[i, j, box, r] == 1])
                            instance['context_text_%d_%d' % (j, box)] = text

                corr_answer = np.argmax(labels[i])
                instance['correct_answer'] = corr_answer

                for a in range(2):
                    corr_idx = 0 if corr_answer == 0 else 1
                    text = ' '.join([rvdict[y] for r,y in enumerate(a_w[i, a]) \
                                if a_wm[i, a, r] == 1])
                    instance['answer_candidate_%d_box_%d'%(corr_idx,a)] = text
                for a in range(2):
                    corr_idx = 1 if corr_answer == 0 else 0
                    text = ' '.join([rvdict[y] for r,y in enumerate(r_aw[i, a]) \
                                if r_awm[i, a, r] == 1])
                    instance['answer_candidate_%d_box_%d'%(corr_idx,a)] = text
                    

                out_file.writerow(instance)

            total_sum += len(batch[0])

        print 'done with %d' % mb_start, total_sum


# takes a megabatch and generates a bunch of minibatches
def generate_minibatches_from_megabatch(data, vdict, mb_start, mb_end, batch_size=64, context_size=3, 
    shortest_answer=3, window_size=3, num_candidates=3, max_unk=2, difficulty='hard', 
    fold_dict=None, shuffle_candidates=True):

    img_mask, book_ids, page_ids, bboxes, bbox_mask, words, word_mask, comics_fc7 = data
    curr_fc7 = comics_fc7[mb_start:mb_end]

    # binarize bounding box mask (no narrative box distinction)
    curr_bmask_raw = bbox_mask[mb_start:mb_end]
    curr_bmask = np.clip(curr_bmask_raw, 0, 1)

    curr_bboxes = bboxes[mb_start:mb_end] / 224.
    curr_words = words[mb_start:mb_end]
    curr_wmask = word_mask[mb_start:mb_end]
    curr_book_ids = book_ids[mb_start:mb_end]
    curr_page_ids = page_ids[mb_start:mb_end]
    curr_imasks = img_mask[mb_start:mb_end]

    num_panels = np.sum(curr_imasks, axis=-1).astype('int32')

    # need to sum the number of words per box
    words_per_box = np.sum(curr_wmask, axis=-1).astype('int32')
    possible_candidates = np.where(words_per_box >= shortest_answer)
    possible_candidates = set(zip(*possible_candidates))
    
    # compute number of UNKs per box for filtering
    unks_in_candidates = np.sum((curr_words == vdict['UNK']), axis=-1)
    unk_candidates = np.where(unks_in_candidates < max_unk)
    unk_candidates = set(zip(*unk_candidates))
    possible_candidates = possible_candidates.intersection(unk_candidates)
    pc_tuple = tuple(possible_candidates)

    # loop through each page, create as many training examples as possible
    context_imgs = []
    context_words = []
    context_wmask = []
    context_bboxes = []
    context_bmask = []
    context_fc7 = []
    answer_ids = []
    answer_imgs = []
    answer_fc7 = []
    answer_bboxes = []
    answer_bmask = []
    candidates = []
    
    iter_end = num_panels.shape[0] - 1
    for i in range(0, iter_end):
        curr_np = num_panels[i]

        # not enough panels to have context and candidate
        if curr_np < context_size + 1:
            continue

        # see if there is a previous and next page
        if curr_page_ids[i - 1] != curr_page_ids[i] - 1 or curr_page_ids[i + 1] != curr_page_ids[i] + 1:
            continue

        # subtract 1 because random.randint is inclusive
        prev_np = num_panels[i - 1] - 1
        next_np = num_panels[i + 1] - 1

        num_examples = curr_np - context_size
        for j in range(num_examples):

            # make sure panel has only two dialogue boxes (no narrative boxes)
            if np.sum(curr_bmask[i, j+context_size]) != 2\
                or np.sum(curr_bmask_raw[i, j+context_size]) != 2:
                continue

            # make sure answer text box isn't blank or super short
            if words_per_box[i, j+context_size, 0] < shortest_answer:
                continue            

            # make sure context/answer text doesn't have too many UNKs
            # because that is an indicator of poor OCR 
            too_many_unks = False
            for c_ind in range(context_size + 1):
                if np.sum(unks_in_candidates[i, j+c_ind] >= max_unk, axis=-1) > 0:
                    too_many_unks = True
            if too_many_unks:
                continue

            context_fc7.append(curr_fc7[i, j:j+context_size])
            context_bboxes.append(curr_bboxes[i, j:j+context_size])                
            context_bmask.append(curr_bmask[i, j:j+context_size])
            context_words.append(curr_words[i, j:j+context_size])
            context_wmask.append(curr_wmask[i, j:j+context_size])
            key = (curr_book_ids[i], curr_page_ids[i], j+context_size)
            answer_ids.append(key)
            answer_fc7.append(curr_fc7[i, j+context_size])
            answer_bboxes.append(curr_bboxes[i, j+context_size][:2])
            answer_bmask.append(curr_bmask[i, j+context_size][:2])

            # if cached fold, just use the stored candidates
            if fold_dict:
                key = '_'.join([str(z) for z in key])
                candidates.append(fold_dict[key])

            else:
                text_candidates = curr_words[i, j+context_size][:2]
                mask_candidates = curr_wmask[i, j+context_size][:2]
                candidates.append((text_candidates, mask_candidates, 
                    text_candidates[::-1], mask_candidates[::-1], [1,0]))

    # create numpy-ized minibatches
    batch_inds = [(x, x + batch_size) for x in range(0, len(candidates), batch_size)]
    for start, end in batch_inds:

        a_id = answer_ids[start:end]
        c_fc7 = np.array(context_fc7[start:end])
        c_bb = np.array(context_bboxes[start:end]).astype('float32')
        c_bbm = np.array(context_bmask[start:end]).astype('float32')
        c_w = np.array(context_words[start:end]).astype('int32')
        c_wm = np.array(context_wmask[start:end]).astype('float32')
        a_fc7 = np.array(answer_fc7[start:end])
        a_bb = np.array(answer_bboxes[start:end]).astype('float32')
        
        a_w = []
        a_wm = []
        rev_aw = []
        rev_awm = []
        label_list = []
        for cand in candidates[start:end]:

            label = cand[4]
            corr_label = np.argmax(label)
            if shuffle_candidates and not fold_dict:
                corr_label = np.random.choice(2)
                label = [0, 0]
                label[corr_label] = 1

            if corr_label == 0:
                a_w.append(cand[0])
                a_wm.append(cand[1])
                rev_aw.append(cand[2])
                rev_awm.append(cand[3])
            else:
                a_w.append(cand[2])
                a_wm.append(cand[3])
                rev_aw.append(cand[0])
                rev_awm.append(cand[1])
            label_list.append(label)

        a_w = np.array(a_w).astype('int32')
        a_wm = np.array(a_wm).astype('float32')

        r_aw = np.array(rev_aw).astype('int32')
        r_awm = np.array(rev_awm).astype('float32')

        labels = np.array(label_list).astype('int32')

        c_bb = np.clip(c_bb, 0, 1)
        a_img = np.array(answer_fc7[start:end])
        a_bb = np.array(answer_bboxes[start:end]).astype('float32')
        batch_data = [a_id, c_fc7, c_bb, c_bbm, c_w, c_wm, a_img, a_bb, a_w, a_wm, r_aw, r_awm, labels]                
        yield batch_data

if __name__ == '__main__':
    vdict, rvdict = pickle.load(open('data/comics_vocab.p', 'rb'))
    comics_data = h5.File('data/comics.h5', 'r')
    comics_fc7 = h5.File('data/vgg_features.h5')

    for fold in ['dev', 'test']:

        fold_data = comics_data[fold]
        imgs = fold_data['images']
        img_mask = fold_data['panel_mask']
        book_ids = fold_data['book_ids']
        page_ids = fold_data['page_ids']
        bboxes = fold_data['bbox']
        bbox_mask = fold_data['bbox_mask']
        words = fold_data['words']
        word_mask = fold_data['word_mask']
        data = (img_mask, book_ids, page_ids, bboxes, bbox_mask, words, \
            word_mask, comics_fc7[fold]['vgg_features'])

        num_pages = imgs.shape[0]
        max_len = words.shape[-1]

        save_fold(data, vdict, fold, num_pages, batch_size=1024, fold_dict=None)
