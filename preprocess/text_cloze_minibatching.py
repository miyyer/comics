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
        candidates = np.zeros((3, max_len)).astype('int32')
        candidate_masks = np.zeros((3, max_len)).astype('float32')
        label = [0,0,0]
        label[int(row['correct_answer'])] = 1
        for i in range(3):
            c = row['answer_candidate_%d_text' % i].split()
            candidates[i,:len(c)] = [vdict[w] for w in c]
            candidate_masks[i, :len(c)] = 1.

        fold_dict[key] = [candidates, candidate_masks, label]

    return fold_dict

# save folds to a csv file
def save_fold(data, vdict, fold_name, num_pages, batch_size=1024, 
        difficulty='easy', fold_dict=None):

    fields = ['book_id', 'page_id', 'answer_panel_id', 'context_panel_0_id', 'context_panel_1_id', 'context_panel_2_id']
    for a in range(3):
        for b in range(3):
            fields.append('context_text_%d_%d' % (a,b))

    fields += ['answer_candidate_%d_text' % a for a in range(3)]
    fields += ['correct_answer']

    out_file = csv.DictWriter(open('folds/text_cloze_%s_%s.csv' % (fold_name, difficulty), 'w'), fieldnames=fields)
    out_file.writerow( dict((f,f) for f in out_file.fieldnames))
    
    # read folds in megabatches
    mbs = [(x, x+batch_size )for x in range(0, num_pages, batch_size)]
    mbs[-1] = (mbs[-1][0], min(num_pages, mbs[-1][-1]))
    total_sum = 0
    rows = []
    for mb_start, mb_end in mbs:

        for batch in generate_minibatches_from_megabatch(data, vdict, mb_start, mb_end, 
            difficulty=difficulty, shuffle_candidates=True, fold_dict=fold_dict):
            
            a_id, c_fc7, c_bb, c_bbm, c_w, c_wm, a_fc7, a_bb, a_w, a_wm, labels = batch
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

                for a in range(3):
                    text = ' '.join([rvdict[y] for r,y in enumerate(a_w[i, a]) \
                                if a_wm[i, a, r] == 1])
                    instance['answer_candidate_%d_text'%a] = text
                    
                instance['correct_answer'] = np.argmax(labels[i])

                out_file.writerow(instance)

            total_sum += len(batch[0])

        print 'done with %d' % mb_start, total_sum


# takes a megabatch and generates a bunch of minibatches
def generate_minibatches_from_megabatch(data, vdict, mb_start, mb_end, batch_size=64, context_size=3, 
    shortest_answer=3, window_size=3, num_candidates=3, max_unk=2, difficulty='easy', 
    only_singleton_panels=True, fold_dict=None, shuffle_candidates=True):

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

            # if text cloze, make sure answer panel has only one text box
            if only_singleton_panels:
                if np.sum(curr_bmask[i, j+context_size]) != 1:
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
            answer_bboxes.append(curr_bboxes[i, j+context_size][0])

            # if cached fold, just use the stored candidates
            if fold_dict:
                key = '_'.join([str(z) for z in key])
                candidates.append(fold_dict[key])

            # otherwise randomly sample candidates (for training)
            else:

                # candidates come from previous / next page
                if difficulty == 'hard':
                    text_candidates = np.zeros((3, curr_words.shape[-1]))
                    mask_candidates = np.zeros((3, curr_words.shape[-1]))

                    # see if any panels in the surrounding pages have long enough text boxes
                    window_start = max(0, i - window_size)
                    window_end = min(i + 1 + window_size, iter_end + 1)

                    coords_1 = [coord for coord in pc_tuple if coord[0] in set(range(window_start,i))]
                    coords_2 = [coord for coord in pc_tuple if coord[0] in set(range(i+1,window_end))]

                    # if no usable candidates found in neighboring pages
                    # just randomly sample from all possible candidates
                    # note: this is very rare!
                    if len(coords_1) == 0:
                        chosen_prev_candidate = random.choice(pc_tuple)
                    else:
                        chosen_prev_candidate = random.choice(coords_1)

                    if len(coords_2) == 0:
                        chosen_next_candidate = random.choice(pc_tuple)
                    else:
                        chosen_next_candidate = random.choice(coords_2)

                    # corr = 0, prev = 1, next = 2
                    text_candidates[0] = curr_words[i, j+context_size, 0]
                    text_candidates[1] = curr_words[chosen_prev_candidate]
                    text_candidates[2] = curr_words[chosen_next_candidate]
                    mask_candidates[0] = curr_wmask[i, j+context_size, 0]
                    mask_candidates[1] = curr_wmask[chosen_prev_candidate]
                    mask_candidates[2] = curr_wmask[chosen_next_candidate]
                    candidates.append((text_candidates, mask_candidates, [1,0,0]))

                # candidates come from random pages in the megabatch
                else:
                    text_candidates = np.zeros((num_candidates, curr_words.shape[-1]))
                    mask_candidates = np.zeros((num_candidates, curr_words.shape[-1]))

                    # corr = 0, all other indices are wrong candidates
                    text_candidates[0] = curr_words[i, j+context_size, 0]
                    mask_candidates[0] = curr_wmask[i, j+context_size, 0]

                    for cand_idx in range(num_candidates - 1):
                        coords = random.choice(pc_tuple)

                        text_candidates[cand_idx + 1] = curr_words[coords]
                        mask_candidates[cand_idx + 1] = curr_wmask[coords]
                
                    candidates.append((text_candidates, mask_candidates, [1,0,0]))

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
        inc_w = []
        inc_wm = []
        labels = []

        for cand in candidates[start:end]:
            a_w.append(cand[0])
            a_wm.append(cand[1])
            labels.append(cand[2])

        a_w = np.array(a_w).astype('int32')
        a_wm = np.array(a_wm).astype('float32')
        labels = np.array(labels).astype('int32')

        if shuffle_candidates and not fold_dict:
            for idx in range(a_w.shape[0]):
                p = np.random.permutation(a_w.shape[1])
                a_w[idx] = a_w[idx, p]
                a_wm[idx] = a_wm[idx, p]
                labels[idx] = labels[idx, p]

        batch_data = [a_id, c_fc7, c_bb, c_bbm, c_w, c_wm, a_fc7, a_bb, a_w, a_wm, labels]
        yield batch_data


if __name__ == '__main__':
    vdict, rvdict = pickle.load(open('data/comics_vocab.p', 'rb'))
    comics_data = h5.File('data/comics.h5', 'r')
    comics_fc7 = h5.File('data/vgg_features.h5')

    for difficulty in ['easy', 'hard']:
        for fold in ['dev', 'test']:

            print difficulty, fold

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

            save_fold(data, vdict, fold, num_pages, difficulty=difficulty,
                    batch_size=1024, fold_dict=None)#read_fold('textcloze_dev_easy.csv'))
