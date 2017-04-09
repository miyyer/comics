import time, sys, csv
import h5py as h5
from os.path import join
import cPickle as pickle
import numpy as np
import random

# read csv, extract answer candidates and label, and store as dict
def read_fold(csv_file):
    reader = csv.DictReader(open(csv_file, 'r'))
    fold_dict = {}
    for row in reader:
        key = '%s_%s_%s' % (row['book_id'], row['page_id'], row['answer_panel_id'])
        fold_dict[key] = []
        candidates = []
        label = [0,0,0]
        label[int(row['correct_answer'])] = 1
        for i in range(3):
            c = row['answer_candidate_id_%d' % i]
            candidates.append([int(x) for x in c.split('_')])

        fold_dict[key] = [candidates, label]

    return fold_dict

# save folds to a csv file
def save_fold(data, fold_name, num_pages, batch_size=1024, 
        difficulty='easy', fold_dict=None):

    fields = ['book_id', 'page_id', 'answer_panel_id', 'context_panel_0_id', 'context_panel_1_id', 'context_panel_2_id']
    for a in range(3):
        for b in range(3):
            fields.append('context_text_%d_%d' % (a,b))

    fields += ['answer_candidate_id_%s' % a for a in range(3)]
    fields += ['correct_answer']

    out_file = csv.DictWriter(open('folds/visual_cloze_%s_%s.csv' % (fold_name, difficulty), 'w'), fieldnames=fields)
    out_file.writerow( dict((f,f) for f in out_file.fieldnames))
    
    # read folds in megabatches
    mbs = [(x, x+batch_size )for x in range(0, num_pages, batch_size)]
    mbs[-1] = (mbs[-1][0], min(num_pages, mbs[-1][-1]))
    total_sum = 0
    rows = []
    for mb_start, mb_end in mbs:

        for batch in generate_minibatches_from_megabatch(data, mb_start, mb_end, 
            difficulty=difficulty, shuffle_candidates=True, fold_dict=fold_dict):
            
            a_id, c_fc7, c_bb, c_bbm, c_w, c_wm, a_i, labels, shuf_cand_ids = batch
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
                    cand_id = shuf_cand_ids[i][a]
                    instance['answer_candidate_id_%d'%a] = '_'.join([str(x) for x in cand_id])
                    
                instance['correct_answer'] = np.argmax(labels[i])
                out_file.writerow(instance)

            total_sum += len(batch[0])

        print 'done with %d' % mb_start, total_sum


# takes a megabatch and generates a bunch of minibatches
def generate_minibatches_from_megabatch(data, mb_start, mb_end, batch_size=64, 
    context_size=3, window_size=3, num_candidates=3,
    difficulty='easy', fold_dict=None, shuffle_candidates=True):

    img_mask, book_ids, page_ids, bboxes, bbox_mask, words, word_mask, comics_fc7 = data
    curr_fc7 = comics_fc7[mb_start:mb_end]

    # binarize bounding box mask (no narrative box distinction)
    curr_bmask_raw = bbox_mask[mb_start:mb_end]
    curr_bmask = np.clip(curr_bmask_raw, 0, 1)

    curr_bboxes = bboxes[mb_start:mb_end] / 224.
    curr_words = words[mb_start:mb_end]
    curr_wmask = word_mask[mb_start:mb_end]
    curr_imasks = img_mask[mb_start:mb_end]
    curr_book_ids = book_ids[mb_start:mb_end]
    curr_page_ids = page_ids[mb_start:mb_end]

    # inverse mapping for visual cloze candidates
    # page_id to actual index
    page_to_idx = {}
    for i, p_id in enumerate(curr_page_ids):
        b_id = curr_book_ids[i]
        page_to_idx['%d_%d'%(b_id,p_id)] = i

    num_panels = np.sum(curr_imasks, axis=-1).astype('int32')

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
                window_start = max(0, i - window_size)
                window_end = min(i + 1 + window_size, iter_end + 1)

                # candidates come from previous / next page
                if difficulty == 'hard':
                    random_page_1 = random.randint(window_start, max(0, i - 1))
                    random_page_2 = random.randint(i + 1, window_end - 1)

                # candidates come from random pages in the megabatch
                else:
                    random_page_1 = random.randint(0, iter_end - 1)
                    random_page_2 = random.randint(0, iter_end - 1)

                prev_sel = random.randint(0, num_panels[random_page_1] - 1)
                next_sel = random.randint(0, num_panels[random_page_2] - 1)

                # corr = 0, prev = 1, next = 2
                candidate_ids = []
                candidate_ids.append(key)
                candidate_ids.append( (curr_book_ids[random_page_1], curr_page_ids[random_page_1], prev_sel) )
                candidate_ids.append( (curr_book_ids[random_page_2], curr_page_ids[random_page_2], next_sel) )

                candidates.append([candidate_ids, [1,0,0]])
           

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
        
        a_i = []
        labels = []
        cand_ids = []
        shuf_cand_ids = []

        for cand in candidates[start:end]:
            cand_ids.append(cand[0])

            # get corresponding fc7s
            mapped_cand_ids = []
            for b,p,pan in cand[0]:
                mapped_cand_ids.append( (page_to_idx['%d_%d' % (b,p)], pan) )
            ims = curr_fc7[zip(*mapped_cand_ids)]
            a_i.append(ims)

            labels.append(cand[1])

        a_i = np.array(a_i).astype('float32')
        labels = np.array(labels).astype('int32')

        if fold_dict:
            shuf_cand_ids = cand_ids

        if shuffle_candidates and not fold_dict:
            for idx in range(a_i.shape[0]):
                p = np.random.permutation(a_i.shape[1])
                a_i[idx] = a_i[idx, p]
                labels[idx] = labels[idx, p]

                # ids of shuffled candidates
                # could be useful if using some other image features
                this_shuf_id = []
                for shuf_idx in p:
                    this_shuf_id.append(cand_ids[idx][shuf_idx])
                shuf_cand_ids.append(this_shuf_id)
                    
        batch_data = [a_id, c_fc7, c_bb, c_bbm, c_w, c_wm, a_i, labels, shuf_cand_ids]
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

            save_fold(data, fold, num_pages, difficulty=difficulty,
                    batch_size=1024, fold_dict=None)
