from __future__ import division
import h5py as h5
import csv
import re
import cv2
from os.path import join
import cPickle as pickle
from collections import defaultdict, Counter
import numpy as np
import glob2
import sys

'''                                                                                
Creates one hdf5 file with the following properties :

# one training sample is one page                                                  
n pages total                                                                      
book_ids (h_book_ids): n                                                           
page_ids (h_page_ids): n                                                           
images (h_ims): n x max_panels x 256 x 256 x 3 (or whatever your size is)          
panel_mask (h_pmask): n x max_panels                                               
text bounding box coords [bbox] (h_coords): n x max_panels x max_boxes x 4         
text bbox mask [bbox_mask] (h_tmask): n x max_panels x max_boxes                   
text box words [words] (h_words): n x max_panels x max_boxes x max_words           
text box word mask [words_mask] (h_wmask): n x max_panels x max_boxes x max_words  
'''                                                                                

class Create_hdf5:

    def __init__(self, panels_path, ocr_file, ad_path, dims, max_panels, max_boxes, max_words, max_vocab_size, h5path):

        self.panels_path = panels_path
        # self.images_path = images_path

        # self.n_pages = glob2.glob(join(self.images_path, '*', '*.jpg')) 
        # takes a really long time

        self.ocr_file = ocr_file
        self.dev_idx, self.test_idx, self.n_pages = self.compute_fold_starts()

        self.n_pages -= self.subtract_ad_pages(ad_path)

        self.dims = dims
        self.h5path = h5path
        self.max_panels = max_panels
        self.max_boxes = max_boxes
        self.max_words = max_words
        self.max_vocab_size = max_vocab_size
        self.dump_vocabulary('./data/comics_vocab.p')
        
    def dump_vocabulary(self, vocab_path): # helper which dumps vocabulary in pickle
        '''
        Creates vocabulary
        # Arguments
            vocab_path : path where the vocabulary is dumped as a pickle file
        '''
        print 'loading ocr dict'
        self.ocr_dict = defaultdict(list)
        self.vcount = Counter()
        self.read_ocr_file()
        print 'done creating ocr dict, now filtering vocab of size %d' % \
                len(self.vcount)

        vocab = self.vcount.most_common(self.max_vocab_size)
        self.vdict = {}
        for i, (w, count) in enumerate(vocab):
            self.vdict[w] = i
        self.vdict['UNK'] = len(self.vdict)
        self.rvdict = dict((v,k) for k,v in self.vdict.iteritems())
        print 'done filtering vocab to %d words' % len(self.vdict)

        pickle.dump([self.vdict, self.rvdict],
                open(vocab_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    # use 500 books for both dev and test, remaining are for training
    def compute_fold_starts(self):
        raw_f = open(self.ocr_file, 'r')
        f = csv.DictReader(raw_f)

        # comic_no,page_no,panel_no,textbox_no,dialog_or_narration,text,x1,y1,x2,y2
        # compute number of books
        unique_books = Counter()

        for row in f:
            unique_books[int(row['comic_no'])] += 1

        num_books = len(unique_books) #3929
        dev_thresh = num_books - 1000
        test_thresh = num_books - 500

        raw_f.close()
        raw_f = open(self.ocr_file, 'r')

        # now count pages to find fold indices
        f = csv.DictReader(raw_f)

        book_count = 0
        page_count = 0
        prev_book = -1
        prev_page = -1
        dev_idx = -1
        test_idx = -1
        for row in f:
            curr_book = int(row['comic_no'])
            curr_page = int(row['page_no'])
            if curr_book != prev_book:
                book_count += 1
                prev_book = curr_book

            if prev_page != curr_page:
                page_count += 1
                prev_page = curr_page

            if book_count == dev_thresh:
                dev_idx = page_count

            if book_count == test_thresh:
                test_idx = page_count


        return dev_idx, test_idx, page_count
        

    def subtract_ad_pages(self, ad_path): # helper
        '''
        counts the number of ad pages
        # Arguments : 
            ad_path : location of all ad pages (using a BOW classifier) 
        # Returns :
            len(self.ad_pages) : number of ad pages
        '''
        ad_file = open(ad_path, 'r')
        self.ad_pages = set()
        for line in ad_file:
            self.ad_pages.add(line.strip())
        return len(self.ad_pages)


    def scale_bbox(self, bbox_coords, panel_coords, new_dims):
        '''
        Scales text bounding box coordinates within the reference 
        frame of panel coordinates (between 0 and 1)

        # Arguments : 
            bbox_coords : absolute coordinates of the text bounding box in the page
            panel_coords : [0, 0, w, h] where w and h are width and height of panel
            new_dims : size of the panel after saving [something like (256, 256) ] 
        # Returns : 
            u1_, v1_, u2_, v2_ which are the scaled coordinates of textbox
        '''
        w, h = new_dims
        u1, v1, u2, v2 = bbox_coords
        x1, y1, x2, y2 = panel_coords
        u1_ = u1 * w / (x2 - x1)
        v1_ = v1 * h / (y2 - y1)
        u2_ = u2 * w / (x2 - x1)
        v2_ = v2 * h / (y2 - y1)
        return int(u1_), int(v1_), int(u2_), int(v2_)

    def scale_textbox_coords(self, w, h, bbox_coords):
        '''
        Scales bounding box coordinates to between 0 and 1.
        # Arguments : 
            w, h : panel width and height
            bbox_coords : textbox_coordinates that are to be scaled
        # Returns : 
            scaled_coords : textbox_coordinates after scaling 
        '''

        scaled_coords = self.scale_bbox(bbox_coords, [0, 0, w, h], self.dims)
        return scaled_coords

    def blacken_box(self, img, coords):
        '''
        We blacken the textboxes in an image (so that the computer doesn't
        learn to predict the answer by trying to read the box)
        # Arguments : 
            img : panel image after resizing 
            coords : using coords, draw a rectangle and paint it black
        # Returns : 
            img : image after all of its text boxes have been blackened
        '''
        for coord_ in coords:
            u1, v1, u2, v2 = coord_.astype('uint8')
            cv2.rectangle(img, (u1, v1), (u2, v2), (0, 0, 0), -1)
        return img


    def read_ocr_file(self, save='No'):
        '''
        read csv files with ocr and return a tuple containing
        (comic, page no, panel no, textbox no, class (n or d), 
        panel img, normalized coords of box, text)
        '''
        with open(self.ocr_file, 'rb') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # ocr_list.append((comic, page, panel, textbox, text))
                comic_no = row['comic_no']
                page_no = row['page_no']
                panel_no = row['panel_no']
                textbox_no = row['textbox_no']
                text = row['text']
                dialog_or_narration = row['dialog_or_narration']
                x1 = row['x1']
                y1 = row['y1']
                x2 = row['x2']
                y2 = row['y2']
                
                text = text.split()
                for w in text:
                    self.vcount[w] += 1
                self.ocr_dict[(int(comic_no), int(page_no), int(panel_no))].append( 
                        [textbox_no, dialog_or_narration, text, x1, y1, x2, y2])

        if save == 'Yes':
            pickle.dump(self.ocr_dict, open('ocr_dict.p', 'w'),
                    protocol=pickle.HIGHEST_PROTOCOL)

    def create_img_hdf5(self):
        '''
        # one training sample is one page                            
        n pages total                                                       
        book_ids (h_book_ids): n                                            
        page_ids (h_page_ids): n                                            
        images (h_ims): n x max_panels x 256 x 256 x 3 (or whatever your size is)
        panel_mask (h_pmask): n x max_panels
        text bounding box coords [bbox] (h_coords): n x max_panels x max_boxes x 4
        text bbox mask [bbox_mask] (h_tmask): n x max_panels x max_boxes
        text box words [words] (h_words): n x max_panels x max_boxes x max_words
        text box word mask [words_mask] (h_wmask): n x max_panels x max_boxes x max_words
        '''
        n_train_pages = self.dev_idx
        n_dev_pages = self.test_idx - self.dev_idx
        # n_test_pages = self.n_pages - self.test_idx

        keys = sorted(self.ocr_dict.keys()) # is in terms of panels. 
        # sort keys in alphabetical order


        unique_pages = sorted(list(set([(key[0], key[1]) for key in keys])))
        train_pages = unique_pages[:self.dev_idx]
        dev_pages = unique_pages[self.dev_idx:self.test_idx]
        test_pages = unique_pages[self.test_idx:]

        # so, for train fold, pick all panels from keys[0] to (dev_pages[0], 0), not including
        # for dev fold, pick all panels from (dev_pages[0], 0) to (test_pages[0], 0), not including
        # for test fold, pick all panels from (test_pages[0], 0) to end

        train_keys = []
        dev_keys = []
        test_keys = []

        train_limit = (dev_pages[0][0], dev_pages[0][1], 0)
        dev_limit = (test_pages[0][0], test_pages[0][1], 0)

        for key in keys:
            if key < train_limit:
                train_keys.append(key)
            elif key >= train_limit and key < dev_limit:
                dev_keys.append(key)
            elif key >= dev_limit:
                test_keys.append(key)
                

        fn_num_pages = lambda keys:len(set([(key[0], key[1]) for key in keys]))

        n_train_ads = len([t for t in train_pages if '%s---%s' % (t[0], t[1]) in self.ad_pages])
        n_dev_ads = len([t for t in dev_pages if '%s---%s' % (t[0], t[1]) in self.ad_pages])
        n_test_ads = len([t for t in test_pages if '%s---%s' % (t[0], t[1]) in self.ad_pages])

        d = {} # will help us loop over train, dev and test folds
        d['train'] = {'n':fn_num_pages(train_keys)-n_train_ads, 'keys':train_keys}
        d['dev'] = {'n':fn_num_pages(dev_keys)-n_dev_ads, 'keys':dev_keys}
        d['test'] = {'n':fn_num_pages(test_keys)-n_test_ads, 'keys':test_keys}

        detected_ads = set()
        for fold in d.keys(): # loop through train, test, dev
            # creating hdf5 dataset
            f = h5.File(self.h5path, 'a')
            h_book_ids = f.create_dataset(fold+'/book_ids',
                    shape=(d[fold]['n'], ), dtype=np.uint32)
            h_page_ids = f.create_dataset(fold+'/page_ids',
                    shape=(d[fold]['n'], ), dtype=np.uint32)
            h_ims = f.create_dataset(fold+'/images',
                    shape=(d[fold]['n'], self.max_panels, self.dims[0], self.dims[1], 3),
                    dtype=np.uint8)
            h_pmask = f.create_dataset(fold+'/panel_mask',
                    shape=(d[fold]['n'], self.max_panels), dtype=np.uint8)
            h_coords = f.create_dataset(fold+'/bbox',
                    shape=(d[fold]['n'], self.max_panels, self.max_boxes, 4),
                    dtype=np.uint8)
            h_tmask = f.create_dataset(fold+'/bbox_mask',
                    shape=(d[fold]['n'], self.max_panels, self.max_boxes),
                    dtype=np.uint8)
            h_words = f.create_dataset(fold+'/words',
                    shape=(d[fold]['n'], self.max_panels, self.max_boxes, self.max_words),
                    dtype=np.uint32)
            h_wmask = f.create_dataset(fold+'/word_mask',
                    shape=(d[fold]['n'], self.max_panels, self.max_boxes, self.max_words),
                    dtype=np.uint8)

            curr_page_idx = 0
            prev_page = None
            for i, key in enumerate(d[fold]['keys']):
                comic_no, page_no, panel_no = key
                if i % 5000 == 0 and i > 0:
                    print '%d of panels done in fold %s, %d of %d ads detected' % (i, fold, len(detected_ads),
                            len(self.ad_pages))

                # check for ads
                ad_check_id = '%s---%s' % (comic_no, page_no)
     
                if ad_check_id in self.ad_pages:
                    detected_ads.add(ad_check_id)
                    continue
                
                if prev_page == None:
                    prev_page = page_no

                if page_no != prev_page:
                    curr_page_idx += 1
                    prev_page = page_no

                h_book_ids[curr_page_idx] = comic_no
                h_page_ids[curr_page_idx] = page_no
                if panel_no >= self.max_panels:
                    continue
               
               # load panel image
                img = cv2.imread(join(self.panels_path, str(comic_no),
                    str(page_no)+'_'+str(panel_no)+'.jpg'))
                h_, w_, c_ = img.shape
                img = cv2.resize(img, self.dims)

                panel_contents = self.ocr_dict[key]
                # are there any lines in the CSV file without OCR?
                if len(panel_contents) == 1 and panel_contents[0][0] == None:
                    # no textboxes, so nothing to blacken
                    h_ims[curr_page_idx, panel_no] = img
                    h_pmask[curr_page_idx, panel_no] = 1 
                else:
                    tcount = 0
                    # there are textboxes, so need to blacken
                    new_coords_blk_list = np.zeros((len(panel_contents), 4))
                    for j, textbox in enumerate(panel_contents):
                        # ocr_key = '%d_%d_%d_%d' % (comic_no, page_no, panel_no, j)
                        # if ocr_key in self.ocr_dict:
                            # textbox contains : [dialog_or_narration, text, x1, y1, x2, y2])
                            words = textbox[2]
                            if len(words) > 0:
                                dialog_or_narration = int(textbox[1])
                                x1, y1, x2, y2 = map(float, textbox[-4:])
                                new_coords = self.scale_textbox_coords(w_, h_, [x1, y1, x2, y2]) 
                                new_coords_blk_list[tcount] = new_coords
                                h_coords[curr_page_idx, panel_no, tcount] = new_coords
                                # mask is 0 if theres no textbox, 1 if its a speech bubble, 2 if its a narrative bo
                                h_tmask[curr_page_idx, panel_no, tcount] = dialog_or_narration

                                # remove unknown words
                                inds = [self.vdict[w] if w in self.vdict else self.vdict['UNK'] for w in words ][:self.max_words]
                                h_words[curr_page_idx, panel_no, tcount, :len(inds)] = inds
                                h_wmask[curr_page_idx, panel_no, tcount, :len(inds)] = 1

                                if tcount == self.max_boxes-1:
                                    # limit the number of textboes per panel
                                    break
                                tcount += 1


                    img = self.blacken_box(img, new_coords_blk_list)
                    h_ims[curr_page_idx, panel_no] = img
                    h_pmask[curr_page_idx, panel_no] = 1

def reconstruct_sentence(rvdict, w_, wm_):               
    return ' '.join([rvdict[y] for r,y in enumerate(w_) \
        if wm_[r] == 1])                             


def preview(h5path, raw_pages_path, vocab_dict_path, fold='train'):
    # open the file, display all contents of the page. 
    # also display the page image itself.
    f = h5.File(h5path, 'r')

    total_pages = len(f[fold+'/book_ids'])
    samples = np.random.randint(0, high=total_pages, size=10)
    for sample in samples:
        comic_no = int(np.array(f[fold+'/book_ids'][sample]))
        page_no = int(np.array(f[fold+'/page_ids'][sample]))
        images = np.array(f[fold+'/images'][sample])
        words = np.array(f[fold+'/words'][sample])
        word_masks = np.array(f[fold+'/word_mask'][sample])
        vdict, rvdict = pickle.load(open(vocab_dict_path, 'rb'))

        print 'comic %d, page %d \n----------------' % (comic_no, page_no)
        raw_page_img = cv2.imread(join(raw_pages_path, str(comic_no), str(page_no)+'.jpg'))
        cv2.imshow('raw page image', cv2.resize(raw_page_img, (512, 768)))
        cv2.waitKey(0)

        non_empty_image_idxs = np.where(np.apply_over_axes(np.sum, images, [1, 2, 3]))[0]
        for i in non_empty_image_idxs:
            cv2.imshow('panel %d of %d' % (i, len(non_empty_image_idxs)),
                    images[i].astype('uint8'))
            for j, (word, word_mask) in enumerate(zip(words[i], word_masks[i])):
                print 'panel %d, dialog %d --> %s' % (i, j, reconstruct_sentence(rvdict, word, word_mask))
            print
            cv2.waitKey(0)
        cv2.destroyAllWindows()
 

def main():
    # images_path = '/home/varunm/Comics/data/raw_page_images/'
    panels_path = './data/raw_panel_images/' 
    ocr_file = './data/COMICS_ocr_file.csv' 
    ad_path = './data/predadpages.txt' # these pages have ads according to ad filter
    h5path = './data/comics.h5' # dumping h5 file at this location
    dims = (224, 224) # dimensions of image
    max_panels = 9
    max_boxes = 3
    max_words = 30
    max_vocab_size = 20000
    c = Create_hdf5(panels_path, ocr_file, ad_path, dims,
            max_panels, max_boxes, max_words, max_vocab_size, h5path) 

    c.create_img_hdf5()


if __name__ == '__main__':
    main()
    # preview('comics.h5', '/data/raw_page_images/',
    #   'comics_vocab.p', fold='dev')
