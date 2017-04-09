# COMICS
code to download comics data and train models described in [The Amazing Mysteries of the Gutter: Drawing Inferences Between Panels in Comic Book Narratives.](https://arxiv.org/abs/1611.05118)

email miyyer@umd.edu and varunm@cs.umd.edu with any comments/problems/questions/suggestions.

### dependencies: 
* requires python 2.7, lasagne, theano, h5py, cv2, glob2

### to download / unzip / preprocess COMICS data:
* bash setup.sh (downloads raw panel images, OCR transcriptions, etc., and preprocesses them into an hdf5 file)
 * if you don't want to download everything at once, you can download individual files at <https://obj.umiacs.umd.edu/comics/index.html>.
 
### to train models after preprocessing (example for text cloze):
* python models/text_cloze.py (make sure to run on GPU; see run.sh for our theano flags)
 * see description of hyperparameters by running python models/text_cloze.py --help
 * note that low-quality data is only filtered out in dev/test data (by throwing out examples with too many UNK tokens). during training, all data is used. 

### results:

|   method   | text cloze easy | text cloze hard | visual cloze easy | visual cloze hard | character coherence 
|:----------:|-----------------|-----------------|-------------------|-------------------|-------------------|
|  text only |     63.4    |     52.9    |      55.9     |      48.4     |     68.2     |
| image only |     51.7    |     49.4    |      85.7     |      63.2     |     70.9     |
| image text |      68.6     |     61.0    |       81.3      |      59.1     |     69.3     |

if you use the COMICS data and/or code, please cite:
    
    @InProceedings{Iyyer:Manjunatha-Comics2017,
        Title = {The Amazing Mysteries of the Gutter: Drawing Inferences Between Panels in Comic Book Narratives},
        Booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
        Author = {Mohit Iyyer and Varun Manjunatha and Anupam Guha and Yogarshi Vyas and Jordan Boyd-Graber and Hal {Daum\'{e} III} and Larry Davis},
        Year = {2017},
    }

