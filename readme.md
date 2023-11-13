# README

## Dataset
20NG: http://ana.cachopo.org/datasets-for-single-label-textcategorization
AG News: https://paperswithcode.com/dataset/ag-news
THUCNews: http://thuctc.thunlp.org/
R8: https://www.cs.umb.edu/∼smimarog/textmining/datasets/
MR: http://www.cs.cornell.edu/people/pabo/movie-review-data/

## Environment
python=3.9.18=h4de0772_0_cpython
jieba=0.42.1=pyhd8ed1ab_0
matplotlib=3.8.0=pypi_0
numpy=1.26.0=pypi_0
openssl=3.1.3=hcfcfb64_0
transformers=4.34.1=pypi_0
pytorch-pretrained-bert=0.6.2=pypi_0

## File description

|---data Dataset storage path
    |---R8
    |---THUCNews
        |---corpus After data cleaning corpus, store training text, text labels, word vector files, degree matrix, weight adjacency matrix and so on
        |---train Uncleaned training corpus
        |---test Uncleaned test data
    |---config.yaml A data-related configuration file that stores configuration items such as training set and test set path
|---model    Model storage path
    |---layer.py    Model file, single layer network
    |---model.py    Model file, multi-layer integrated network, contains the complete model network
    |---config.yaml A model-related configuration file that stores configuration items such as model hyperparameters
    |---bert_base_uncased_pytorch   bert model folder
    |---chinese_wwm_ext_pytorch bert model folder
|---preprocess   Store preprocessed files
    |---build_dataset.py    Extract parts of the original (relatively large) data set to form a new small data set
    |---remove_words.py Clean the corpus, remove stops, and write all documents into a single text file, where each line is a document
    |---build_graph.py  The adjacency matrix A, A_1, A_2, A_3, degree matrix, word vector table and so on are constructed based on data enhancement
    |---get_embedding_cos.py    Gets the bert embedding word vector
    |---utils.py    Utility function scripts containing multiple interface functions (cleaning text, segmentation, and so on)
|---loader.py   Data is read into the script file
|---train.py    Training data script file
|---condalist.txt   Project environment, required python packages and their versions

## Operational item

1. Modify 'num_per_class' in 'data/config.yaml' file (this item is the number of samples extracted from each category, default is 50) and 'raw_dataset' (this item is the absolute path to the original dataset), Then run 'preprocess/build_dataset.py', 'python build_dataset.py';
2. Run 'preprocess/remove_words.py' to remove stops in the corpus, clean the corpus and write all training text and labels to a single text file, 'python remove_words.py';
3. Run 'preprocess/build_graph.py', build the graph, 'python build_graph.py';
4. Start training, 'python train.py'