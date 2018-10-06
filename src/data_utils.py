# By Bogo
import numpy as np
import re

# read in data files from io and apply convert_articles and text_to_idx
# to get an np array of size (N, M)
# the file names should be specified by the "filename" argument as the prefix
# each file should consist of one article and be named "filename-{X}", X in [1, N]
# the file type is txt 
# TODO: allow binary files and come up with a better way for article seperation
# input:
#   filename: string of filename (with directory), common prefix
#   N: number of articles
#   mode: 'letter' or 'word', for keyword seperation
# output:
#   indices: int[][], size of (N, M)
#   word_to_idx: dictionary of size V
#   idx_to_word: string[]
def load_data(filename, N, mode='word'):
  articles = []
  for i in range(1, N+1):
    name = filename + '-%d' % i
    with open(name) as f:
      article = f.read()
      articles.append(article)
  texts, word_to_idx, idx_to_word = convert_articles(articles, mode=mode)
  indices = text_to_idx(texts, word_to_idx)
  return indices, word_to_idx, idx_to_word

# NO NEED
# split the data into training set and validation set
# input:
#   data_in: int[][], converted indices of articles
#   train_percent: percentage of training set
# output:
#   data_out: dictionary, key is 'train_texts' and 'valid'
# def split_data(data, train_percent=0.9):


# read in articles and splits the keywords
# two modes are provided:
#   'letter', devide each character (including the space),
#     the word list will be short consequently (basically an alphebet with special characters)
#   'word', devide by space, each character other than letters (namely '\W') will be be marked as an independent keyword,
#     the word list ma be large.
# also place the input inbetween <START> and <END>, with <NULL> do the padding and the end
# input: 
#   articles: string[] of size N
#   mode: 'letter' or 'word'
# return:
#   texts: string[][] of size (N, max(len(articles)))
#   word_to_idx: dictionary of size V
#   idx_to_word: string[]
def convert_articles(articles, mode='word'):
  # input validation
  if not mode in ('letter', 'word'):
    raise ValueError('Invalid mode %s' % mode)
  # init word list
  idx_to_word = ['<NULL>', '<START>', '<END>']
  word_to_idx = {'<NULL>': 0, '<START>': 1, '<END>': 2}
  # split
  N = len(articles)
  max_length = 0
  keywords_arr = []
  for article in articles:
    if mode == 'word':
      pattern = re.compile(r"[\w\-']+|\W+")
      keywords = pattern.findall(article)
    elif mode == 'letter':
      keywords = list(article)
    length = len(keywords)
    max_length = max(max_length, length)
    # update word list
    for word in keywords:
      if not word in word_to_idx.keys():
        idx_to_word.append(word)
        word_to_idx[word] = len(idx_to_word) - 1
    keywords_arr.append(keywords)
  # augment keywords to be the same size
  texts = []
  for keywords in keywords_arr:
    length = len(keywords)
    keywords.extend(['<NULL>'] * (max_length-length))
    texts.append(keywords)
  texts = np.array(texts)
  return texts, word_to_idx, idx_to_word

# convert a batch of texts into a batch of corresponding indices
# input:
# 	texts: string[][], size of (N, M)
#	  word_to_idx: dictionary of word string to index int
# return:
#   indices: int[][], size of (N, M)
def text_to_idx(texts, word_to_idx):
  N, M = texts.shape
  indices = np.ones_like(texts)
  for i in range(N):
    for j in range(M):
      indices[i][j] = word_to_idx[texts[i][j]]
  indices = indices.astype(np.int64)
  return indices
	
# convert a batch of indices into a batch of corresponding articles
# note that the process within each article (namely each row of data) should stop at <END>
# <NULL> should not be rendered
# input:
# 	indices: int[][], size of (N, M)
#	  idx_to_word: string[], mapping from index to word
#   keep_tag: boolean, whether to keep the <START> and <END> tag
# return:
#   articles: string[], size N
def idx_to_article(indices, idx_to_word, keep_tag=False):
  N, M = indices.shape
  articles = []
  for i in range(N):
    article = ''
    for j in range(M):
      word = idx_to_word[indices[i][j]]
      # skip <NULL>
      if word == '<NULL>':
        continue
      # stop at <END>
      if word == '<END>':
        article += word if keep_tag == True else ''
        break
      # whether render <START>
      # <END> is considered at last case
      if keep_tag == False and word == '<START>':
        continue
      article += word + ' '
    articles.append(article)
  return articles
