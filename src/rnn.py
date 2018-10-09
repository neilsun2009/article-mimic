# Mainly from Stanford CS231n

from src.rnn_layers import *
from src.layers import *
from src.data_utils import *

class RNN():
  """
  A RNN produces a new series of string using a recurrent
  neural network.
  The RNN receives input vectors of size 1, has a vocab size of V, works on
  sequences of length T, has an RNN hidden dimension of H, uses word vectors
  of dimension W, and operates on minibatches of size N.
  Note that we don't use any regularization for the RNN.
  
  Route:
  1. init word embedding layer (vocab_size -> wordvec_dim)
  2. rnn layer (wordvec_dim + hidden_dim -> hidden_dim)
  3. rnn to word (hidden_dim -> vocab_size)
  4. softmax
  """

  def __init__(self, word_to_idx, idx_to_word, wordvec_dim=128,
                hidden_dim=128, dtype=np.float32):
    """
    Construct a new RNN instance.
    Inputs:
    - word_to_idx: A dictionary giving the vocabulary. It contains V entries,
      and maps each string to a unique integer in the range [0, V).
    - wordvec_dim: Dimension W of word vectors.
    - hidden_dim: Dimension H for the hidden state of the RNN.
    - dtype: numpy datatype to use; use float32 for training and float64 for
      numeric gradient checking.
    """
    self.dtype = dtype
    self.word_to_idx = word_to_idx
    self.idx_to_word = idx_to_word
    self.idx_to_word = {i: w for w, i in word_to_idx.items()}
    self.params = {}

    vocab_size = len(word_to_idx)

    self._null = word_to_idx['<NULL>']
    self._start = word_to_idx.get('<START>', None)
    self._end = word_to_idx.get('<END>', None)

    # Initialize word vectors
    self.params['W_embed'] = np.random.randn(vocab_size, wordvec_dim)
    self.params['W_embed'] /= 100

    # Initialize parameters for the RNN
    self.params['Wx'] = np.random.randn(wordvec_dim, 4 * hidden_dim)
    self.params['Wx'] /= np.sqrt(wordvec_dim)
    self.params['Wh'] = np.random.randn(hidden_dim, 4 * hidden_dim)
    self.params['Wh'] /= np.sqrt(hidden_dim)
    self.params['b'] = np.zeros(4 * hidden_dim)

    # Initialize output to vocab weights
    self.params['W_vocab'] = np.random.randn(hidden_dim, vocab_size)
    self.params['W_vocab'] /= np.sqrt(hidden_dim)
    self.params['b_vocab'] = np.zeros(vocab_size)

    # Cast parameters to correct dtype
    for k, v in self.params.items():
        self.params[k] = v.astype(self.dtype)

  def loss(self, texts):
    """
    Compute training-time loss for the RNN. We input a batch of texts, and use an LSTM to compute
    loss and gradients on all parameters.
    Inputs:
    - texts: an integer array of shape (N, T) where
      each element is in the range 0 <= texts[i, t] < V
    Returns a tuple of:
    - loss: Scalar loss
    - grads: Dictionary of gradients parallel to self.params
    """
    # Cut texts into two pieces: texts_in has everything but the last word
    # and will be input to the RNN; texts_out has everything but the first
    # word and this is what we will expect the RNN to generate. These are offset
    # by one relative to each other because the RNN should produce word (t+1)
    # after receiving word t. The first element of texts_in will be the START
    # token, and the first element of texts_out will be the first word.
    texts_in = texts[:, :-1]
    texts_out = texts[:, 1:]

    # You'll need this
    mask = (texts_out != self._null)

    # Word embedding matrix
    W_embed = self.params['W_embed']

    # Input-to-hidden, hidden-to-hidden, and biases for the RNN
    Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']

    # Weight and bias for the hidden-to-vocab transformation.
    W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']

    # shape
    N, T = texts.shape
    H = Wh.shape[0]

    loss, grads = 0.0, {}
    ############################################################################
    # TODO: Implement the forward and backward passes for the CaptioningRNN.   #
    # In the forward pass you will need to do the following:                   #
    # (1) Use a random generator to initialize hidden state.     #
    #     This should produce an array of shape (N, H)#
    # (2) Use a word embedding layer to transform the words in texts_in     #
    #     from indices to vectors, giving an array of shape (N, T, W).         #
    # (3) Use an LSTM to    #
    #     process the sequence of input word vectors and produce hidden state  #
    #     vectors for all timesteps, producing an array of shape (N, T, H).    #
    # (4) Use a (temporal) affine transformation to compute scores over the    #
    #     vocabulary at every timestep using the hidden states, giving an      #
    #     array of shape (N, T, V).                                            #
    # (5) Use (temporal) softmax to compute loss using texts_out, ignoring  #
    #     the points where the output word is <NULL> using the mask above.     #
    #                                                                          #
    # In the backward pass you will need to compute the gradient of the loss   #
    # with respect to all model parameters. Use the loss and grads variables   #
    # defined above to store loss and gradients; grads[k] should give the      #
    # gradients for self.params[k].                                            #
    #                                                                          #                                   #
    ############################################################################
    h0 = np.random.randn(N, H)
    embed, embed_cache = word_embedding_forward(texts_in, W_embed)
    rnn, rnn_cache = lstm_forward(embed, h0, Wx, Wh, b)
    vocab, vocab_cache = temporal_affine_forward(rnn, W_vocab, b_vocab)
    loss, dloss = temporal_softmax_loss(vocab, texts_out, mask)
    dvocab, grads['W_vocab'], grads['b_vocab'] = temporal_affine_backward(dloss, vocab_cache)
    drnn, dh0, grads['Wx'], grads['Wh'], grads['b'] = lstm_backward(dvocab, rnn_cache)
    grads['W_embed'] = word_embedding_backward(drnn, embed_cache)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads
  
  def sample(self, N, starts=None, max_length=300, keep_tag=False):
    """
    Run a test-time forward pass for the model, outputing a result text given the start word.
    At each timestep, we embed the current word, pass it and the previous hidden
    state to the RNN to get the next hidden state, use the hidden state to get
    scores for all vocab words, and choose the word with the highest score as
    the next word. The initial hidden state is randomized and the initial words are to be prepended
    by a <START> token. The actual RNN calculation will start from each last input word of the augmented batch.
    For LSTMs you will also have to keep track of the cell state; in that case
    the initial cell state should be zero.
    Inputs:
    - N: Batch size.
    - starts: A batch of start words of outer size N.
    - max_length: Maximum length T of generated texts.
    - keep_tag: boolean, whether to keep the <START> and <END> tag
    Returns:
    - texts: [][], size N*max_length
    """

    # augment start word
    if starts is None:
      starts = [[self._start]] * N
    else:
      if len(starts) != N:
        raise ValueError("Invalid size of starts, should be %d" % N)
      for i in range(N):
        starts[i].insert(0, self._start)

    # Unpack parameters
    W_embed = self.params['W_embed']
    Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']
    W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']
    H = Wh.shape[0]
    V = W_vocab.shape[1]

    ###########################################################################
    # TODO: Implement test-time sampling for the model. You will need to      #
    # initialize the hidden state of the RNN by applying the learned affine   #
    # transform to the input image features. The first word that you feed to  #
    # the RNN should be the <START> token; its value is stored in the         #
    # variable self._start. At each timestep you will need to do to:          #
    # (1) Embed the previous word using the learned word embeddings           #
    # (2) Make an RNN step using the previous hidden state and the embedded   #
    #     current word to get the next hidden state.                          #
    # (3) Apply the learned affine transformation to the next hidden state to #
    #     get scores for all words in the vocabulary                          #
    # (4) Select the word with the highest score as the next word, writing it #
    #     (the word index) to the appropriate slot in the texts variable   #
    #                                                                         #
    # For simplicity, you do not need to stop generating after an <END> token #
    # is sampled, but you can if you want to.                                 #
    #                                                                         #
    # HINT: You will not be able to use the rnn_forward or lstm_forward       #
    # functions; you'll need to call rnn_step_forward or lstm_step_forward in #
    # a loop.                                                                 #
    #                                                                         #
    # NOTE: we are still working over minibatches in this function. Also if   #
    # you are using an LSTM, initialize the first cell state to zeros.        #
    ###########################################################################
    prev_h = np.random.randn(N, H)
    texts_in = []
    for i in range(N):
      texts_in.append(starts[i][-1])
    # texts_in = (starts[:][-1]).astype(np.int64)
    texts = np.zeros((N, max_length))
    texts[:, 0] = texts_in
    prev_c = np.zeros_like(prev_h)
    # TODO: when <END> appears, do not do further RNN for that article
    for i in range(1, max_length-1):
      embed, _ = word_embedding_forward(texts_in, W_embed)
      next_h, next_c, _ = lstm_step_forward(embed, prev_h, prev_c, Wx, Wh, b)
      vocab, _ = affine_forward(next_h, W_vocab, b_vocab)
      for j in range(N):
        vocab[j] -= np.min(vocab[j]) - 0.1
        vocab[j] = vocab[j] / np.sum(vocab[j])
      # texts_in = np.argmax(vocab, axis=1)
      texts_in = []
      for j in range(N):
        texts_in.append(np.random.choice(V, 1, p=vocab[j])[0])
      texts[:, i] = texts_in
      prev_h = next_h
      prev_c = next_c
    texts[:, max_length-1] = self._end
    texts = texts.astype(np.int64)
    return texts
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################