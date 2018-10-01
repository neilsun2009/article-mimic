import numpy as np

def sigmoid(x):
    """
    A numerically stable version of the logistic sigmoid function.
    """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)

def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """
    Forward pass for a single timestep of an LSTM.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Note that a sigmoid() function has already been provided for you in this file.

    Inputs:
    - x: Input data, of shape (N, D)
    - prev_h: Previous hidden state, of shape (N, H)
    - prev_c: previous cell state, of shape (N, H)
    - Wx: Input-to-hidden weights, of shape (D, 4H)
    - Wh: Hidden-to-hidden weights, of shape (H, 4H)
    - b: Biases, of shape (4H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - next_c: Next cell state, of shape (N, H)
    - cache: Tuple of values needed for backward pass.
    """
    next_h, next_c, cache = None, None, None
    #############################################################################
    # TODO: Implement the forward pass for a single timestep of an LSTM.        #
    # You may want to use the numerically stable sigmoid implementation above.  #
    #############################################################################
    N, H = prev_h.shape
    a = x.dot(Wx) + prev_h.dot(Wh) + b
    ifo = sigmoid(a[:, :3*H])
    input_gate = ifo[:, :H]
    forget_gate = ifo[:, H:2*H]
    output_gate = ifo[:, 2*H:3*H]
    gate_gate = np.tanh(a[:, 3*H:])
    next_c = forget_gate * prev_c + input_gate * gate_gate
    next_h = output_gate * np.tanh(next_c)
    cache = (x, prev_h, prev_c, Wx, Wh, b, a, input_gate, forget_gate, output_gate, gate_gate, next_c, next_h)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
    """
    Backward pass for a single timestep of an LSTM.

    Inputs:
    - dnext_h: Gradients of next hidden state, of shape (N, H)
    - dnext_c: Gradients of next cell state, of shape (N, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data, of shape (N, D)
    - dprev_h: Gradient of previous hidden state, of shape (N, H)
    - dprev_c: Gradient of previous cell state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dprev_h, dprev_c, dWx, dWh, db = None, None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for a single timestep of an LSTM.       #
    #                                                                           #
    # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
    # the output value from the nonlinearity.                                   #
    #############################################################################
    x, prev_h, prev_c, Wx, Wh, b, a, i, f, o, g, next_c, next_h = cache
    dnext_c += dnext_h * o * (1 - np.tanh(next_c) ** 2)
    dprev_c = dnext_c * f
    do = dnext_h * np.tanh(next_c)
    df = dnext_c * prev_c
    di = dnext_c * g
    dg = dnext_c * i
    da = np.hstack((di * i * (1 - i), df * f * (1 - f), do * o * (1 - o), dg * (1 - g**2)))
    db = np.sum(da, axis=0)
    dprev_h = da.dot(Wh.T)
    dWh = prev_h.T.dot(da)
    dx = da.dot(Wx.T)
    dWx = x.T.dot(da)
    # daffine = dnext_h * output_gate * (1 - next_h * next_h)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
    """
    Forward pass for an LSTM over an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the LSTM forward, we return the hidden states for all timesteps.

    Note that the initial cell state is passed as input, but the initial cell
    state is set to zero. Also note that the cell state is not returned; it is
    an internal variable to the LSTM and is not accessed from outside.

    Inputs:
    - x: Input data of shape (N, T, D)
    - h0: Initial hidden state of shape (N, H)
    - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
    - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
    - b: Biases of shape (4H,)

    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
    - cache: Values needed for the backward pass.
    """
    h, cache = None, None
    #############################################################################
    # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
    # You should use the lstm_step_forward function that you just defined.      #
    #############################################################################
    N, T, D = x.shape
    H = h0.shape[1]
    x = np.transpose(x, (1, 0, 2))
    prev_h = h0
    prev_c = np.zeros((N, H))
    h = []
    cache = []
    for i in range(T):
        next_h, next_c, cache_now = lstm_step_forward(x[i], prev_h, prev_c, Wx, Wh, b)
        h.append(next_h)
        prev_h = next_h
        prev_c = next_c
        cache.append(cache_now)
    h = np.transpose(h, (1, 0, 2))
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return h, cache


def lstm_backward(dh, cache):
    """
    Backward pass for an LSTM over an entire sequence of data.]

    Inputs:
    - dh: Upstream gradients of hidden states, of shape (N, T, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data of shape (N, T, D)
    - dh0: Gradient of initial hidden state of shape (N, H)
    - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
    # You should use the lstm_step_backward function that you just defined.     #
    #############################################################################
    N, T, H = dh.shape
    D = cache[0][3].shape[0]
    dh = np.transpose(dh, (1, 0, 2))
    dx = []
    dprev_h, dprev_c, dWx, dWh, db = np.zeros((N, H)), np.zeros((N, H)), np.zeros((D, 4*H)), np.zeros((H, 4*H)), np.zeros(4*H)
    for i in reversed(range(T)):
        dnext_h = dh[i] + dprev_h
        dnext_c = dprev_c
        dx_now, dprev_h, dprev_c, dWx_now, dWh_now, db_now = lstm_step_backward(dnext_h, dnext_c, cache[i])
        dx.append(dx_now)
        dWx, dWh, db = dWx + dWx_now, dWh + dWh_now, db + db_now
        
    dx = np.transpose(dx[::-1], (1, 0, 2))
    dh0 = dprev_h
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dh0, dWx, dWh, db
