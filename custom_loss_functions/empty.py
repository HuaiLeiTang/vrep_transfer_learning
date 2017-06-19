import theano.tensor as T

# This does not work as the primary loss in Theano, will get DisconnectedInput error
def empty_constant(y_true,y_pred):
    return T.constant(0)


# This on the other hand, which has the same meaning as the above function, works
def empty(y_true,y_pred):
    return T.mean(y_pred) - T.mean(y_pred)

