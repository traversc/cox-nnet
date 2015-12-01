#This is a package for running the neural network extension of Cox regression.  
#Classes: 
#   CoxRegression - the Cox regression output layer
#   HiddenLayer - hidden layers between input and output, based on deeplearning.net/tutorial/code/mlp.py
#   CoxMlp - container class for output layer and hidden layers

#Functions:
#   createSharedDataset - helper function to create shared dataset in theano
#   trainCoxMlp - main function for training a cox-nnet model
#   predictNewData - function for predicting new data
#   L2CVSearch - helper function for performing cross-validation on a training set, to select optimal L2 regularization parameter
#   CVLoglikelihood - calculates the cross validation likelihood (using method from Houwelingen et al. 2005)
#   varImportance - calculates variable importance (using method from Fischer 2015)
#   saveModel - saves a model to a file: saveModel(model, file_name)
#   loadModel - loads model from file: loadModel(fileName)



import time
import numpy
import theano
import random
import theano.tensor as T
from sklearn import cross_validation
import cPickle

theano.config.openmp=True

class CoxRegression(object):
    def __init__(self, input, n_in):
        self.W = theano.shared(value=numpy.zeros((n_in,1),dtype=theano.config.floatX), name='W_cox',borrow=True)
        b_values = numpy.zeros((1,), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, name='b_cox', borrow=True) #intercept term is unnecessary, but doesn't actually matter - so leave it
        self.theta = T.dot(input, self.W) + self.b
        self.theta = T.reshape(self.theta, newshape=[T.shape(self.theta)[0]]) #recast theta as vector
        self.exp_theta = T.exp(self.theta)
        self.params = [self.W, self.b]
        self.input = input

    def negative_log_likelihood(self, R_batch, ystatus_batch):
        return(-T.mean((self.theta - T.log(T.sum(self.exp_theta * R_batch,axis=1))) * ystatus_batch)) #exp_theta * R_batch ~ sum the exp_thetas of the patients with greater time e.g., R(t)
        #e.g., all columns of product will have same value or zero, then do a rowSum

        
#This hidden layer class code is used from the multilayer perceptron code on deeplearning.net
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, activation=T.tanh):
        self.input = input
        W_values = numpy.asarray(
            rng.uniform(
                low=-numpy.sqrt(6. / (n_in + n_out)),
                high=numpy.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)
            ),
            dtype=theano.config.floatX
        )
        W = theano.shared(value=W_values, name='W_' + str(n_in) + '_' + str(n_out), borrow=True)

        b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
        b = theano.shared(value=b_values, name='b_' + str(n_in) + '_' + str(n_out), borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        self.params = [self.W, self.b]

class CoxMlp(object):
    def __init__(self, rng, input, n_in, n_hidden):
        self.hidden_list = []
        self.W = []
        self.b = []
        for i in xrange(len(n_hidden)):
            hidden_layer = HiddenLayer(
            rng=rng,
            input=input if i == 0 else self.hidden_list[i-1].output,
            n_in=n_in if i == 0 else n_hidden[i-1],
            n_out=n_hidden[i],
            activation=T.tanh
            )
            self.hidden_list.append(hidden_layer)
            self.W.append(hidden_layer.W)
            self.b.append(hidden_layer.b)

        self.cox_regression = CoxRegression(
            input=self.hidden_list[-1].output if len(n_hidden) != 0 else input, #last element in hidden_list
            n_in=n_hidden[-1] if len(n_hidden) != 0 else n_in,
        )
        self.W.append(self.cox_regression.W)
        self.b.append(self.cox_regression.b)
        
        self.L0 = 0
        for i in xrange(len(self.W)):
            self.L0 = self.L0 + self.W[i].shape[0] * self.W[i].shape[1]
        
        self.L1 = 0
        for i in xrange(len(self.W)):
            self.L1 = self.L1 + abs(self.W[i]).sum()
        
        self.L2_sqr = 0
        for i in xrange(len(self.W)):
            self.L2_sqr = self.L2_sqr + pow(self.W[i], 2).sum()
        
        self.negative_log_likelihood = self.cox_regression.negative_log_likelihood
        self.input = input 
        
def createSharedDataset(data, borrow=True, cast_int=False):
	shared_data = theano.shared(numpy.asarray(data,dtype=theano.config.floatX),borrow=borrow)
	if cast_int:
		return T.cast(shared_data, 'int32')
	else:
		return shared_data
        
        
def trainCoxMlp(x_train, ytime_train, ystatus_train, L1_reg, L2_reg, n_hidden, learning_rate=0.010, max_iter=10000, stop_threshold=0.995, patience=2000, patience_incr=1.5, rand_seed = 123):

    N_train = ytime_train.shape[0] #number of training examples
    n_in = x_train.shape[1] #number of features
    
    R_matrix_train = numpy.zeros([N_train, N_train], dtype=int)
    for i in range(N_train):
        for j in range(N_train):
            R_matrix_train[i,j] = ytime_train[j] >= ytime_train[i]

    train_x = createSharedDataset(x_train)
    train_R = createSharedDataset(R_matrix_train)
    train_ystatus = createSharedDataset(ystatus_train, cast_int=False)

    rng = numpy.random.RandomState(rand_seed)
     
    index = T.lscalar()
    x = T.matrix('x')
    r = T.matrix('r')
    ystatus = T.vector('ystatus')
    model = CoxMlp(rng = rng, input=x, n_in=n_in, n_hidden = n_hidden)

    cost = (
        model.negative_log_likelihood(r, ystatus)
        + L1_reg * model.L1
        + L2_reg * model.L2_sqr
    )

    #gradiant example from http://deeplearning.net/tutorial/code/mlp.py
    g_W = T.grad(cost=cost, wrt=model.W)
    g_b = T.grad(cost=cost, wrt=model.b)

    updates = [(param, param - gparam * learning_rate) for param, gparam in zip(model.W + model.b, g_W + g_b)]

    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_x,
            ystatus: train_ystatus,
            r: train_R
        },
        on_unused_input='ignore'
    )
    
    start = time.time()
    best_cost = numpy.inf
    print "training model"
    for iter in xrange(max_iter):
        cost_iter = train_model(iter)
        if (iter+1) % 23 == 0:
            if cost_iter > best_cost:
                best_cost = cost_iter
                learning_rate = learning_rate * 0.9 #update learning rate if cost increasing (i.e., bouncing out)
                updates = [(param, param - gparam * learning_rate) for param, gparam in zip(model.W + model.b, g_W + g_b)]
                train_model = theano.function(
                    inputs=[index],
                    outputs=cost,
                    updates=updates,
                    givens={
                        x: train_x,
                        ystatus: train_ystatus,
                        r: train_R
                    },
                    on_unused_input='ignore'
                )
                print (('Adjusting learning rate: %f') % (learning_rate))
            
        if cost_iter < best_cost * stop_threshold:
            best_cost = cost_iter
            print(('cost: %f, iteration: %i') % (best_cost, iter))
            patience = max(patience, iter * patience_incr)
        
        if iter >= patience:
            break
            
        #print cost_iter
    
    print (('running time: %f seconds') % (time.time() - start))
    return(model, cost_iter)


def predictNewData(model, x_test, ytime_test, ystatus_test):
    N_test = ytime_test.shape[0]
    R_matrix_test = numpy.zeros([N_test, N_test], dtype=int)
    for i in range(N_test):
        for j in range(N_test):
            R_matrix_test[i,j] = ytime_test[j] >= ytime_test[i]
    
    theta = x_test
    for i in xrange(len(model.W)):
        w = model.W[i].eval()
        b = model.b[i].eval()
        theta = numpy.dot(theta, w) + b
        if (i+1) != len(model.W):
            theta = numpy.tanh(theta)
        
    theta = theta[:,0]
    
    return(theta)
    
#computes partial log likelihood of validation set as PL_validation = PL_full(beta) - PL_train_cv(beta)
def CVLoglikelihood(model, x_full, ytime_full, ystatus_full, x_train, ytime_train, ystatus_train):
    N_full = ytime_full.shape[0]
    R_matrix_full = numpy.zeros([N_full, N_full], dtype=int)
    for i in range(N_full):
        for j in range(N_full):
            R_matrix_full[i,j] = ytime_full[j] >= ytime_full[i]
    
    theta = x_full
    for i in xrange(len(model.W)):
        w = model.W[i].eval()
        b = model.b[i].eval()
        theta = numpy.dot(theta, w) + b
        if (i+1) != len(model.W):
            theta = numpy.tanh(theta)
        
    theta = theta[:,0]
    exp_theta = numpy.exp(theta)
    PL_full = numpy.sum((theta - numpy.log(numpy.sum(exp_theta * R_matrix_full,axis=1))) * ystatus_full)
    

    N_train = ytime_train.shape[0]
    R_matrix_train = numpy.zeros([N_train, N_train], dtype=int)
    for i in range(N_train):
        for j in range(N_train):
            R_matrix_train[i,j] = ytime_train[j] >= ytime_train[i]
    
    theta = x_train
    for i in xrange(len(model.W)):
        w = model.W[i].eval()
        b = model.b[i].eval()
        theta = numpy.dot(theta, w) + b
        if (i+1) != len(model.W):
            theta = numpy.tanh(theta)
        
    theta = theta[:,0]
    exp_theta = numpy.exp(theta)
    PL_train = numpy.sum((theta - numpy.log(numpy.sum(exp_theta * R_matrix_train,axis=1))) * ystatus_train)
    
    return(PL_full - PL_train)

def crossValidate(x_train, ytime_train, ystatus_train, n_hidden, learning_rate=0.01, max_iter=10000, stop_threshold=0.995, patience=2000, patience_incr=1.5, L2_reg = numpy.exp(-3), L1_reg = 0, rand_seed = 123, cv_seed=1, n_folds=10):
    N_train = ytime_train.shape[0]
    cv_likelihoods = numpy.zeros([n_folds], dtype=float)
    cv_folds=cross_validation.KFold(N_train,n_folds=n_folds, shuffle=True, random_state=cv_seed)
    k=0
    for traincv, testcv in cv_folds:
        x_train_cv = x_train[traincv]
        ytime_train_cv = ytime_train[traincv]
        ystatus_train_cv = ystatus_train[traincv]

        model, cost_iter = trainCoxMlp(x_train = x_train_cv, ytime_train = ytime_train_cv, ystatus_train = ystatus_train_cv, L1_reg = L1_reg, L2_reg = L2_reg, n_hidden = n_hidden, learning_rate=learning_rate, max_iter=max_iter, stop_threshold=stop_threshold, patience=patience, patience_incr=patience_incr, rand_seed = rand_seed)
        
        x_test_cv = x_train[testcv]
        ytime_test_cv = ytime_train[testcv]
        ystatus_test_cv = ystatus_train[testcv]
        
        loglikelihood = CVLoglikelihood(model, x_train, ytime_train, ystatus_train, x_train_cv, ytime_train_cv, ystatus_train_cv)
        cv_likelihoods[k] = loglikelihood
        k += 1
        
    return(cv_likelihoods)

        
def L2CVSearch(x_train, ytime_train, ystatus_train, n_hidden, learning_rate=0.01, max_iter=10000, stop_threshold=0.995, patience=2000, patience_incr=1.5, L2_reg_params = numpy.exp([-6,-5,-4,-3,-2,-1]), rand_seed = 123, cv_seed=1, n_folds=10):
    
    N_train = ytime_train.shape[0]
    
    cv_likelihoods = numpy.zeros([0, n_folds], dtype=float)
    for i in xrange(len(L2_reg_params)):
        L2_reg = L2_reg_params[i]        
        cv_res = crossValidate(x_train, ytime_train, ystatus_train, n_hidden, learning_rate=learning_rate, max_iter=max_iter, stop_threshold=stop_threshold, patience=patience, patience_incr=patience_incr, L2_reg = L2_reg, L1_reg = 0, rand_seed = rand_seed, cv_seed=cv_seed, n_folds=n_folds)
        cv_likelihoods = numpy.concatenate((cv_likelihoods, [cv_res]), axis=0)

    return(cv_likelihoods)



def L2CVSearch2(x_train, ytime_train, ystatus_train, n_hidden, learning_rate=0.01, max_iter=10000, stop_threshold=0.995, patience=2000, patience_incr=1.5, rand_seed = 100, rand_mlp_seed = 123, cv_seed=1, n_folds=5, search_iters = 5, L2_range = [-7,1]):
    
    N_train = ytime_train.shape[0]
    
    cv_likelihoods = numpy.zeros([0, n_folds], dtype=float)
    L2_reg_params = numpy.zeros([0], dtype="float")
    mean_cvpl = numpy.zeros([0], dtype="float")
    search_cvpl = numpy.zeros([0], dtype="float")
    
    left_L2 = L2_range[0]
    right_L2 = L2_range[1]
    
    left_cvpl = crossValidate(x_train, ytime_train, ystatus_train, n_hidden, learning_rate=learning_rate, max_iter=max_iter, stop_threshold=stop_threshold, patience=patience, patience_incr=patience_incr, L2_reg = numpy.exp(left_L2), L1_reg = 0, rand_seed = rand_seed, cv_seed=cv_seed, n_folds=n_folds)
    right_cvpl = crossValidate(x_train, ytime_train, ystatus_train, n_hidden, learning_rate=learning_rate, max_iter=max_iter, stop_threshold=stop_threshold, patience=patience, patience_incr=patience_incr, L2_reg = numpy.exp(right_L2), L1_reg = 0, rand_seed = rand_seed, cv_seed=cv_seed, n_folds=n_folds)

    cv_likelihoods = numpy.concatenate((cv_likelihoods, [left_cvpl]), axis=0)
    cv_likelihoods = numpy.concatenate((cv_likelihoods, [right_cvpl]), axis=0)
    
    left_cvpl_mean = numpy.mean(left_cvpl)
    right_cvpl_mean = numpy.mean(right_cvpl)
    
    mean_cvpl = numpy.append(mean_cvpl,left_cvpl_mean)
    mean_cvpl = numpy.append(mean_cvpl,right_cvpl_mean)

    search_cvpl = numpy.append(search_cvpl,left_cvpl_mean)
    search_cvpl = numpy.append(search_cvpl,right_cvpl_mean)
    
    L2_reg_params = numpy.append(L2_reg_params,left_L2)
    L2_reg_params = numpy.append(L2_reg_params,right_L2)

    #take two highest points and search between them iteratively
    for i in xrange(search_iters):
        idx = numpy.argsort(search_cvpl)
        mid_L2 = numpy.mean([L2_reg_params[idx[-1]], L2_reg_params[idx[-2]]])
        mid_cvpl = crossValidate(x_train, ytime_train, ystatus_train, n_hidden, learning_rate=learning_rate, max_iter=max_iter, stop_threshold=stop_threshold, patience=patience, patience_incr=patience_incr, L2_reg = numpy.exp(mid_L2), L1_reg = 0, rand_seed = rand_seed, cv_seed=cv_seed, n_folds=n_folds)
        mid_cvpl_mean = numpy.mean(mid_cvpl)  
        
        if mid_cvpl_mean < search_cvpl[idx[-2]]:
            search_cvpl[idx[-2]] = -numpy.inf
        
        cv_likelihoods = numpy.concatenate((cv_likelihoods, [mid_cvpl]), axis=0)
        mean_cvpl = numpy.append(mean_cvpl,mid_cvpl_mean)
        search_cvpl = numpy.append(search_cvpl,mid_cvpl_mean)
        L2_reg_params = numpy.append(L2_reg_params,mid_L2)
        
    idx = numpy.argsort(L2_reg_params)
    return(cv_likelihoods[idx], L2_reg_params[idx], mean_cvpl[idx])
    

def varImportance(model, x_train, ytime_train, ystatus_train):
    N_train = ytime_train.shape[0]
    R_matrix_train = numpy.zeros([N_train, N_train], dtype=int)
    for i in range(N_train):
        for j in range(N_train):
            R_matrix_train[i,j] = ytime_train[j] >= ytime_train[i]
    
    theta = x_train
    for i in xrange(len(model.W)):
        w = model.W[i].eval()
        b = model.b[i].eval()
        theta = numpy.dot(theta, w) + b
        if (i+1) != len(model.W):
            theta = numpy.tanh(theta)
        
    theta = theta[:,0]
    exp_theta = numpy.exp(theta)
    PL_train = numpy.sum((theta - numpy.log(numpy.sum(exp_theta * R_matrix_train,axis=1))) * ystatus_train)
    
    PL_mod = numpy.zeros([x_train.shape[1]])
    for k in xrange(x_train.shape[1]):
        if (k+1) % 100 == 0:
            print str(k+1) + "..."
            
        xk_mean = numpy.mean(x_train[:,k])
        theta = numpy.copy(x_train)
        theta[:,k] = xk_mean
    
        for i in xrange(len(model.W)):
            w = model.W[i].eval()
            b = model.b[i].eval()
            theta = numpy.dot(theta, w) + b
            if (i+1) != len(model.W):
                theta = numpy.tanh(theta)
            
        theta = theta[:,0]
        exp_theta = numpy.exp(theta)
        PL_mod[k] = numpy.sum((theta - numpy.log(numpy.sum(exp_theta * R_matrix_train,axis=1))) * ystatus_train)
        print PL_train - PL_mod[k]
        
    return(PL_train - PL_mod)
    
def saveModel(model, file_name):
    b = map(lambda tvar : tvar.eval(), model.b)
    W = map(lambda tvar : tvar.eval(), model.W)
    cPickle.dump( (W,b), open( file_name, "wb" ))
    
def loadModel(file_name):
    f = file(file_name, 'rb')
    W,b = cPickle.load(f)
    f.close()
    n_in = W[0].shape[0]
    n_hidden = map(lambda W : W.shape[1], W[:-1])
    rng = numpy.random.RandomState(123)
    x = T.matrix('x')
    model = CoxMlp(rng = rng, input=x, n_in=n_in, n_hidden = n_hidden)
    for i in xrange(len(W)):
        model.b[i].set_value(b[i])
        model.W[i].set_value(W[i])
    
    return(model)
    

