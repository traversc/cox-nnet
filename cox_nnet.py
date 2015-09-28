import sys
import time
import numpy
import theano
import random
import theano.tensor as T
from sklearn import cross_validation

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
        
        
def trainCoxMlp(x_train, ytime_train, ystatus_train, L1_reg, L2_reg, netmap, learning_rate=0.010, max_iter=10000, stop_threshold=0.995, patience=2000, patience_incr=1.5, rand_seed = 123):

    N_train = ytime_train.shape[0] #number of training examples
    n_in = x_train.shape[1] #number of features
    n_hidden = []
    for i in xrange(len(netmap)):
        n_hidden.append(netmap[i].shape[1])
    
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
    model = CoxMlp(rng = rng, input=x, n_in=n_in, n_hidden = n_hidden, network_mapping = netmap)

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
        
    theta = theta[:,0]
    exp_theta = numpy.exp(theta)
    PL_train = numpy.sum((theta - numpy.log(numpy.sum(exp_theta * R_matrix_train,axis=1))) * ystatus_train)
    
    return(PL_full - PL_train)


def L2CVSearch(x_train, ytime_train, ystatus_train, netmap, learning_rates=None, max_iter=10000, stop_threshold=0.995, patience=2000, patience_incr=1.5, L2_reg_params = numpy.exp([-6,-5,-4,-3,-2,-1]), rand_seed = 100, cv_seed=1, n_folds=10):
    
    N_train = ytime_train.shape[0]
    
    cv_likelihoods = numpy.zeros([len(L2_reg_params), n_folds], dtype=float)
    #cv_PL_full = numpy.zeros([len(L2_reg_params), n_folds], dtype=float)
    #cv_PL_train = numpy.zeros([len(L2_reg_params), n_folds], dtype=float)
    random.seed(cv_seed)
    cv=cross_validation.KFold(N_train,n_folds=n_folds, shuffle=True, random_state=1)
    
    for i in xrange(len(L2_reg_params)):
        L2_reg = L2_reg_params[i]
        learning_rate = 0.01 if learning_rates == None else learning_rates[i]
        #print learning_rate
        k = 0
        for traincv, testcv in cv:

            x_train_cv = x_train[traincv]
            ytime_train_cv = ytime_train[traincv]
            ystatus_train_cv = ystatus_train[traincv]

            model, cost_iter = trainCoxMlp(x_train = x_train_cv, ytime_train = ytime_train_cv, ystatus_train = ystatus_train_cv, L1_reg = 0, L2_reg = L2_reg, netmap = netmap, learning_rate=learning_rate, max_iter=max_iter, stop_threshold=stop_threshold, patience=patience, patience_incr=patience_incr, rand_seed = rand_seed)
            
            x_test_cv = x_train[testcv]
            ytime_test_cv = ytime_train[testcv]
            ystatus_test_cv = ystatus_train[testcv]
            
            loglikelihood = CVLoglikelihood(model, x_train, ytime_train, ystatus_train, x_train_cv, ytime_train_cv, ystatus_train_cv)
            cv_likelihoods[i,k] = loglikelihood
            k += 1
            
    return(cv_likelihoods)



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
        
    theta = theta[:,0]
    exp_theta = numpy.exp(theta)
    PL_train = numpy.sum((theta - numpy.log(numpy.sum(exp_theta * R_matrix_train,axis=1))) * ystatus_train)
    
    PL_mod = numpy.zeros([x_train.shape[1]])
    for k in xrange(x_train.shape[1]):
        xk_mean = numpy.mean(x_train[:,k])
        theta = numpy.copy(x_train)
        theta[:,k] = xk_mean
    
        for i in xrange(len(model.W)):
            w = model.W[i].eval()
            b = model.b[i].eval()
            theta = numpy.dot(theta, w) + b
            
        theta = theta[:,0]
        exp_theta = numpy.exp(theta)
        PL_mod[k] = numpy.sum((theta - numpy.log(numpy.sum(exp_theta * R_matrix_train,axis=1))) * ystatus_train)
        
    return(PL_train - PL_mod)
    
