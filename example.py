from cox_nnet import *
import numpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#training data (80%)
x_train = numpy.loadtxt(open("WIHS/x_train.csv","rb"),delimiter=",",skiprows=0)
ytime_train = numpy.loadtxt(open("WIHS/ytime_train.csv","rb"),delimiter=",",skiprows=0)
ystatus_train = numpy.loadtxt(open("WIHS/ystatus_train.csv","rb"),delimiter=",",skiprows=0)

#test data (20% holdout)
x_test = numpy.loadtxt(open("WIHS/x_test.csv","rb"),delimiter=",",skiprows=0)
ytime_test = numpy.loadtxt(open("WIHS/ytime_test.csv","rb"),delimiter=",",skiprows=0)
ystatus_test = numpy.loadtxt(open("WIHS/ystatus_test.csv","rb"),delimiter=",",skiprows=0)

#model parameters
N_train = ytime_train.shape[0] #number of training examples
n_in = x_train.shape[1] #number of features
n_hidden = [n_in ** 0.5] #array of number of hidden nodes in each layer

#training parameters
max_iter = 10000 #Maximum number of iterations
stop_threshold = 0.995 #factor for considering new best cost
patience = 2000 #minimum number of iterations
patience_incr = 1.5 #number of minimum iterations to increase every time a new best is found (i.e., 2 * current_iteration)
L1_reg = 0 #L1 regularization parameter
learning_rate = 0.01 #Starting learning rate
n_folds = 10 #Folds for cross validation

#cross validate training set to determine lambda parameter
cv_likelihoods, L2_reg_params, mean_cvpl = L2CVSearch2(x_train = x_train, ytime_train = ytime_train, ystatus_train = ystatus_train, n_hidden = n_hidden, learning_rate=learning_rate, max_iter=max_iter, stop_threshold=stop_threshold, patience=patience, patience_incr=patience_incr, rand_seed=123, cv_seed=1, n_folds=n_folds)

L2_reg = L2_reg_params[numpy.argmax(mean_cvpl)]
cvpl_best = mean_cvpl[numpy.argmax(mean_cvpl)]

plt.plot(L2_reg_params, mean_cvpl, 'ro')
plt.title("L2: exp " + str(L2_reg) + ", cv loglik: " + str(cvpl_best))
plt.xlabel("log lambda")
plt.ylabel("log likelihood")
plt.margins(0.05, 0.1)
plt.savefig("log_likelihoods.png")

#build model

model, cost_iter = trainCoxMlp(x_train = x_train, ytime_train = ytime_train, ystatus_train = ystatus_train, L1_reg = L1_reg, L2_reg = numpy.exp(L2_reg), n_hidden = n_hidden, learning_rate=learning_rate, max_iter=max_iter, stop_threshold=stop_threshold, patience=patience, patience_incr=patience_incr)

theta = predictNewData(model, x_test, ytime_test, ystatus_test)
numpy.savetxt("theta.csv", theta, delimiter=",")

