import sys
sys.path.append("/home/travers/WindowsDesktop/cox-nnet/cox_nnet")

from cox_nnet import *
import numpy
import sklearn

# load data
x = numpy.loadtxt(fname="KIRC/log_counts.csv.gz",delimiter=",",skiprows=0)
ytime = numpy.loadtxt(fname="KIRC/ytime.csv",delimiter=",",skiprows=0)
ystatus = numpy.loadtxt(fname="KIRC/ystatus.csv",delimiter=",",skiprows=0)

# split into test/train sets
x_train, x_test, ytime_train, ytime_test, ystatus_train, ystatus_test = \
    sklearn.cross_validation.train_test_split(x, ytime, ystatus, test_size = 0.2, random_state = 1)

# split training into optimization and validation sets
x_opt, x_validation, ytime_opt, ytime_validation, ystatus_opt, ystatus_validation = \
    sklearn.cross_validation.train_test_split(x_train, ytime_train, ystatus_train, test_size = 0.2, random_state = 123)

# set parameters
model_params = dict(node_map = None, input_split = None)
search_params = dict(method = "nesterov", learning_rate=0.01, momentum=0.9, 
    max_iter=4000, stop_threshold=0.995, patience=1000, patience_incr=2, 
    rand_seed = 123, eval_step=23, lr_decay = 0.9, lr_growth = 1.0)
cv_params = dict(cv_metric = "cindex", L2_range = numpy.arange(-3,1.67,0.33))

#profile log likelihood to determine lambda parameter
likelihoods, L2_reg_params = L2Profile(x_opt,ytime_opt,ystatus_opt,
    x_validation,ytime_validation,ystatus_validation,
    model_params, search_params, cv_params, verbose=False)

numpy.savetxt("KIRC_cindex.csv", likelihoods, delimiter=",")

#build model based on optimal lambda parameter
L2_reg = L2_reg_params[numpy.argmax(likelihoods)]
model_params = dict(node_map = None, input_split = None, L2_reg=numpy.exp(L2_reg))
model, cost_iter = trainCoxMlp(x_train, ytime_train, ystatus_train, model_params, search_params, verbose=True)

theta = model.predictNewData(x_test)
numpy.savetxt("KIRC_theta.csv", theta, delimiter=",")
numpy.savetxt("KIRC_ytime_test.csv", ytime_test, delimiter=",")
numpy.savetxt("KIRC_ystatus_test.csv", ystatus_test, delimiter=",")

