from cox_nnet import *
import numpy
import sklearn

# load data
x = numpy.loadtxt(fname="PBC/x.csv",delimiter=",",skiprows=0)
ytime = numpy.loadtxt(fname="PBC/ytime.csv",delimiter=",",skiprows=0)
ystatus = numpy.loadtxt(fname="PBC/ystatus.csv",delimiter=",",skiprows=0)

# split into test/train sets
x_train, x_test, ytime_train, ytime_test, ystatus_train, ystatus_test = \
    sklearn.cross_validation.train_test_split(x, ytime, ystatus, test_size = 0.2, random_state = 100)

#Define parameters
model_params = dict(node_map = None, input_split = None)
search_params = dict(method = "nesterov", learning_rate=0.01, momentum=0.9,
    max_iter=2000, stop_threshold=0.995, patience=1000, patience_incr=2, rand_seed = 123,
    eval_step=23, lr_decay = 0.9, lr_growth = 1.0)
cv_params = dict(cv_seed=1, n_folds=5, cv_metric = "loglikelihood", L2_range = numpy.arange(-4.5,1,0.5))

#cross validate training set to determine lambda parameters
cv_likelihoods, L2_reg_params, mean_cvpl = L2CVProfile(x_train,ytime_train,ystatus_train,
    model_params,search_params,cv_params, verbose=False)

numpy.savetxt("PBC_cv_likelihoods.csv", cv_likelihoods, delimiter=",")


#build model based on optimal lambda parameter
L2_reg = L2_reg_params[numpy.argmax(mean_cvpl)]
model_params = dict(node_map = None, input_split = None, L2_reg=numpy.exp(L2_reg))
model, cost_iter = trainCoxMlp(x_train, ytime_train, ystatus_train, model_params, search_params, verbose=True)

theta = model.predictNewData(x_test)

numpy.savetxt("PBC_theta.csv", theta, delimiter=",")
numpy.savetxt("PBC_ytime_test.csv", ytime_test, delimiter=",")
numpy.savetxt("PBC_ystatus_test.csv", ystatus_test, delimiter=",")

