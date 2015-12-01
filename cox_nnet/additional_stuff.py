
def varImportance(model, x_train, ytime_train, ystatus_train, groups=None):
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
        
    return(PL_train - PL_mod)
   