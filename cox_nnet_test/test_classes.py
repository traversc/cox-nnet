if model.input_split == None:
    test_input = [createSharedDataset(x_test)]
else:
    test_input = [0] * len(model.input_split)
    for i in range(len(model.input_split)):
        test_input[i] = createSharedDataset(x_test[:,model.input_split[i]])

theta = test_input
for i in xrange(len(model.hidden_list)):
    theta = model.hidden_list[i].evalNewData(theta)

theta = model.cox_regression.evalNewData(theta).eval()
return(theta[:,0])


output = [0] * len(model.hidden_list[i].W)
for i in xrange(len(model.hidden_list[i].W)):
    input_cat_i = test_data[model.hidden_list[i].map[i][1][0]] if len(model.hidden_list[i].map[i][1]) == 1 else numpy.concatenate(test_data[model.hidden_list[i].map[i][1]], axis=1)
    output[i] = T.dot(input_cat_i, model.hidden_list[i].W[i]) + model.hidden_list[i].b[i]
    