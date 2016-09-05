# coding: utf-8
#define the model
num_class = 10
num_features = x_train.shape[1]

l_in = InputLayer(shape=(None,num_features))
l_dropout_input = DropoutLayer(incoming=l_in,p=0.0)
#l_noise_input = GaussianNoiseLayer(incoming=l_dropout_input,sigma=0.1)
l_hid_1 = DenseLayer(incoming=l_in, num_units=800, nonlinearity=elu)
l_dropout_1 = DropoutLayer(incoming=l_hid_1,p=0.0)
l_hid_2 = DenseLayer(incoming=l_dropout_1, num_units=400, nonlinearity=elu)
l_dropout_2 = DropoutLayer(incoming=l_hid_2,p=0.0)
l_hid_3 = DenseLayer(incoming=l_dropout_2,num_units=100,nonlinearity=elu)

l_out = DenseLayer(incoming=l_hid_3, num_units=num_class, nonlinearity=softmax)

layers = {l_hid_1: 1e-4 , l_hid_2:1e-4, l_hid_3:1e-4}
