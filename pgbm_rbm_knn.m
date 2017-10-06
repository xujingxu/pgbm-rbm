%%% demo code for point-wise gated Boltzmann machines (PGBM)
%   on variations of MNIST datasets.
%
%   the pipeline is given as follows:
%   1. load data
%   2. pretrain RBM
%   3. train PGBM with pretrained RBM as an initialization
%   4. evaluate the performance of PGBM using linear SVM (liblinear)
%   5. write the results on log file


startup;
dataset = 'mnist_bgrand';
usejacket = 0;

%%% 1. load data
[xtrain, ytrain, xval, yval] = load_mnist(dataset);


%%% 2. pretrain RBM
hyperpars_rbm;
params = rbm_set_params(dataset,numhid,epsilon,l2reg,pbias,plambda,kcd,maxiter,batchsize,savepath);
params.numvis = size(xtrain,1);
[w_rbm, params] = rbm_train(xtrain, params, usejacket);


%%% 3. train PGBM
hyperpars_pgbm;
params = pgbm_set_params(dataset,numhid1,numhid2,epsilon,l2reg,pbias,plambda,kcd,ngibbs,use_meanfield,maxiter,batchsize,savepath);
params.numvis = size(xtrain,1);
fname = sprintf('pgbm_%s_vis%d_hid1_%02d_hid2_%02d_eps%g_l2reg%g_pb%g_pl%g_kcd%d_ngibbs%d_usemf%d_iter%d', ...
    params.dataset, params.numvis, params.numhid1, params.numhid2, params.epsilon, params.l2reg, params.pbias, params.plambda, params.kcd, params.ngibbs, params.use_meanfield, params.maxiter);
[w_pgbm, params] = pgbm_train(xtrain, params, w_rbm, ytrain, xval, yval, usejacket);


%%% 4. test
[~, ~, ~, ~, xtest, ytest] = load_mnist(dataset);
ztrain = pgbm_inference(xtrain, w_pgbm, params);
zval = pgbm_inference(xval, w_pgbm, params);
ztest = pgbm_inference(xtest, w_pgbm, params);

%%% 5. train RBM

hyperpars_rbm2;
params = rbm_set_params(dataset,numhid,epsilon,l2reg,pbias,plambda,kcd,maxiter,batchsize,savepath);
params.numvis = size(ztrain,1);
[w_rbm, params] = rbm_train(ztrain, params, usejacket);

ztrain = rbm_inference(ztrain, w_rbm, params);
zval = rbm_inference(zval, w_rbm, params);
ztest = rbm_inference(ztest, w_rbm, params);


%%% 6.KNN
ztrain = double(ztrain');
zval = double(zval');
ztest = double(ztest');

md1 = fitcknn(double(ztrain),ytrain);
label = predict(md1,ztest);% label for after classification






















