function params = rbm_set_params(dataset,numhid,epsilon,l2reg,pbias,plambda,kcd,maxiter,batchsize,savepath)
% parameters

if ~exist('dataset','var') || isempty(dataset),
    dataset = 'dummy';
end
if ~exist('numhid','var') || isempty(numhid),
    numhid = 1200;
end
if ~exist('epsilon','var') || isempty(epsilon),
    % learning rate
    epsilon = 0.01;
end
if ~exist('pbias','var') || isempty(pbias),
    % target sparsity
    pbias = 0;
end
if ~exist('plambda','var') || isempty(plambda),
    % sparsity regularization coefficient
    plambda = 0;
end
if ~exist('l2reg','var') || isempty(l2reg),
    % weight decay coefficient
    l2reg = 1e-4;
end
if ~exist('kcd','var') || isempty(kcd),
    % number of contrastive divergence steps
    kcd = 1;
end
if ~exist('maxiter','var') || isempty(maxiter),
    maxiter = 100;
end
if ~exist('batchsize','var') || isempty(batchsize),
    batchsize = 200;
end
if ~exist('savepath','var') || isempty(savepath),
    savepath = 'dictionary';
end

params.dataset = dataset;
params.numhid = numhid;
params.epsilon = epsilon;
params.epsdecay = 0.01;
params.l2reg = l2reg;
params.l1reg = 0;
params.pbias = pbias;
params.plambda = plambda;
params.kcd = kcd;
params.maxiter = maxiter;
params.batchsize = batchsize;
params.savepath = savepath;
if ~exist(params.savepath,'dir'),
    mkdir(params.savepath);
end

return;