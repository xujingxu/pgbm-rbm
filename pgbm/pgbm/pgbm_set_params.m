function params = pgbm_set_params(dataset,numhid1,numhid2,epsilon,l2reg,pbias,plambda,kcd,ngibbs,use_meanfield,maxiter,batchsize,savepath)
% parameters

if ~exist('dataset','var') || isempty(dataset),
    % number of task-relevant hidden unit
    dataset = 'dummy';
end
if ~exist('numhid1','var') || isempty(numhid1),
    % number of task-relevant hidden unit
    numhid1 = 500;
end
if ~exist('numhid2','var') || isempty(numhid2),
    % number of task-irrelevant hidden unit
    numhid2 = numhid1;
end
if ~exist('epsilon','var') || isempty(epsilon),
    % learning rate
    epsilon = 0.01;
end
if ~exist('pbias','var') || isempty(pbias),
    % target sparsity
    pbias = 0.02;
end
if ~exist('plambda','var') || isempty(plambda),
    % sparsity regularization coefficient
    plambda = 1;
end
if ~exist('l2reg','var') || isempty(l2reg),
    % weight decay coefficient
    l2reg = 1e-3;
end
if ~exist('kcd','var') || isempty(kcd),
    % number of contrastive divergence steps
    kcd = 1;
end
if ~exist('ngibbs','var') || isempty(ngibbs),
    % number of iteration for gibbs / mean-field approximation
    ngibbs = 1;
end
if ~exist('use_meanfield','var') || isempty(use_meanfield),
    % 1 for mean-field / 0 for gibbs sampling
    use_meanfield = 1;
end
if ~exist('maxiter','var') || isempty(maxiter),
    maxiter = 100;
end
if ~exist('batchsize','var') || isempty(batchsize),
    batchsize = 100;
end
if ~exist('savepath','var') || isempty(savepath),
    savepath = 'dictionary';
end

params.dataset = dataset;
params.numhid1 = numhid1;
params.numhid2 = numhid2;
params.epsilon = epsilon;
params.epsdecay = 0.01;
params.momch = 5;
params.l2reg = l2reg;
params.l1reg = 0;
params.pbias = pbias;
params.plambda = plambda;
params.kcd = kcd;
params.ngibbs = ngibbs;
params.use_meanfield = use_meanfield;
params.maxiter = maxiter;
params.batchsize = batchsize;
params.savepath = savepath;
if ~exist(params.savepath,'dir'),
    mkdir(params.savepath);
end

return;