function [w_rbm, params] = rbm_train(xtrain, params, usejacket)

if ~exist('usejacket','var'),
    usejacket = 0;
end

% random seed
rng('default');

params.numvis = size(xtrain,1);

% momentum
initialmomentum  = 0.5;
finalmomentum    = 0.9;

% initialization
[w_rbm, params] = rbm_init_params(params);

% convert to jacket (gpu) variables
if usejacket,
    w_rbm.vishid = gsingle(w_rbm.vishid);
    w_rbm.visbias = gsingle(w_rbm.visbias);
    w_rbm.hidbias = gsingle(w_rbm.hidbias);
    
    shared.vhinc = gzeros(size(w_rbm.vishid));
    shared.hbiasinc = gzeros(size(w_rbm.hidbias));
    shared.vbiasinc = gzeros(size(w_rbm.visbias));
else
    w_rbm.vishid = single(w_rbm.vishid);
    w_rbm.visbias = single(w_rbm.visbias);
    w_rbm.hidbias = single(w_rbm.hidbias);
    
    shared.vhinc = single(zeros(size(w_rbm.vishid)));
    shared.hbiasinc = single(zeros(size(w_rbm.hidbias)));
    shared.vbiasinc = single(zeros(size(w_rbm.visbias)));
end
shared.runningavg_prob = [];

% monitoring variables
error_history = zeros(params.maxiter,1);
sparsity_history = zeros(params.maxiter,1);

% display parameters
disp(params);


%%% train binary-binary restricted Boltzmann machine
%   using stochastic gradient descent with mini batch
%   using contrastive divergence to approximate gradient

N = size(xtrain,2);
numiter = min(ceil(N/params.batchsize), 1000);

for t = 1:params.maxiter,
    % monitoring variable
    recon_epoch = zeros(numiter, 1);
    sparsity_epoch = zeros(numiter, 1);
    
    % learning rate
    epsilon = params.epsilon;
    
    % momentum
    if t < 5,
        momentum = initialmomentum;
    else
        momentum = finalmomentum;
    end
    
    % shuffle input data order
    randidx = randperm(N);
    
    for b = 1:numiter,
        % input data for current batch
        batchidx = randidx((b-1)*params.batchsize+1:b*params.batchsize);
        xb = xtrain(:, batchidx);
        if usejacket,
            xb = gsingle(xb);
        else
            xb = single(xb);
        end
        
        
        %%% contrastive divergence steps
        vbiasmat = repmat(w_rbm.visbias,1,params.batchsize);
        hbiasmat = repmat(w_rbm.hidbias,1,params.batchsize);
        
        % positive phase: sample h ~ P(h|v)
        poshidprob = sigmoid(w_rbm.vishid'*xb + hbiasmat);
        if usejacket,
            poshidstates = gsingle(rand(size(poshidprob)) < poshidprob);
        else
            poshidstates = single(rand(size(poshidprob)) < poshidprob);
        end
        
        % negative phase: sample v' ~ P(v|h), sample h' ~ P(h|v')
        neghidstates = poshidstates;
        for kcd = 1:params.kcd,
            % negative data
            negdata = sigmoid(w_rbm.vishid*neghidstates + vbiasmat);
            
            % hidden unit inference
            neghidprob = sigmoid(w_rbm.vishid'*negdata + hbiasmat);
            if usejacket,
                neghidstates = gsingle(rand(size(neghidprob)) < neghidprob);
            else
                neghidstates = single(rand(size(neghidprob)) < neghidprob);
            end
        end
        
        %%% monitor reconstruction error
        recon = sigmoid(w_rbm.vishid*poshidprob + vbiasmat);
        recon_err = norm(xb - recon, 'fro')/params.batchsize;
        
        recon_epoch(b) = recon_err;
        sparsity_epoch(b) = double(mean(poshidprob(:)));
        
        
        %%% compute gradient
        dvh = (xb*poshidprob' - negdata*neghidprob')/params.batchsize;
        dhbias = mean(poshidprob,2) - mean(neghidprob,2);
        dvbias = mean(xb,2) - mean(negdata,2);
        
        
        %%% (approximate) sparsity regularization
        if isempty(shared.runningavg_prob)
            shared.runningavg_prob = mean(poshidprob,2);
        else
            shared.runningavg_prob = 0.9*shared.runningavg_prob + 0.1*mean(poshidprob,2);
        end
        dhbias_sparsity = params.plambda*(params.pbias - shared.runningavg_prob);
        
        % update parameters
        shared.vhinc = momentum*shared.vhinc + epsilon*(dvh - params.l2reg*w_rbm.vishid);
        shared.hbiasinc = momentum*shared.hbiasinc + epsilon*(dhbias + dhbias_sparsity);
        shared.vbiasinc = momentum*shared.vbiasinc + epsilon*dvbias;
        
        w_rbm.vishid = w_rbm.vishid + shared.vhinc;
        w_rbm.hidbias = w_rbm.hidbias + shared.hbiasinc;
        w_rbm.visbias = w_rbm.visbias + shared.vbiasinc;
    end
    
    % monitoring variable
    error_history(t) = double(mean(recon_epoch));
    sparsity_history(t) = double(mean(sparsity_epoch));
    
    if ~mod(t,10),
        fprintf('epoch %d:\t error=%g,\t sparsity=%g\n', t, double(error_history(t)), double(sparsity_history(t)));
        save_vars(params.fname_mat, w_rbm, params, error_history, sparsity_history);
    end
end

w_rbm = remove_rbm_fields(w_rbm);
save_vars(params.fname_mat, w_rbm, params, error_history, sparsity_history);

clear shared;

return;


function save_vars(fname_mat,w_rbm,params,error_history,sparsity_history)

w_rbm = remove_rbm_fields(w_rbm);
error_history = double(error_history);
sparsity_history = double(sparsity_history);
save(fname_mat, 'w_rbm', 'params', 'error_history', 'sparsity_history');

return;


function w_rbm = remove_rbm_fields(w_rbm)

% remove temporary parameters if exist
if isfield(w_rbm,'vbiasmat')
    w_rbm = rmfield(w_rbm,'vbiasmat');
end
if isfield(w_rbm,'hbiasmat'),
    w_rbm = rmfield(w_rbm,'hbiasmat');
end

% convert to cpu variables
w_rbm.vishid = double(w_rbm.vishid);
w_rbm.visbias = double(w_rbm.visbias);
w_rbm.hidbias = double(w_rbm.hidbias);

return;

