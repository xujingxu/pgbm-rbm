function [w_pgbm_best, params, w_pgbm] = pgbm_train(xtrain, params, w_rbm, ytrain, xval, yval, usejacket)

% random seed
rng('default');

params.numvis = size(xtrain,1);

% momentum
initialmomentum  = 0.5;
finalmomentum    = 0.9;

%%% initialization
[w_pgbm, params] = pgbm_init_params(params, w_rbm, xtrain, ytrain);

% convert to jacket (gpu) variable
if usejacket,
    w_pgbm = cpu2jacket(w_pgbm);
    
    shared.vh1inc = gzeros(size(w_pgbm.vishid1));
    shared.hbias1inc = gzeros(size(w_pgbm.hid1bias));
    shared.vbias1inc = gzeros(size(w_pgbm.vis1bias));
    shared.vh2inc = gzeros(size(w_pgbm.vishid2));
    shared.hbias2inc = gzeros(size(w_pgbm.hid2bias));
    shared.vbias2inc = gzeros(size(w_pgbm.vis2bias));
else    
    shared.vh1inc = single(zeros(size(w_pgbm.vishid1)));
    shared.hbias1inc = single(zeros(size(w_pgbm.hid1bias)));
    shared.vbias1inc = single(zeros(size(w_pgbm.vis1bias)));
    shared.vh2inc = single(zeros(size(w_pgbm.vishid2)));
    shared.hbias2inc = single(zeros(size(w_pgbm.hid2bias)));
    shared.vbias2inc = single(zeros(size(w_pgbm.vis2bias)));
end


% monitoring variables
error_history = zeros(params.maxiter,1);
sparsity1_history = zeros(params.maxiter,1);
sparsity2_history = zeros(params.maxiter,1);

% display parameters
disp(params);


%%% train binary-binary point-wise mixture restricted Boltzmann machine
%   with two mixture component,
%   using stochastic gradient descent with mini-batch,
%   using mean-field for posterior inference,
%   using contrastive divergence to approximate gradient.

N = size(xtrain,2);
numiter = min(ceil(N/params.batchsize), 1000);
best_val = 0;

for t = 1:params.maxiter,
    % monitoring variable
    recon_epoch = zeros(numiter, 1);
    sparsity1_epoch = zeros(numiter, 1);
    sparsity2_epoch = zeros(numiter, 1);
    
    % learning rate
    epsilon = params.epsilon/(1+params.epsdecay*t);
    
    % momentum
    if t < 5,
        momentum = initialmomentum;
    else
        momentum = finalmomentum;
    end
    
    % shuffle input data order
    randidx = randperm(N);
    
    tS = tic;
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
        w_pgbm.vbias1mat = repmat(w_pgbm.vis1bias,1,params.batchsize);
        w_pgbm.vbias2mat = repmat(w_pgbm.vis2bias,1,params.batchsize);
        w_pgbm.hbias1mat = repmat(w_pgbm.hid1bias,1,params.batchsize);
        w_pgbm.hbias2mat = repmat(w_pgbm.hid2bias,1,params.batchsize);
        
        % positive phase: sample (h,z) ~ P(h,z|v)
        [poshid1prob, poshid2prob, posswprob] = pgbm_inference(xb,w_pgbm,params,usejacket);
        if usejacket,
            poshid1states = gsingle(rand(size(poshid1prob)) < poshid1prob);
            poshid2states = gsingle(rand(size(poshid2prob)) < poshid2prob);
            posswstates = gsingle(rand(size(posswprob)) < posswprob);
        else
            poshid1states = single(rand(size(poshid1prob)) < poshid1prob);
            poshid2states = single(rand(size(poshid2prob)) < poshid2prob);
            posswstates = single(rand(size(posswprob)) < posswprob);
        end
        
        % negative phase: sample v' ~ P(v|h), sample (h',z') ~ P(h,z|v')
        neghid1states = poshid1states;
        neghid2states = poshid2states;
        negswstates = posswstates;
        for kcd = 1:params.kcd,
            % negative data
            negdata = negswstates.*(w_pgbm.vishid1*neghid1states + w_pgbm.vbias1mat);
            negdata = negdata + (1-negswstates).*(w_pgbm.vishid2*neghid2states + w_pgbm.vbias2mat);
            negdata = sigmoid(negdata);
            
            % inference
            [neghid1prob, neghid2prob, negswprob] = pgbm_inference(negdata,w_pgbm,params,usejacket);
            if usejacket,
                neghid1states = gsingle(rand(size(neghid1prob)) < neghid1prob);
                neghid2states = gsingle(rand(size(neghid2prob)) < neghid2prob);
                negswstates = gsingle(rand(size(negswprob)) < negswprob);
            else
                neghid1states = single(rand(size(neghid1prob)) < neghid1prob);
                neghid2states = single(rand(size(neghid2prob)) < neghid2prob);
                negswstates = single(rand(size(negswprob)) < negswprob);
            end
        end
        
        %%% monitor reconstruction error
        recon = posswprob.*(w_pgbm.vishid1*poshid1prob + w_pgbm.vbias1mat);
        recon = recon + (1-posswprob).*(w_pgbm.vishid2*poshid2prob + w_pgbm.vbias2mat);
        recon = sigmoid(recon);
        recon_err = norm(xb - recon, 'fro')/params.batchsize;
        
        recon_epoch(b) = recon_err;
        sparsity1_epoch(b) = double(mean(poshid1prob(:)));
        sparsity2_epoch(b) = double(mean(poshid2prob(:)));
        
        
        %%% compute gradient
        % first component
        dvh1 = ((posswprob.*xb)*poshid1prob' - (negswprob.*negdata)*neghid1prob')/params.batchsize;
        dhbias1 = mean(poshid1prob,2) - mean(neghid1prob,2);
        dvbias1 = mean(posswprob.*xb,2) - mean(negswprob.*negdata,2);
        % second component
        dvh2 = (((1-posswprob).*xb)*poshid2prob' - ((1-negswprob).*negdata)*neghid2prob')/params.batchsize;
        dhbias2 = mean(poshid2prob,2) - mean(neghid2prob,2);
        dvbias2 = mean((1-posswprob).*xb,2) - mean((1-negswprob).*negdata,2);
        
        
        %%% (approximate) sparsity regularization
        dh1_sparsity = params.plambda*(params.pbias - mean(poshid1prob,2));
        dh2_sparsity = params.plambda*(params.pbias - mean(poshid2prob,2));
        
        
        %%% update parameters
        shared.vh1inc = momentum*shared.vh1inc + epsilon*(dvh1 - params.l2reg*w_pgbm.vishid1);
        shared.hbias1inc = momentum*shared.hbias1inc + epsilon*(dhbias1 + dh1_sparsity);
        shared.vbias1inc = momentum*shared.vbias1inc + epsilon*dvbias1;
        shared.vh2inc = momentum*shared.vh2inc + epsilon*(dvh2 - params.l2reg*w_pgbm.vishid2);
        shared.hbias2inc = momentum*shared.hbias2inc + epsilon*(dhbias2 + dh2_sparsity);
        shared.vbias2inc = momentum*shared.vbias2inc + epsilon*dvbias2;
        
        w_pgbm.vishid1 = w_pgbm.vishid1 + shared.vh1inc;
        w_pgbm.hid1bias = w_pgbm.hid1bias + shared.hbias1inc;
        w_pgbm.vis1bias = w_pgbm.vis1bias + shared.vbias1inc;
        w_pgbm.vishid2 = w_pgbm.vishid2 + shared.vh2inc;
        w_pgbm.hid2bias = w_pgbm.hid2bias + shared.hbias2inc;
        w_pgbm.vis2bias = w_pgbm.vis2bias + shared.vbias2inc;
    end
    
    % monitoring variable
    error_history(t) = double(mean(recon_epoch));
    sparsity1_history(t) = double(mean(sparsity1_epoch));
    sparsity2_history(t) = double(mean(sparsity2_epoch));
    
    if ~mod(t,5),
        tE = toc(tS);
        fprintf('epoch %d:\t error=%g,\t sparsity1=%g,\t sparsity2=%g (time/epoch=%g)\n', t, error_history(t), sparsity1_history(t), sparsity2_history(t), tE/1);
        save_vars(params.fname_mat, w_pgbm, params, error_history, sparsity1_history, sparsity2_history);
    end
    
    % compute validation performance every 20 epoch
    if ~mod(t, 10),
        if ~isempty(xval),
            w_pgbm = remove_pgbm_fields(w_pgbm);
            ztrain = pgbm_inference(xtrain,w_pgbm,params);
            zval = pgbm_inference(xval,w_pgbm,params);
            [~, acc_val, ~, bestC] = liblinear_wrapper([], ztrain, ytrain, zval, yval);
            if acc_val > best_val,
                best_val = acc_val;
                tbest = t;
                Cbest = bestC;
                w_pgbm_best = w_pgbm;
            end
            if usejacket,
                w_pgbm = cpu2jacket(w_pgbm);
            end
        end
    end
end

save(params.fname_mat, 'w_pgbm', 'params', 'error_history','sparsity1_history','sparsity2_history');
if ~exist('w_pgbm_best','var'),
    w_pgbm_best = w_pgbm;
else
    params.Cbest = Cbest;
    params.maxiter = tbest;
    save_vars(sprintf('%s/%s_%04d.mat',params.savepath,params.fname,tbest), w_pgbm_best, params, error_history, sparsity1_history, sparsity2_history);
end

w_pgbm_best = remove_pgbm_fields(w_pgbm_best);
w_pgbm = remove_pgbm_fields(w_pgbm);

clear shared;

return;


function save_vars(fname_mat,w_pgbm,params,error_history,sparsity1_history,sparsity2_history)

w_pgbm = remove_pgbm_fields(w_pgbm);
error_history = double(error_history);
sparsity1_history = double(sparsity1_history);
sparsity2_history = double(sparsity2_history);
save(fname_mat, 'w_pgbm', 'params', 'error_history', 'sparsity1_history', 'sparsity2_history');

return;


function w_pgbm = remove_pgbm_fields(w_pgbm)

% remove temporary parameters if exist
if isfield(w_pgbm,'vbias1mat')
    w_pgbm = rmfield(w_pgbm,'vbias1mat');
end
if isfield(w_pgbm,'vbias2mat'),
    w_pgbm = rmfield(w_pgbm,'vbias2mat');
end
if isfield(w_pgbm,'hbias1mat'),
    w_pgbm = rmfield(w_pgbm,'hbias1mat');
end
if isfield(w_pgbm,'hbias2mat'),
    w_pgbm = rmfield(w_pgbm,'hbias2mat');
end
if isfield(w_pgbm,'sbiasmat'),
    w_pgbm = rmfield(w_pgbm,'sbiasmat');
end

% convert to cpu variables
w_pgbm = jacket2cpu(w_pgbm);

return;


function w_pgbm = cpu2jacket(w_pgbm)

w_pgbm.vishid1 = gsingle(w_pgbm.vishid1);
w_pgbm.vis1bias = gsingle(w_pgbm.vis1bias);
w_pgbm.hid1bias = gsingle(w_pgbm.hid1bias);
w_pgbm.vishid2 = gsingle(w_pgbm.vishid2);
w_pgbm.vis2bias = gsingle(w_pgbm.vis2bias);
w_pgbm.hid2bias = gsingle(w_pgbm.hid2bias);

return


function w_pgbm = jacket2cpu(w_pgbm)

w_pgbm.vishid1 = single(w_pgbm.vishid1);
w_pgbm.vis1bias = single(w_pgbm.vis1bias);
w_pgbm.hid1bias = single(w_pgbm.hid1bias);
w_pgbm.vishid2 = single(w_pgbm.vishid2);
w_pgbm.vis2bias = single(w_pgbm.vis2bias);
w_pgbm.hid2bias = single(w_pgbm.hid2bias);

return