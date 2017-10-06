function [hidprob, swprobs] = rbm_inference(vis,w_rbm,params,usejacket)

if ~exist('usejacket','var'),
    usejacket = 0;
end
if ~exist('use_meanfield','var'),
    use_meanfield = 1;
end
batchsize = size(vis,2);
if ~isfield(w_rbm,'vbiasmat'),
     w_rbm.vbiasmat = repmat(w_rbm.visbias,1,batchsize);
end
% if ~isfield(w_pgbm,'vbias2mat'),
%     w_pgbm.vbias2mat = repmat(w_pgbm.vis2bias,1,batchsize);
% end
if ~isfield(w_rbm,'hbiasmat'),
     w_rbm.hbiasmat = repmat(w_rbm.hidbias,1,batchsize);
end
% if ~isfield(w_pgbm,'hbias2mat'),
%     w_pgbm.hbias2mat = repmat(w_pgbm.hid2bias,1,batchsize);
% end
% w_rbm.vbiasmat = repmat(w_rbm.visbias,1,batchsize);
% w_rbm.hbiasmat = repmat(w_rbm.hidbias,1,batchsize);

% intialize switch units uniformly
if usejacket,
    swstates = gones(params.numvis,batchsize);
else
    swstates = single(ones(params.numvis,batchsize));
end

for gibbs = 1:25,
    %%% infer h give v and z
    % foreground
    hidprob = w_rbm.hbiasmat;
    hidprob = hidprob + w_rbm.vishid'*(vis.*swstates);
    hidprob = 1./(1+exp(-hidprob));
%     % background
%     hid2prob = w_pgbm.hbias2mat;
%     hid2prob = hid2prob + w_pgbm.vishid2'*(vis.*(1-swstates));
%     hid2prob = 1./(1+exp(-hid2prob));
    
    % mean-field or sampling on h
    if use_meanfield,
        hidstate = hidprob;
       
    else
        if usejacket,
            hidstate = gsingle(rand(size(hidprob)) < hidprob);
            
        else
            hidstate = single(rand(size(hidprob)) < hidprob);
           
        end
    end
    
    % infer z
    swprobs = (w_rbm.vishid*hidstate + w_rbm.vbiasmat);
    swprobs = swprobs - (w_rbm.vishid*hidstate + w_rbm.vbiasmat);
    swprobs = swprobs.*vis;
    swprobs = 1./(1+exp(-swprobs));
    
    % mean-field or sampling on z
    if use_meanfield,
        swstates = swprobs;
    else
        if usejacket,
            swstates = gsingle(rand(size(swprobs)) < swprobs);
        else
            swstates = single(rand(size(swprobs)) < swprobs);
        end
    end
end

return;
