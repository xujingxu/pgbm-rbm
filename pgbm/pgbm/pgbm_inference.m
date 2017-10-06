function [hid1prob, hid2prob, swprobs] = pgbm_inference(vis,w_pgbm,params,usejacket)

if ~exist('usejacket','var'),
    usejacket = 0;
end

batchsize = size(vis,2);
if ~isfield(w_pgbm,'vbias1mat'),
    w_pgbm.vbias1mat = repmat(w_pgbm.vis1bias,1,batchsize);
end
if ~isfield(w_pgbm,'vbias2mat'),
    w_pgbm.vbias2mat = repmat(w_pgbm.vis2bias,1,batchsize);
end
if ~isfield(w_pgbm,'hbias1mat'),
    w_pgbm.hbias1mat = repmat(w_pgbm.hid1bias,1,batchsize);
end
if ~isfield(w_pgbm,'hbias2mat'),
    w_pgbm.hbias2mat = repmat(w_pgbm.hid2bias,1,batchsize);
end


% intialize switch units uniformly
if usejacket,
    swstates = gones(params.numvis,batchsize);
else
    swstates = single(ones(params.numvis,batchsize));
end

for gibbs = 1:params.ngibbs,
    %%% infer h give v and z
    % foreground
    hid1prob = w_pgbm.hbias1mat;
    hid1prob = hid1prob + w_pgbm.vishid1'*(vis.*swstates);
    hid1prob = 1./(1+exp(-hid1prob));
    % background
    hid2prob = w_pgbm.hbias2mat;
    hid2prob = hid2prob + w_pgbm.vishid2'*(vis.*(1-swstates));
    hid2prob = 1./(1+exp(-hid2prob));
    
    % mean-field or sampling on h
    if params.use_meanfield,
        hid1state = hid1prob;
        hid2state = hid2prob;
    else
        if usejacket,
            hid1state = gsingle(rand(size(hid1prob)) < hid1prob);
            hid2state = gsingle(rand(size(hid2prob)) < hid2prob);
        else
            hid1state = single(rand(size(hid1prob)) < hid1prob);
            hid2state = single(rand(size(hid2prob)) < hid2prob);
        end
    end
    
    % infer z
    swprobs = (w_pgbm.vishid1*hid1state + w_pgbm.vbias1mat);
    swprobs = swprobs - (w_pgbm.vishid2*hid2state + w_pgbm.vbias2mat);
    swprobs = swprobs.*vis;
    swprobs = 1./(1+exp(-swprobs));
    
    % mean-field or sampling on z
    if params.use_meanfield,
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
