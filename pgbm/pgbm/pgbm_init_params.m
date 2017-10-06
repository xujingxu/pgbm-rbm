function [w_pgbm, params] = pgbm_init_params(params, w_rbm, xtrain, ytrain)
%

if ~exist('w_rbm','var') || isempty(w_rbm),
    %%% random initialization
    % first component
    w_pgbm.vishid1 = 0.02*randn(params.numvis, params.numhid1);
    w_pgbm.vis1bias = zeros(params.numvis,1);
    w_pgbm.hid1bias = zeros(params.numhid1,1);
    
    % second component
    w_pgbm.vishid2 = 0.02*randn(params.numvis, params.numhid2);
    w_pgbm.vis2bias = zeros(params.numvis,1);
    w_pgbm.hid2bias = zeros(params.numhid2,1);
else
    %%% initialize with feature selection
    % feature selection
    addpath ttest/
    ztrain = sigmoid(bsxfun(@plus,w_rbm.vishid'*xtrain,w_rbm.hidbias));
    out = fsTtest_multiclass_wrapper(ztrain', ytrain);
    [~,sidx] = sort(out.meanscores, 'descend');
    clear ztrain xtrain ytrain;
    
    % first component
    w_pgbm.vishid1 = w_rbm.vishid(:,sidx(1:params.numhid1));
    w_pgbm.vis1bias = w_rbm.visbias;
    w_pgbm.hid1bias = w_rbm.hidbias(sidx(1:params.numhid1));
    
    % second component
    w_pgbm.vishid2 = w_rbm.vishid(:,sidx(end-params.numhid2+1:end));
    w_pgbm.vis2bias = zeros(params.numvis,1);
    w_pgbm.hid2bias = w_rbm.hidbias(sidx(end-params.numhid2+1:end));
end

% convert to single precision
w_pgbm.vishid1 = single(w_pgbm.vishid1);
w_pgbm.vis1bias = single(w_pgbm.vis1bias);
w_pgbm.hid1bias = single(w_pgbm.hid1bias);
w_pgbm.vishid2 = single(w_pgbm.vishid2);
w_pgbm.vis2bias = single(w_pgbm.vis2bias);
w_pgbm.hid2bias = single(w_pgbm.hid2bias);

datelearn = datestr(now, 30);
params.fname = sprintf('pgbm_%s_vis%d_hid1_%02d_hid2_%02d_eps%g_l2reg%g_pb%g_pl%g_kcd%d_ngibbs%d_usemf%d_iter%d_date_%s', ...
    params.dataset, params.numvis, params.numhid1, params.numhid2, params.epsilon, params.l2reg, params.pbias, params.plambda, params.kcd, params.ngibbs, params.use_meanfield, params.maxiter, datelearn); % TEST version
params.fname_mat = sprintf('%s/%s.mat',params.savepath, params.fname);

return;