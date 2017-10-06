function [w_rbm, params] = rbm_init_params(params)

%%% random initialization
w_rbm.vishid = 0.02*randn(params.numvis, params.numhid);
w_rbm.visbias = zeros(params.numvis,1);
w_rbm.hidbias = zeros(params.numhid,1);

datelearn = datestr(now, 30);
params.fname = sprintf('rbm_%s_vis%d_hid%02d_eps%g_l2reg%g_pb%g_pl%g_kcd%d_iter%d_date_%s', ...
    params.dataset, params.numvis, params.numhid, params.epsilon, params.l2reg, params.pbias, params.plambda, params.kcd, params.maxiter, datelearn); % TEST version
params.fname_mat = sprintf('%s/%s.mat',params.savepath, params.fname);

return;