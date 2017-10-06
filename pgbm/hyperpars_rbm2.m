%   hyperparameters for rbm pretraining
% 

switch dataset,
    case 'mnist_bgimg',
        numhid = 500;
        epsilon = 0.05;
        l2reg = 0.0002;
        pbias = 0.05;
        plambda = 0.1;
        kcd = 1;
        maxiter = 800;
        batchsize = 20;
    case 'mnist_bgrand',
        numhid = 500;
        epsilon = 0.02;
        l2reg = 0.0001;
        pbias = 0.05;
        plambda = 0.3;
        kcd = 1;
        maxiter = 800;
        batchsize = 20;
end
