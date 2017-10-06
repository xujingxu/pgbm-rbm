%   hyperparameters for pgbm training
%

switch dataset
    case 'mnist_bgimg',
        numhid1 = 500;
        numhid2 = 500;
        epsilon = 0.003;
        l2reg = 0.0005;
        pbias = 0.15;
        plambda = 0.3;
        kcd = 5;
        ngibbs = 10;
        use_meanfield = 1;
        maxiter = 200;
        batchsize = 200;
    case 'mnist_bgrand',
        numhid1 = 500;
        numhid2 = 500;
        epsilon = 0.005;
        l2reg = 0.001;
        pbias = 0.15;
        plambda = 1;
        kcd = 5;
        ngibbs = 25;
        use_meanfield = 1;
        maxiter = 400;
        batchsize = 200;
end