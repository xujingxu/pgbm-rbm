%   load mnist variations
% 
%   xtrain  : 784 x 10000
%   ytrain  : 1 x 10000
%   xval    : 784 x 2000
%   yval    : 1 x 2000
%   xtest   : 784 x 50000
%   ytest   : 1 x 50000
%   
%   written by Kihyuk Sohn
% 

function [xtrain, ytrain, xval, yval, xtest, ytest] = load_mnist(dataset)

switch dataset,
    case 'mnist_bgrand',
        [xtrain, ytrain] = loadamat('mnist_background_random_train.amat');
        [xtest, ytest] = loadamat('mnist_background_random_test.amat');
    case 'mnist_bgimg',
        [xtrain, ytrain] = loadamat('mnist_background_images_train.amat');
        [xtest, ytest] = loadamat('mnist_background_images_test.amat');
end

% use last 2000 examples of training set for validation
xval = xtrain(:,10001:end);
yval = ytrain(10001:end);

xtrain = xtrain(:,1:10000);
ytrain = ytrain(1:10000);

return;


function [digits, labels] = loadamat(filename)

dd = load(filename);
digits = dd(:, 1:end-1)';
labels = dd(:, end) + 1;

return