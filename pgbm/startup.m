addpath utils/;
addpath ttest/;
addpath rbm/;
addpath pgbm/;
% addpath data/;
datapath = 'data';
if ~exist(datapath,'dir'),
    mkdir(datapath);
end
addpath(datapath);

savepath = 'results';
if ~exist(savepath,'dir'),
    mkdir(savepath);
end

logpath = 'log';
if ~exist(logpath,'dir'),
    mkdir(logpath);
end

% add path liblinear
