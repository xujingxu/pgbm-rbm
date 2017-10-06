function y_mult = multi_output(y,numlabel)
%%%
%   y       : 1         x batchsize
%   y_mult  : numlabel  x batchsize
n = length(y);
y_mult = sparse(1:n,y,1,n,numlabel,n);
y_mult = full(y_mult');
return;
