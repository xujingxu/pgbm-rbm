function featsel = fsTtest_multiclass_wrapper(X, Y)

nfeat = size(X,2);
assert(size(X,1) == length(Y));

c_list = unique(Y(:))';

featsel_mat = zeros(nfeat, length(c_list));
% featsel_weights = zeros(nfeat,1);
for j=1:length(c_list)
    c= c_list(j);

    out = fsTtest(X, 2*(Y==c)+1*(Y~=c));
    % featsel_mat(out.fList, j) = 1;
    featsel_mat(:, j) = out.W;
end

featsel.meanscores = mean(featsel_mat,2);
featsel.maxscores = max(featsel_mat,[],2);
featsel.rawscores = featsel_mat;

% function [out] = fsTtest(X,Y)
%     [~,n] = size(X);
%     W = zeros(n,1);
%     
%     for i=1:n
%         X1 = X(Y == 1,i);
%         X2 = X(Y == 2,i);
% 
%         n1 = size(X1,1);
%         n2 = size(X2,1);
% 
%         mean_X1 = sum(X1)/n1;
%         mean_X2 = sum(X2)/n2 ;   
% 
%         var_X1 = sum((X1 - mean_X1).^2)/n1;
%         var_X2 = sum((X2 - mean_X2).^2)/n2;
%         
%         W(i) = (mean_X1 - mean_X2)/sqrt(var_X1/n1 + var_X2/n2);
% 
%     end
%     
%     [foo out.fList] = sort(W, 'descend'); 
%     out.W = W;   
%     out.prf = -1;
% end
% 
% 
