function [out] = fsTtest(X,Y)
    [~,n] = size(X);
    W = zeros(n,1);
    
    for i=1:n
        X1 = X(Y == 1,i);
        X2 = X(Y == 2,i);

        n1 = size(X1,1);
        n2 = size(X2,1);

        mean_X1 = sum(X1)/n1;
        mean_X2 = sum(X2)/n2 ;   

        var_X1 = sum((X1 - mean_X1).^2)/n1;
        var_X2 = sum((X2 - mean_X2).^2)/n2;
        
        W(i) = (mean_X1 - mean_X2)/sqrt(var_X1/n1 + var_X2/n2);

    end
    
    [foo out.fList] = sort(W, 'descend'); 
    out.W = W;   
    out.prf = -1;
end