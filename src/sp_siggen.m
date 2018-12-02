function x = sp_siggen(N,K,flag)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
x = zeros(N,1);

xval = randn(K,1);

ind = randperm(N);

sup = ind(1:K);

if (flag == 1)
    x(sup) = xval;
else
    xval = sign(xval);
    x(sup) = xval;
end

end

