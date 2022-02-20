clear
clc

x1 = ones(10,1);
idx = (1:10)';
kk = (1:5)';
A_gen = exp(idx*(1./idx')) .* cos(idx*idx');
A_gen = triu(A_gen, 1);
A_gen = A_gen + A_gen';
A = cell(5,1);
for k = 1:5
    A{k} = A_gen * sin(k) ...
        + diag((idx/10+sum(abs(A_gen),2))*abs(sin(k)));
end
% each column is b_k
bb = exp(idx*(1./kk')) .* sin(idx*kk');

fvalv = zeros(5,1);
for ii = 1:5
    fvalv(ii) = x1'*A{ii}*x1 - bb(:,ii)'*x1;
end
fvalv
fval_1 = max(fvalv)

cvx_begin
    variables z x(10)
    for kk = 1:5
        z >= x'*A{kk}*x - bb(:,kk)'*x;
    end
    minimize z
cvx_end
fvalv = zeros(5,1);
for ii = 1:5
    fvalv(ii) = x'*A{ii}*x - bb(:,ii)'*x;
end
fval_opt = max(fvalv)

D = norm(x1-x, 2);
T = 100000;
data = zeros(T,1);
data(1) = fval_1;
xpt = x1;
fvalv = zeros(5,1);
for ii = 1:5
    fvalv(ii) = xpt'*A{ii}*xpt - bb(:,ii)'*xpt;
end
for kk = 2:T
    gamma_t = D / sqrt(kk-1);
    [~, jidx] = max(fvalv);
    grad = 2*A{jidx}*xpt - bb(:,jidx);
    xpt = xpt - gamma_t * grad / norm(grad, 2);
    fvalv = zeros(5,1);
    for ii = 1:5
        fvalv(ii) = xpt'*A{ii}*xpt - bb(:,ii)'*xpt;
    end
    fval_t = max(fvalv);
    if fval_t < data(kk-1)
        data(kk) = fval_t;
    else
        data(kk) = data(kk-1);
    end
end
figure()
loglog(1:T, data - fval_opt)
xlabel("$t$ (iteration step)","Interpreter","latex")
ylabel("$\min_{i=1,\cdots,t} f(x^i) - f(x^*)$","Interpreter","latex")

data = zeros(T,1);
data(1) = fval_1;
xpt = x1;
fvalv = zeros(5,1);
for ii = 1:5
    fvalv(ii) = xpt'*A{ii}*xpt - bb(:,ii)'*xpt;
end
for kk = 2:T
    [~, jidx] = max(fvalv);
    grad = 2*A{jidx}*xpt - bb(:,jidx);
    gamma_t = (max(fvalv) - fval_opt) / norm(grad, 2);
    xpt = xpt - gamma_t * grad / norm(grad, 2);
    fvalv = zeros(5,1);
    for ii = 1:5
        fvalv(ii) = xpt'*A{ii}*xpt - bb(:,ii)'*xpt;
    end
    fval_t = max(fvalv);
    if fval_t < data(kk-1)
        data(kk) = fval_t;
    else
        data(kk) = data(kk-1);
    end
end
figure()
loglog(1:T, data - fval_opt)
xlabel("$t$ (iteration step)","Interpreter","latex")
ylabel("$\min_{i=1,\cdots,t} f(x^i) - f(x^*)$","Interpreter","latex")
