using GPTinf
using PyPlot
using Distributions
###generate synthetic data

sd = 0.8; mu1 = -6; mu2 = 0; mu3 = 2;
Xtrain = [sd*randn(10,1) + mu1, sd*randn(10,1) + mu2, sd*randn(10,1) + mu3];
ytrain = [ones(10,1), zeros(10,1), ones(10,1)];
Xtest = reshape(linspace(-8,4,100),100,1);


D = size(Xtrain,2)
r = 2; n = 10; Q = r^D; length_scale = 1.07; seed = 17;

function yhat(w_store::Array,U_store::Array,I::Array,phitest::Array)
Ntest=size(phitest,1);
    T=size(w_store,2);

    meanfhat= @parallel (+) for i=1:T
        cdf(Normal(),pred(w_store[:,i],U_store[:,:,:,i],I,phitest));
    end
    meanfhat=meanfhat/T;
    return meanfhat;
end
function RMSEvec(w_store::Array,U_store::Array,I::Array,phitest::Array,ytest::Array)
    Ntest=length(ytest);
    T=size(w_store,2);
    yfit = Array(Float64,Ntest,T); RMSE = Array(Float64,1,T);
    for i = 1:T
      yfit[:,i] = yhat(w_store[:,i],U_store[:,:,:,i],I,phitrain)
      RMSE[1,i] = norm(ytest-yfit[:,i])/sqrt(Ntest);
    end
    meanfhat=mean(yfit,2)
return RMSE
end


burnin=5;  ## this is for traceplots purpose
m=30; epsw = 0.01; epsU = 0.01; maxepoch = 100 
seed = 123;scale=sqrt(n/(Q^(1/D)));


phitrain=feature(Xtrain,n,length_scale,seed,scale);
phitest=feature(Xtest,n,length_scale,seed,scale);
I=samplenz(r,D,Q,seed);
w_store,U_store=GPT_SGLDERM_probit(phitrain,ytrain,I,r,Q,m, epsw, epsU, burnin, maxepoch);
yfit = yhat(w_store,U_store,I,phitrain);
yfit_test = yhat(w_store,U_store,I,phitest);
RMSEtrain = norm(ytrain-yfit)/sqrt(size(ytrain,1));
plot(Xtest,yfit_test,label="fitted probability");scatter(Xtrain,ytrain,label="synthetic data");
legend(loc = "lower left")
title("1-D classfication")
RMSEtrainvec = RMSEvec(w_store,U_store,I,phitrain,ytrain);
figure()
plot(RMSEtrainvec[:]);title("RMSE vs epoch")




