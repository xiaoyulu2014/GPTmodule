using GPTinf
using PyPlot
###generate synthetic data
N = 500;
#Xtrain = [linspace(-10,10,N) linspace(-10,10,N)];
Xtrain = randn(N,3)
D = size(Xtrain,2)
r = 10; n = r; Q = r^D; sigma = 0.1; length_scale = 5; seed = 17;
ytrain = data_simulator(Xtrain,n,r,Q,sigma,length_scale,seed);

XtrainMean=mean(Xtrain,1);
XtrainStd=zeros(1,D);
for i=1:D
  XtrainStd[1,i]=std(Xtrain[:,i]);
end
ytrainMean=mean(ytrain);
ytrainStd=std(ytrain);
Xtrain = datawhitening(Xtrain);
ytrain = datawhitening(ytrain);


###make inference to learn the data

function yhat(w_store::Array,U_store::Array,I::Array,phitest::Array)
Ntest=size(phitest,1);
    T=size(w_store,2);

    meanfhat= @parallel (+) for i=1:T
        pred(w_store[:,i],U_store[:,:,:,i],I,phitest);
    end
    meanfhat=meanfhat/T;
    return meanfhat;
end
burnin=0;  ## this is for traceplots purpose
numiter=5; burnin1 = 1 ## omit first 30 iterations when prediting
seed = 123;

phitrain=feature(Xtrain,n,length_scale,seed,1);
I=samplenz(r,D,Q,seed);
w_store,U_store=GPTgibbs(phitrain,ytrain,sigma,I,r,Q,burnin,numiter);
RMSEgibbs = ytrainStd*RMSE(w_store[:,end-burnin1:end],U_store[:,:,:,end-burnin1:end],I,phitrain,ytrain);
yfitgibbs = yhat(w_store,U_store,I,phitrain);




#=
# RMSE  with varying Q and n, fixed r
r = 10;
Q = 100;
# RMSE  with varying Q and n, fixed r
n_vec = [5,20,50,100];RMSEtrain = Array(Float64,length(n_vec))
subplot(121);
for i in 1:length(n_vec)
  n = convert(Int, n_vec[i])
  phitrain=feature(Xtrain,n,length_scale,seed,1);
  I=samplenz(r,D,Q,seed);
  w_store,U_store=GPTgibbs(phitrain,ytrain,sigma,I,r,Q,burnin,numiter);
  RMSEtrain[i] = ytrainStd*RMSE(w_store[:,end-burnin1:end],U_store[:,:,:,end-burnin1:end],I,phitrain,ytrain);
  RMSEvec = ytrainStd*RMSESGLDvec(w_store,U_store,I,phitrain,ytrain);
  println("n = ", n, " RMSEtrain = ", round(RMSEtrain[i],3));
  yfit = yhat(w_store,U_store,I::Array,phitrain::Array);
  plot(RMSEvec[:],label = "n = $n");xlabel("number of iterations");ylabel("RMSE")
  title("r = $r, Q = $Q, varying n")
end
legend(loc="upper right",fancybox="false") # Create a legend of all the existing plots using their labels as names
=#

#=
r = 5;
Q = 100;
# RMSE  with varying Q and n, fixed r
n_vec = [5,20,50,100];RMSEtrain = Array(Float64,length(n_vec))
figure()
subplot(122);
for i in 1:length(n_vec)
  n = convert(Int, n_vec[i])
  phitrain=feature(Xtrain,n,length_scale,seed,1);
  I=samplenz(r,D,Q,seed);
  w_store,U_store=GPTgibbs(phitrain,ytrain,sigma,I,r,Q,burnin,numiter);
  RMSEtrain[i] = ytrainStd*RMSE(w_store[:,end-burnin1:end],U_store[:,:,:,end-burnin1:end],I,phitrain,ytrain);
  RMSEvec = ytrainStd*RMSESGLDvec(w_store,U_store,I,phitrain,ytrain);
  println("n = ", n, " RMSEtrain = ", round(RMSEtrain[i],3));
  yfit = yhat(w_store,U_store,I::Array,phitrain::Array);
  plot(RMSEvec[:],label = "n = $n");xlabel("number of iterations");ylabel("RMSE")
  title("r = $r, Q = $Q, varying n")
end
legend(loc="upper right",fancybox="false") # Create a legend of all the existing plots using their labels as names
=#


#are we able to recover the data with n = r =10?
#=n = 10; r = 10
phitrain=feature(Xtrain,n,length_scale,seed,1);
Q_vec = [r,15,500];RMSEtrain = Array(Float64,3)
subplot(122);
for i in 1:3
  Q = convert(Int, Q_vec[i])
  I=samplenz(r,D,Q,seed);
  w_store,U_store=GPTgibbs(phitrain,ytrain,sigma,I,r,Q,burnin,numiter);
  RMSEtrain[i] = ytrainStd*RMSE(w_store[:,end-burnin1:end],U_store[:,:,:,end-burnin1:end],I,phitrain,ytrain);
  RMSEvec = ytrainStd*RMSESGLDvec(w_store,U_store,I,phitrain,ytrain);
  println("Q = ", Q, " RMSEtrain = ", round(RMSEtrain[i],3));
  yfit = yhat(w_store,U_store,I::Array,phitrain::Array);
  plot(RMSEvec[:],label = "Q = $Q");xlabel("number of iterations");ylabel("RMSE")
  title("r = $r, n = $n, varying Q")
end
legend(loc="upper right",fancybox="false") # Create a legend of all the existing plots using their labels as names

suptitle("training RMSE on 4D synthetic data \n generated with r = 10 features")

=#











