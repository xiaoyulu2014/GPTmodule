@everywhere using GPTinf
@everywhere using DataFrames
@everywhere using Iterators
using PyPlot

@everywhere data=DataFrames.readtable("Folds5x2_pp.csv", header = true);
@everywhere data = convert(Array,data);
@everywhere N=size(data,1);
@everywhere D=4;
@everywhere Ntrain=500;
@everywhere Ntest=500;
@everywhere seed=17;
#@everywhere length_scale=2.435;
#@everywhere sigma=0.253;
#@everywhere signal_var = sigma^2;
#@everywhere sigma_RBF=0.767;
@everywhere Xtrain = data[1:Ntrain,1:D];
@everywhere ytrain = data[1:Ntrain,D+1];
@everywhere XtrainMean=mean(Xtrain,1); 
@everywhere XtrainStd=zeros(1,D);
@everywhere for i=1:D
	    XtrainStd[1,i]=std(Xtrain[:,i]);
	    end
@everywhere ytrainMean=mean(ytrain);
@everywhere ytrainStd=std(ytrain);
@everywhere Xtrain = datawhitening(Xtrain);
@everywhere ytrain=datawhitening(ytrain);
@everywhere Xtest = (data[Ntrain+1:Ntrain+Ntest,1:D]-repmat(XtrainMean,Ntest,1))./repmat(XtrainStd,Ntest,1);
@everywhere ytest = (data[Ntrain+1:Ntrain+Ntest,D+1]-ytrainMean)/ytrainStd;
@everywhere burnin=10;
@everywhere numiter=30;
@everywhere Q=100;   #200
@everywhere n=150;
#@everywhere scale=1;
@everywhere r = 30;

#GPNT_hyperparameters(Xtrain,ytrain,n,1.0,1.0,0.2,seed)



###SGLD on hyperparameters

@everywhere scale = sqrt(n/(Q^(1/D)));
@everywhere I=samplenz(r,D,Q,seed); 
@everywhere m = 500;
@everywhere maxepoch = 10;
##tuning epsw, epsU, epsl, epsSrbf, epsSignalVar
@everywhere t=Iterators.product(3:5,6:8)
@everywhere myt=Array(Any,9);
@everywhere it=1;
@everywhere for prod in t
	myt[it]=prod;
        it+=1;
        end

@everywhere hyp_init = [1,1,0.5]
@everywhere length_scale, signal_var,sigma_RBF = hyp_init;
@everywhere phitrain0=feature(Xtrain,n,length_scale,sigma_RBF,seed,scale);
@everywhere phitest0=feature(Xtest,n,length_scale,sigma_RBF,seed,scale);

testRMSE_adam = SharedArray(Float64, maxepoch,25);testRMSE = SharedArray(Float64, maxepoch,25)
@sync @parallel for iter = 1:9
    i,j=myt[iter];
    epsw=float(string("1e-",i)); epsU=float(string("1e-",j));
    w_store_hyper,U_store_hyper,l_store, SigmaRBF_store, SignalVar_store=GPTinf.GPT_SGLDERM_adam(Xtrain, ytrain, I, n,r, Q, m, epsw,epsU,burnin,maxepoch,seed,hyp_init);
    w_store,U_store=GPT_SGLDERM(phitrain0, ytrain,signal_var, I, r, Q, m, epsw, epsU, burnin, maxepoch);
   # testRMSE=Array(Float64,maxepoch)
    numbatches=int(ceil(Ntrain/m))
    for epoch=1:maxepoch
	phitest = feature(Xtest,n,l_store[epoch],SigmaRBF_store[epoch],seed,scale)
        testpred_adam=pred(w_store_hyper[:,epoch],U_store_hyper[:,:,:,epoch],I,phitest)
        testRMSE_adam[epoch,iter]=ytrainStd*norm(ytest-testpred_adam)/sqrt(Ntest)
 	testpred=pred(w_store[:,epoch],U_store[:,:,:,epoch],I,phitest0)
        testRMSE[epoch,iter]=ytrainStd*norm(ytest-testpred)/sqrt(Ntest)
    end
println("r=",r,";minRMSE=",minimum(testRMSE[:,iter]),";minepoch=",indmin(testRMSE[:,iter]),";minRMSE_adam=",minimum(testRMSE_adam[:,iter]),";epsw=",epsw,";epsU=",epsU);
end


##SGLD
#@everywhere length_scale= 2.435;
#@everywhere sigma=0.253;
#@everywhere signal_var = sigma^2;
#@everywhere sigma_RBF= 0.767;

#=
@everywhere length_scale,sigma_RBF,signal_var = exp(randn(3))
@everywhere scale = sqrt(n/(Q^(1/D)));
@everywhere phitrain=feature(Xtrain,n,length_scale,sigma_RBF,seed,scale);
@everywhere phitest=feature(Xtest,n,length_scale,sigma_RBF,seed,scale);
@everywhere I=samplenz(r,D,Q,seed); 
@everywhere m = 50;
@everywhere maxepoch = 100;


@everywhere t=Iterators.product(3:7,4:8)
@everywhere myt=Array(Any,25);
@everywhere it=1;
@everywhere for prod in t
	myt[it]=prod;
        it+=1;
        end

@sync @parallel for Tuple in myt
    i,j=Tuple;
    epsw=float(string("1e-",i)); epsU=float(string("1e-",j));
    w_store,U_store=GPT_SGLDERM(phitrain, ytrain,signal_var, I, r, Q, m, epsw, epsU, burnin, maxepoch);
    testRMSE=Array(Float64,maxepoch)
    numbatches=int(ceil(Ntrain/m))
    for epoch=1:maxepoch
        testpred=pred(w_store[:,epoch],U_store[:,:,:,epoch],I,phitest)
        testRMSE[epoch]=ytrainStd*norm(ytest-testpred)/sqrt(Ntest)
    end
    
println("r=",r,";minRMSE=",minimum(testRMSE),";minepoch=",indmin(testRMSE),";epsw=",epsw,";epsU=",epsU,";burnin=",burnin,";maxepoch=",maxepoch);
end

=#



#RMSESGLD = ytrainStd*RMSE(w_storeSGLD[:,end-burnin:end],U_storeSGLD[:,:,:,end-burnin:end],I,phitest,ytest);
#RMSEvec_SGLD = ytrainStd*RMSESGLDvec(w_storeSGLD,U_storeSGLD,I,phitest,ytest);
#=
@everywhere function yhat_func(w_store::Array,U_store::Array,I::Array,X::Array,l_store::Array,SigmaRBF_store::Array)
    Ntest=size(X,1);
    T=size(w_store,2);
    n=size(U_store,1)
    meanfhat= @parallel (+) for i=1:T
	phi = feature(X,n,l_store[i],SigmaRBF_store[i],seed,scale)
        pred(w_store[:,i],U_store[:,:,:,i],I,phi);
    end
    meanfhat=meanfhat/T;
    return meanfhat;
end


@everywhere function RMSE_func(w_store::Array,U_store::Array,I::Array,Xtest::Array,ytest::Array,l_store::Array,SigmaRBF_store::Array)
    Ntest=length(ytest);
    T=size(w_store,2);
    yfit = Array(Float64,Ntest,T); RMSEvec = Array(Float64,1,T);
    for i = 1:T
      phitest = feature(Xtest,n,l_store[i],SigmaRBF_store[i],seed,scale)
      yfit[:,i] = pred(w_store[:,i],U_store[:,:,:,i],I,phitest);
      RMSEvec[1,i] = norm(ytest-yfit[:,i])/sqrt(Ntest);
     end
return RMSEvec
end
=#



### plot for whether learning U is helpful
#=rvec = convert(Array{Int,1},round(linspace(2,20,5)));

@everywhere function func_w(r::Real)
	I=samplenz(r,D,Q,seed);
	w_store,U_store=GPT_w(phitrain,ytrain,sigma,I,r,Q);
	RMSEtest = ytrainStd*RMSE(w_store,U_store,I,phitest,ytest);
	return(RMSEtest)
end
RMSE_w = pmap(func_w,rvec)

@everywhere function func(r::Real)
	I=samplenz(r,D,Q,seed);
	w_store,U_store=GPTgibbs(phitrain,ytrain,sigma,I,r,Q,burnin,numiter)
	RMSEtest = ytrainStd*RMSE(w_store,U_store,I,phitest,ytest);
	return(RMSEtest)
end

RMSE_wU = pmap(func,rvec)
=#






#=

data=DataFrames.readtable("Folds5x2_pp.csv", header = true);
data = convert(Array,data);
data = data[1:1000,:];
N=size(data,1);
D=4;
Ntrain=500;
#@everywhere length_scale=1.4332;
#@everywhere sigma=0.2299;
Xtrain = data[1:Ntrain,1:D];
ytrain = data[1:Ntrain,D+1];
XtrainMean=mean(Xtrain,1);
XtrainStd=zeros(1,D);
for i=1:D
    XtrainStd[1,i]=std(Xtrain[:,i]);
end
ytrainMean=mean(ytrain);
ytrainStd=std(ytrain);
Xtrain = datawhitening(Xtrain);
ytrain = datawhitening(ytrain);
Xtest = (data[Ntrain+1:end,1:D]-repmat(XtrainMean,N-Ntrain,1))./repmat(XtrainStd,N-Ntrain,1);
ytest = (data[Ntrain+1:end,D+1]-ytrainMean)/ytrainStd;
burnin=250;
numiter=50;
r = 8;
n = 150;
seed = 17;
GPNT_hyperparameters(Xtrain,ytrain,n,1.0,1.0,0.2,seed)




#### RMSE on test set #######

restrain_Q=SharedArray(Float64,5,5);timertrain_Q=SharedArray(Float64,5,5);timer_Q = SharedArray(Float64,5,5)
restest_Q=SharedArray(Float64,5,5);timertest_Q=SharedArray(Float64,5,5);
Qvec = round(linspace(100,5000,5)); seed = 17;
phitrain=feature(Xtrain,n,length_scale,seed,1);
phitest=feature(Xtest,n,length_scale,seed),1;
@parallel for i in 1:5
    Q = convert(Int,Qvec[i])
    @parallel for j in 1:5
	tic();
   	seed = j;
    I=samplenz(r,D,Q,seed);
   	w_store,U_store=GPTgibbs(phitrain,ytrain,sigma,I,r,Q,burnin,numiter);
   	timer_Q[i,j] = toq();
  	tic();
  	restrain_Q[i,j] = ytrainStd*RMSE(w_store,U_store,I,phitrain,ytrain);
  	timertrain_Q[i,j] = toq() ;
        tic();
        restest_Q[i,j] = ytrainStd*RMSE(w_store,U_store,I,phitest,ytest);
   	timertest_Q[i,j] = toq() ;
    println(j)
    end
    println(Q)
end


outfile=open("RMSEgibbsQ","a") #append to file
    println(outfile,"restrain_Q=",restrain_Q,";timertrain_Q=",timertrain_Q,";timer_Q=",timer_Q,
            ";restest_Q=",restest_Q,";timertest_Q=",timertest_Q);
    close(outfile)


@everywhere r = 8; @everywhere Q = 100
restrain_n=SharedArray(Float64,5,5);timertrain_n=SharedArray(Float64,5,5);timer_n = SharedArray(Float64,5,5)
restest_n=SharedArray(Float64,5,5);timertest_n=SharedArray(Float64,5,5);
nvec = round(linspace(r,120,5))
@parallel for i in 1:5
    n = convert(Int,nvec[i])
    @parallel for j in 1:5
    	tic();
	seed = j;
	I=samplenz(r,D,Q,seed);
 	phitrain=feature(Xtrain,n,length_scale,seed,1);
 	phitest=feature(Xtest,n,length_scale,seed,1);
        w_store,U_store=GPTgibbs(phitrain,ytrain,sigma,I,r,Q,burnin,numiter);
        timer_n[i,j] = toq();
	tic();
	restrain_n[i,j] = ytrainStd*RMSE(w_store,U_store,I,phitrain,ytrain);
	timertrain_n[i,j] = toq();
	tic();
	restest_n[i,j] = ytrainStd*RMSE(w_store,U_store,I,phitest,ytest);
	timertest_n[i,j] = toq();
        println(j)
    end
    println(n)
end


outfile=open("RMSEgibbs_n","a") #append to file
    println(outfile,"restrain_n=",restrain_n,";timertrain_n=",timertrain_n,";timer_n=",timer_n,
            ";restest_n=",restest_n,";timertest_n=",timertest_n);
    close(outfile)




@everywhere n = 50; @everywhere Q = 100; @everywhere seed = 17;
@everywhere phitrain=feature(Xtrain,n,length_scale,seed,1);
@everywhere phitest=feature(Xtest,n,length_scale,seed,1);
restrain_r=SharedArray(Float64,5,5);timertrain_r=SharedArray(Float64,5,5);;timer_r = SharedArray(Float64,5,5)
restest_r=SharedArray(Float64,5,5);timertest_r=SharedArray(Float64,5,5);
rvec = round(linspace(5,50,5));
@parallel for i in 1:5
    r = convert(Int,rvec[i])
    @parallel for j in 1:5
   	 tic();
	   seed = j;
  	 I=samplenz(r,D,Q,seed);
  	 w_store,U_store=GPTgibbs(phitrain,ytrain,sigma,I,r,Q,burnin,numiter);
  	 timer_r[i,j] = toq();
  	 tic();
  	 restrain_r[i,j] = ytrainStd*RMSE(w_store,U_store,I,phitrain,ytrain);
  	 timertrain_r[i,j] = toq();
  	 tic();
  	 restest_r[i,j] = ytrainStd*RMSE(w_store,U_store,I,phitest,ytest);
  	 timertest_r[i,j] = toq();
         println(j);
    end
    println(r)
end


outfile=open("RMSEgibbsr","a") #append to file
    println(outfile,"restrain_r=",restrain_r,";timertrain_r=",timertrain_r,";timer_r=",timer_r,
            ";restest_r=",restest_r,";timertest_r=",timertest_r,
            ";n=",n,";Q=",Q,";rvec=",rvec);
    close(outfile)
=#

