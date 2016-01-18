using DataFrames

@everywhere Xtrain = readdlm("/homes/xlu/Downloads/kin40k_train_data.txt", Float64);
@everywhere ytrain = readdlm("/homes/xlu/Downloads/kin40k_train_labels.txt", Float64);
@everywhere Xtest = readdlm("/homes/xlu/Downloads/kin40k_test_data.txt", Float64);
@everywhere ytest = readdlm("/homes/xlu/Downloads/kin40k_test_labels.txt", Float64);

@everywhere using GPTinf
@everywhere Ntrain,D=size(Xtrain);
@everywhere seed=17;
@everywhere length_scale=3.1772;
@everywhere sigma=6.35346;
@everywhere sigma_RBF=0.686602;
@everywhere XtrainMean=mean(Xtrain,1); 
@everywhere XtrainStd=zeros(1,D);
@everywhere for i=1:D
	    XtrainStd[1,i]=std(Xtrain[:,i]);
	    end
@everywhere ytrainMean=mean(ytrain);
@everywhere ytrainStd=std(ytrain);
@everywhere Xtrain = datawhitening(Xtrain);
@everywhere ytrain=datawhitening(ytrain);
@everywhere Ntest = size(Xtest,1);
@everywhere Xtest = (Xtest-repmat(XtrainMean,Ntest,1))./repmat(XtrainStd,Ntest,1);
@everywhere ytest = (ytest-ytrainMean)/ytrainStd;
@everywhere burnin=10;
@everywhere numiter=50;
@everywhere Q=16;   #200
@everywhere n=150;
@everywhere scale=1;
@everywhere phitrain=feature(Xtrain,n,length_scale,sigma_RBF,seed,scale);
@everywhere phitest=feature(Xtest,n,length_scale,sigma_RBF,seed,scale);

#find hyperparameters
#GPNT_hyperparameters(Xtrain,ytrain,n,1.0,1.0,0.2,seed)
### plot for whether learning U is helpful
rvec = convert(Array{Int,1},round(linspace(2,20,5)));

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





