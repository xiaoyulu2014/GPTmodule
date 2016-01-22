using DataFrames

@everywhere Xtrain = readdlm("/homes/xlu/Downloads/kin40k_train_data.txt", Float64);
@everywhere ytrain = readdlm("/homes/xlu/Downloads/kin40k_train_labels.txt", Float64);
@everywhere Xtest = readdlm("/homes/xlu/Downloads/kin40k_test_data.txt", Float64);
@everywhere ytest = readdlm("/homes/xlu/Downloads/kin40k_test_labels.txt", Float64);

@everywhere using GPTinf
@everywhere Ntrain,D=size(Xtrain);
@everywhere seed=17;
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
@everywhere numiter=5;
@everywhere Q=200;   #200
@everywhere n=100;
@everywhere scale=1;

#find hyperparameters
#GPNT_hyperparameters(Xtrain,ytrain,n,0.5,0.5,0.5,seed)
@everywhere length_scale=2.57;
@everywhere sigma_RBF=3.11;
@everywhere sigma=0.65;
@everywhere phitrain=feature(Xtrain,n,length_scale,sigma_RBF,seed,scale);
@everywhere phitest=feature(Xtest,n,length_scale,sigma_RBF,seed,scale);


#RMSE as a function of n
@everywhere r = 5; @everywhere nvec = convert(Array{Int,1},round(linspace(5,50,5)));
@everywhere function func(n::Real)
	tic();
	I=samplenz(r,D,Q,seed);
	phitrain=feature(Xtrain,n,length_scale,sigma_RBF,seed,scale);
	phitest=feature(Xtest,n,length_scale,sigma_RBF,seed,scale);
	w_store,U_store=GPTgibbs(phitrain,ytrain,sigma,I,r,Q,burnin,numiter)
	RMSEtest = ytrainStd*RMSE(w_store,U_store,I,phitest,ytest);
	return(RMSEtest,toq())
end
res = pmap(func,nvec)

#RMSE_n = [res[i][1] for i=1:5]; timer_n =  [res[i][2] for i=1:5]

#=cd("plot")
outfile=open("kin_RMSEn","a") #append to file
	println(outfile,"RMSE_n=",RMSE_n,"timer_n=",timer_n,"Q=",Q,"r=",r,"nvec=",nvec);
close(outfile)
=#
#GP exact using learnt hyperparameters 
#=using GPexact
f =  GPexact.SECov(length_scale,sigma_RBF);
gp = GPexact.GP(0,f,size(Xtrain,2));

tic();yfittrain = GPpost(gp,Xtrain,ytrain,Xtrain,sigma);timer_train = toc();
RMSEtrain = ytrainStd* (norm(ytrain-yfittrain)/sqrt(Ntrain));
tic();yfittest = GPpost(gp,Xtrain,ytrain,Xtest,sigma);timer_test = toc();
RMSEtest = ytrainStd* (norm(ytest-yfittest)/sqrt(N - Ntrain));
=#





#=

### plot for whether learning U is helpful
rvec = convert(Array{Int,1},round(linspace(5,50,5)));
@everywhere function func_w(r::Real)
	tic();
	I=samplenz(r,D,Q,seed);
	w_store,U_store=GPT_w(phitrain,ytrain,sigma,I,r,Q);
	RMSEtest = ytrainStd*RMSE(w_store,U_store,I,phitest,ytest);
	return(RMSEtest, toq())
end
RMSE_w, timer_w = pmap(func_w,rvec)

@everywhere function func(r::Real)
	tic();
	I=samplenz(r,D,Q,seed);
	w_store,U_store=GPTgibbs(phitrain,ytrain,sigma,I,r,Q,burnin,numiter)
	RMSEtest = ytrainStd*RMSE(w_store,U_store,I,phitest,ytest);
	return(RMSEtest,toq())
end
RMSE_wU, timer_wU = pmap(func,rvec)

=#



