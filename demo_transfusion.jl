@everywhere using GPTinf
@everywhere using DataFrames
@everywhere using Iterators
@everywhere using Distributions
using PyPlot

@everywhere data=DataFrames.readtable("/homes/xlu/Downloads/TransfusionData", header = true);
@everywhere data = convert(Array{Float64,2},data);
@everywhere N=size(data,1);
@everywhere data = data[randperm(N),:]
@everywhere D=4;
@everywhere Ntrain=500;
@everywhere Ntest=N-Ntrain;
@everywhere seed=17;
@everywhere length_scale= 5*exp(randn(1))[1]; #exp(1.6732);
@everywhere sigma_RBF= 5*exp(randn(1))[1] #exp(sqrt(0.7650));
@everywhere hyp_init = [length_scale,sigma_RBF];
@everywhere Xtrain = data[1:Ntrain,1:D];
@everywhere ytrain = data[1:Ntrain,D+1];
@everywhere XtrainMean=mean(Xtrain,1); 
@everywhere XtrainStd=zeros(1,D);
@everywhere for i=1:D
	    XtrainStd[1,i]=std(Xtrain[:,i]);
	    end
@everywhere Xtrain = datawhitening(Xtrain);
@everywhere Xtest = (data[Ntrain+1:Ntrain+Ntest,1:D]-repmat(XtrainMean,Ntest,1))./repmat(XtrainStd,Ntest,1);
@everywhere ytest = data[Ntrain+1:Ntrain+Ntest,D+1]
@everywhere burnin=0;
@everywhere Q=100;   #200
@everywhere n=80;
@everywhere r = 20;

@everywhere scale = sqrt(n/(Q^(1/D)));
@everywhere phitrain=feature(Xtrain,n,length_scale,sigma_RBF,seed,scale);
@everywhere phitest=feature(Xtest,n,length_scale,sigma_RBF,seed,scale);
@everywhere I=samplenz(r,D,Q,seed); 
@everywhere m = 200;



@everywhere t=Iterators.product([2,4,6,8],[3,5,7,9])
@everywhere myt=Array(Any,16);
@everywhere it=1;
@everywhere for prod in t
	myt[it]=prod;
        it+=1;
        end
 
@everywhere srand(seed)
@everywhere   Zmat=randn(n,D)
@everywhere   bmat=rand(n,D)*2*pi
@everywhere     numbatches=int(ceil(Ntrain/m))


@everywhere function nlp_func(w_store::Array,U_store::Array,phitest::Array,ytest::Array)
	T = size(w_store,2)
	nlp=zeros(T);
	tmp=zeros(size(ytest,1))
	for epoch=1:T
		prob = cdf(Normal(),pred(w_store[:,epoch],U_store[:,:,:,epoch],I,phitest))
		nlp[epoch] = -sum(ytest.*log(prob) + (1-ytest).*log(1-prob))
		tmp += prob
	end
	tmp = tmp/T
	return(nlp,-sum(ytest.*log(tmp) + (1-ytest).*log(1-tmp)))
end

@everywhere function nlp_func_full(theta_store::Array,phitest::Array,ytest::Array)
	T = size(theta_store,2)
	nlp=zeros(T);
	tmp=zeros(size(ytest,1))
	for epoch=1:T
		prob = cdf(Normal(),phitest'*theta_store[:,epoch] )
		nlp[epoch] = -sum(ytest.*log(prob) + (1-ytest).*log(1-prob))
		tmp += prob
	end
	tmp = tmp/T
	return(nlp,-sum(ytest.*log(tmp) + (1-ytest).*log(1-tmp)))
end

##learning hyperparameters for notensor model
@everywhere function nlp_full_hyper(theta_store::Array,Xtest::Array,ytest::Array,l_store::Array,SigmaRBF_store::Array)
	T = size(theta_store,2)
	nlp=zeros(T);
	tmp=zeros(size(ytest,1))
	for epoch=1:T
		phitest = GPTinf.featureNotensor(Xtest,n,l_store[epoch],SigmaRBF_store[epoch],seed);
		prob = cdf(Normal(),phitest'*theta_store[:,epoch])
		nlp[epoch] = -sum(ytest.*log(prob) + (1-ytest).*log(1-prob))
		tmp += prob
	end
	tmp = tmp/T
	return(nlp,-sum(ytest.*log(tmp) + (1-ytest).*log(1-tmp)))
end

@everywhere function nlp_hyper_func(w_store::Array,U_store::Array,Xtest::Array,ytest::Array,l_store::Array,SigmaRBF_store::Array)
	T = size(w_store,2)
	nlp=zeros(T);
	tmp=zeros(size(ytest,1))
	for epoch=1:T
		phitest = GPTinf.feature1(Xtest,n,l_store[epoch],SigmaRBF_store[epoch],scale,Zmat,bmat);
		prob = cdf(Normal(),pred(w_store[:,epoch],U_store[:,:,:,epoch],I,phitest))
		nlp[epoch] = -sum(ytest.*log(prob) + (1-ytest).*log(1-prob))
		tmp += prob
	end
	tmp = tmp/T
	return(nlp,-sum(ytest.*log(tmp) + (1-ytest).*log(1-tmp)))
end




#check whether the hyperparameters are sensitive
@everywhere maxepoch = 150;@everywhere eps = 0.01; @everywhere n=800;
@everywhere hyper_mat = exp(randn(10,2));
nlp = SharedArray(Float64, maxepoch*numbatches,10);nlpmean = SharedArray(Float64, 10);
@sync @parallel for i in 1:10
	length_scale, sigma_RBF = hyper_mat[i,:]; 
	phitrainfull = featureNotensor(Xtrain,n,length_scale,sigma_RBF,seed);
	phitestfull = featureNotensor(Xtest,n,length_scale,sigma_RBF,seed); 
	theta_store = GPTinf.GPT_probit_SGD(phitrainfull, ytrain, m, eps, burnin, maxepoch);
	nlp[:,i],nlpmean[i] = nlp_func_full(theta_store,phitestfull,ytest)
	println("length_scale = ", length_scale,  "; sigma_RBF = ", sigma_RBF)
end

figure()
for i=1:10 plot(nlp[:,i],label=round(hyper_mat[i,:],2)) end
plot([0,450],[118,118],label="GPML",color="k")
legend();title("nlp on test set, full theta SGD, transfusion data"); xlabel("epoch"); 

# nlp vs n for different m
@everywhere length_scale= 0.712; #exp(1.6732);
@everywhere sigma_RBF= 1.877#; exp(sqrt(0.7650));
@everywhere burnin=150; @everywhere maxepoch=200; @everywhere eps = 0.01
nvec = round(linspace(10,800,20));mvec = [50,100,150,200]; nlpmn=SharedArray(Float64,20,4)
@sync @parallel for j in 1:4
	m = mvec[j]
	for i in 1:20
		n = convert(Int,nvec[i])
		phitrainfull = featureNotensor(Xtrain,n,length_scale,sigma_RBF,seed);
		phitestfull = featureNotensor(Xtest,n,length_scale,sigma_RBF,seed); 
		theta_store = GPTinf.GPT_probit_SGD(phitrainfull, ytrain, m, eps, burnin, maxepoch);
		nlpmn[i,j] = nlp_func_full(theta_store,phitrainfull,ytrain)[2]
	end
end

figure()
subplot(121)
for j=1:4 plot(nvec,nlpmn[:,j],label=["m=",mvec[j]]) end
plot([0,800],[230,230],label="GPML",color="k")
legend();title("nlp on training set vs n for different batch size, \n full theta SGD, transfusion data"); xlabel("n"); 
subplot(122)
for j=1:4 plot(nvec,nlpmntest[:,j],label=["m=", mvec[j]]) end
plot([0,800],[118,118],label="GPML",color="k")
legend();title("nlp on test set vs n for different batch size, \n full theta SGD, transfusion data"); xlabel("n"); 

##learnin hyperparameters

theta_store,l_store,s_store = GPTinf.GPT_probit_SGD_adam(Xtrain, ytrain, I, n, m, alpha, burnin, maxepoch, hyp_init, seed);
tmp11,tmp22=nlp_full_hyper(theta_store,Xtest,ytest,l_store,s_store)

phitrainfull = featureNotensor(Xtrain,n,length_scale,sigma_RBF,seed);
phitestfull = featureNotensor(Xtest,n,length_scale,sigma_RBF,seed); 
theta_store = GPTinf.GPT_probit_SGD(phitrainfull, ytrain, m, eps, burnin, maxepoch);
tmp1,tmp2 = nlp_func_full(theta_store,phitestfull,ytest)

figure()
subplot(131);plot(l_store);title("traceplot of length_scale");subplot(132);plot(s_store);title("traceplot of sigma_RBF")
subplot(133);plot(tmp11,label="learning hyperparameters");plot(tmp1,label="fixed hyperparameters");title("nlp on test set");legend()


#############################################################################TENSOR MODEL##################################################
@everywhere length_scale= 1#exp(1.6732);
@everywhere sigma_RBF= 1#exp(sqrt(0.7650));
@everywhere phitrain=feature(Xtrain,n,length_scale,sigma_RBF,seed,scale);
@everywhere phitest=feature(Xtest,n,length_scale,sigma_RBF,seed,scale);

@everywhere epsw,epsU = 0.0001, 0.0001
ErrorRate=SharedArray(Float64,5);timer_r = SharedArray(Float64,5)
rvec = round(linspace(5,20,5));
@sync @parallel for i in 1:5
    r = convert(Int,rvec[i])
   	 tic(); 
  	 I=samplenz(r,D,Q,seed);
         w_store,U_store=GPTinf.GPT_SGLDERM_probit(phitrain,ytrain,I,r,Q,m, epsw, epsU, burnin, maxepoch,true);
	nlptest,nlpmean = nlp_func(w_store,U_store,phitest,ytest)
	nlptrain,nlpmean = nlp_func(w_store,U_store,phitrain,ytrain)
#not on steifel 
@everywhere phitrain=feature(Xtrain,n,length_scale,sigma_RBF,seed,1);
@everywhere phitest=feature(Xtest,n,length_scale,sigma_RBF,seed,1);
 	w_store,U_store=GPTinf.GPT_SGLDERM_probit(phitrain,ytrain,I,r,Q,m, epsw, epsU, burnin, maxepoch,false);
	nlptest1,nlpmean1 = nlp_func(w_store,U_store,phitest,ytest)
	nlptrain1,nlpmean1 = nlp_func(w_store,U_store,phitrain,ytrain)
	#w_store1,U_store1,l_store,SigmaRBF_store = GPTinf.GPT_SGLDERM_probit_SEM(Xtrain, ytrain, I, n, r, Q, m, epsw, epsU, burnin, maxepoch, hyp_init, Zmat,bmat);
	#nlp1,nlpmean1 = nlp_hyper_func(w_store1,U_store1,Xtrain,ytrain,l_store,SigmaRBF_store)
        #theta_store = GPTinf.GPT_probit_SGD(phitrainfull, ytrain, m, eps, burnin, maxepoch);
	#nlp2,nlpmean2 = nlp_func_full(theta_store,phitrainfull,ytrain)
  	 timer_r[i] = toq();
println(r)
end


figure()
subplot(121);plot(nlptrain,label="steifel manifold");plot(nlptrain1,label="Gaussian U");title("nlp on training set");xlabel("epoch");legend()
subplot(122);plot(nlptest,label="steifel manifold");plot(nlptest1,label="Gaussian U");title("nlp on test set");xlabel("epoch");legend()

=#

plot(nlp,label="GPT");plot(nlp2,label="full theta");plot(nlp1,label="GPT hyper");legend()
 ###works for epsU = 1e+7
##tuning epsw and epsU, results is good for epsU = 1e-5 and all epsw, best for epsw = 1e-3
nlp = SharedArray(Float64, maxepoch*numbatches,25);nlpmean= SharedArray(Float64,25)
@sync @parallel for iter = 1:25
    i,j=myt[iter];
    epsw=float(string("1e-",i)); epsU=float(string("1e-",j));
    w_store,U_store=GPT_SGLDERM_probit(phitrain,ytrain,I,r,Q,m, epsw, epsU, burnin, maxepoch);
    nlp[:,iter],nlpmean[iter] = nlp_func(w_store,U_store,phitest,ytest)
    println(";minnlp=",minimum(nlp[:,iter]),";minepoch=",indmin(nlp[:,iter]),";epsw=",epsw,";epsU=",epsU,";burnin=",burnin,";maxepoch=",maxepoch);
end


nlp1 = SharedArray(Float64, maxepoch*numbatches,16);nlpmean1= SharedArray(Float64,16)
@sync @parallel for iter = 1:16
    i,j=myt[iter];
    epsw=float(string("1e-",i)); epsU=float(string("1e-",j));
    w_store1,U_store1,l_store,SigmaRBF_store = GPTinf.GPT_SGLDERM_probit_SEM(Xtrain, ytrain, I, n, r, Q, m, epsw, epsU, burnin, maxepoch, hyp_init, Zmat,bmat);
    nlp1[:,iter],nlpmean1[iter] = nlp_hyper_func(w_store1,U_store1,Xtrain,ytrain,l_store,SigmaRBF_store)
    println(";minnlp=",minimum(nlp1[:,iter]),";minepoch=",indmin(nlp1[:,iter]),";epsw=",epsw,";epsU=",epsU,";burnin=",burnin,";maxepoch=",maxepoch);
end
 	


#=
outfile=open("tranfusion_nlp","a") #append to file
    println(outfile,"nlptrainmean=",nlptrainmean,"; nlptestmean=",nlptestmean,"; n=", n, "; r= ",r, "; m=",m ,"; Q=", Q,
		    "; nlptrain=", nlptrain, "; nlptest=", nlptest, "; maxepoch=", 20, "; number of averaged predictions= ", 20,
			"; nlp_tuning=", nlp, "; nlpmean_tuning=", nlpmean, "; myt=", myt)
    close(outfile)

figure()
subplot(121)
plot(nlptrain,label="GPTensor"); plot([0,80],[238,238],label="GPML EP"); xlabel("epoch");ylabel("negative log likelihood");legend()
title("transfusion data, training set")
subplot(122)
plot(nlptest,label="GPTensor"); plot([0,80],[131,131],label="GPML EP"); xlabel("epoch");ylabel("negative log likelihood");legend()
title("transfusion data, test set")
=#





