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
@everywhere length_scale= 1.0 #exp(1.6732);
@everywhere sigma_RBF= 1.0 #exp(sqrt(0.7650));
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
@everywhere burnin=100;
@everywhere Q=100;   #200
@everywhere n=80;
@everywhere r = 20;

@everywhere scale = sqrt(n/(Q^(1/D)));
@everywhere phitrain=feature(Xtrain,n,length_scale,sigma_RBF,seed,scale);
@everywhere phitest=feature(Xtest,n,length_scale,sigma_RBF,seed,scale);
@everywhere I=samplenz(r,D,Q,seed); 
@everywhere m = 150;
@everywhere maxepoch = 50;


@everywhere t=Iterators.product(3:7,5:9)
@everywhere myt=Array(Any,25);
@everywhere it=1;
@everywhere for prod in t
	myt[it]=prod;
        it+=1;
        end


# error rate as a function of r, on training set
#=
@everywhere epsw,epsU = 0.0001, 0.0001
ErrorRate=SharedArray(Float64,5);timer_r = SharedArray(Float64,5)
rvec = round(linspace(5,20,5));
@sync @parallel for i in 1:5
    r = convert(Int,rvec[i])
   	 tic(); 
  	 I=samplenz(r,D,Q,seed);
         w_store,U_store=GPT_SGLDERM_probit(phitrain,ytrain,I,r,Q,m, epsw, epsU, burnin, maxepoch);
	w_store1,U_store1,l_store,SigmaRBF_store = GPTinf.GPT_SGLDERM_probit_SEM(Xtrain, ytrain, I, n, r, Q, m, epsw, epsU, burnin, maxepoch, [1.0,1.0], seed);
	nlp,nlpmean = nlp_func(w_store,U_store,phitrain,ytrain)
	nlp1,nlpmean1 = nlp_hyper_func(w_store1,U_store1,Xtrain,ytrain,l_store,SigmaRBF_store)
  	 timer_r[i] = toq();
println(r)
end
=#


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



@everywhere function nlp_hyper_func(w_store::Array,U_store::Array,Xtest::Array,ytest::Array,l_store::Array,SigmaRBF_store::Array)
	T = size(w_store,2)
	nlp=zeros(T);
	tmp=zeros(size(ytest,1))
	for epoch=1:T
		phitest = feature(Xtest,n,length_scale,sigma_RBF,seed,scale);
		prob = cdf(Normal(),pred(w_store[:,epoch],U_store[:,:,:,epoch],I,phitest))
		nlp[epoch] = -sum(ytest.*log(prob) + (1-ytest).*log(1-prob))
		tmp += prob
	end
	tmp = tmp/T
	return(nlp,-sum(ytest.*log(tmp) + (1-ytest).*log(1-tmp)))
end


##tuning epsw and epsU, results is good for epsU = 1e-5 and all epsw, best for epsw = 1e-3
nlp = SharedArray(Float64, maxepoch*numbatches,25);nlpmean= SharedArray(Float64,25)
@sync @parallel for iter = 1:25
    i,j=myt[iter];
    epsw=float(string("1e-",i)); epsU=float(string("1e-",j));
    w_store,U_store=GPT_SGLDERM_probit(phitrain,ytrain,I,r,Q,m, epsw, epsU, burnin, maxepoch);
    nlp[:,iter],nlpmean[iter] = nlp_func(w_store,U_store,phitest,ytest)
    println(";minnlp=",minimum(nlp[:,iter]),";minepoch=",indmin(nlp[:,iter]),";epsw=",epsw,";epsU=",epsU,";burnin=",burnin,";maxepoch=",maxepoch);
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





