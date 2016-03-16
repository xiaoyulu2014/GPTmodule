@everywhere using DataFrames
@everywhere using GPTinf
using Iterators
using HDF5
### data processing
@everywhere function getdummy{R}(df::DataFrame, cname::Symbol, ::Type{R})
    darr = df[cname]
    vals = sort(levels(darr))#[2:end]
    namedict = Dict(vals, 1:length(vals))   
    arr = zeros(R, length(darr), length(namedict))
    for i=1:length(darr)
        if haskey(namedict, darr[i])
            arr[i, namedict[darr[i]]] = 1
        end        
    end
    newdf = convert(DataFrame, arr)
    names!(newdf, [symbol("$(cname)_$k") for k in vals])
    return newdf
end

@everywhere function convertdummy{R}(df::DataFrame, cnames::Array{Symbol}, ::Type{R})
    # consider every variable from cnames as categorical
    # and convert them into set of dummy variables,
    # return new dataframe
    newdf = DataFrame()
    for cname in names(df)
        if !in(cname, cnames)
            newdf[cname] = df[cname]
        else
            dummydf = getdummy(df, cname, R)
            for dummyname in names(dummydf)
                newdf[dummyname] = dummydf[dummyname]
            end
        end
    end
    return newdf
end

@everywhere convertdummy(df::DataFrame, cnames::Array{Symbol}) = convertdummy(df, cnames, Int32)

@everywhere function bin_age(age::Array)
	q=quantile(age,[0.2,0.4,0.6,0.8,1.0])
	indmin(q.<UserData[30,2])
        map(x->indmin(q.<x),age)
end

##data clearing
@everywhere UserData = readdlm("/homes/xlu/Downloads/ml-100k/u.user", '|');
@everywhere MovieData = readdlm("/homes/xlu/Downloads/ml-100k/u.item",'|');
@everywhere Rating = readdlm("/homes/xlu/Downloads/ml-100k/u.data",Float64);

@everywhere Ntrain = 80000;
@everywhere Ntest = 20000;
@everywhere UserData[:,2] = bin_age(UserData[:,2])
@everywhere UserData = convertdummy(convert(DataFrame,UserData),[:x2,:x3,:x4])[:,1:end-1];
@everywhere MovieData = MovieData[:,[1,6:end]];
@everywhere UserData = convert(Array{Float64,2},UserData)[:,2:end];
@everywhere MovieData = convert(Array{Float64,2},MovieData)[:,3:end]; 

@everywhere ytrain = Rating[1:Ntrain,3];
@everywhere ytest = Rating[(Ntrain+1):(Ntrain+Ntest),3];
@everywhere ytrainMean=mean(ytrain);
@everywhere ytrainStd=std(ytrain);
@everywhere ytrain=datawhitening(ytrain);
@everywhere ytest = (ytest-ytrainMean)/ytrainStd;
@everywhere n = 500; 
@everywhere M = 5;
@everywhere burnin=10;
@everywhere r = 15
@everywhere using Distributions
@everywhere param_seed=17;
@everywhere m = 100;
@everywhere maxepoch = 100;

@everywhere epsw=0.0
@everywhere signal_varvec=[0.001,0.01];
@everywhere var_uvec=[0.01,0.1];
@everywhere epsUvec=[0.00001,0.0000001];
@everywhere numbatches=int(ceil(maximum(Ntrain)/m));
@everywhere myn=2*2*2
@everywhere t=Iterators.product(signal_varvec,var_uvec,epsUvec)
@everywhere myt=Array(Any,myn);
@everywhere it=1;
@everywhere for prod in t
	myt[it]=prod;
        it+=1;
        end


function GPT_PMF(phiU::Array, phiV::Array,y::Array, signal_var::Real,var_u::Real, r::Integer, m::Integer, epsw::Real, epsU::Real, maxepoch::Integer,param_seed::Integer,phiUtest::Array, phiVtest::Array; langevin::Bool=true, stiefel::Bool=true)
 
    n1,N=size(phiU); n2=size(phiV,1)
    numbatches=int(ceil(N/m))
    var_w=1;
    trainRMSE= Array(Float64,maxepoch)
    testRMSE = Array(Float64,maxepoch)
    
    # initialise w,U^(k)
    srand(param_seed);
    w_store=Array(Float64,r,maxepoch)
    U_store=Array(Float64,n1,r,maxepoch)
    V_store=Array(Float64,n2,r,maxepoch)
    #w=sqrt(var_w)*randn(r)
    w=ones(r)
 
    if stiefel
	Z1=randn(r,n1);	Z2=randn(r,n2)
	U=transpose(\(sqrtm(Z1*Z1'),Z1))
	V=transpose(\(sqrtm(Z2*Z2'),Z2))
    else U=sqrt(var_u)*randn(n1,r);V=sqrt(var_u)*randn(n2,r)
    end
    U=HDF5.h5read("/homes/xlu/Downloads/ml100k_UVhyperparams.h5","U")
    V=HDF5.h5read("/homes/xlu/Downloads/ml100k_UVhyperparams.h5","V")

    for epoch=1:maxepoch
        #randomly permute training data and divide into mini_batches of size m
        perm=randperm(N)
        phiU_perm=phiU[:,perm]; phiV_perm=phiV[:,perm];y_perm=y[perm];

        # run SGLD on w and SGLDERM on U
        for batch=1:numbatches
            # random samples for the stochastic gradient
            idx=(m*(batch-1)+1):min(m*batch,N)
            phi_batchU=phiU_perm[:,idx];phi_batchV=phiV_perm[:,idx]; y_batch=y_perm[idx];
            batch_size=length(idx) 

	    PhidotU=U'*phi_batchU; PhidotV=V'*phi_batchV ;fw=PhidotU.*PhidotV;
	    fhat=fw'*w;
            # now can compute gradw, the stochastic gradient of log post wrt w
           # gradw=(N/batch_size)*fw*(y_batch-fhat)/signal_var-w/var_w

            gradU=Array(Float64,n1,r); gradV=Array(Float64,n2,r);PsiU=Array(Float64,n1*r,m);PsiV=Array(Float64,n2*r,m);
	    for i=1:m
		PsiU[:,i]=kron(w.*PhidotV[:,i],phi_batchU[:,i])
		PsiV[:,i]=kron(w.*PhidotU[:,i],phi_batchV[:,i])
	    end
	    if stiefel
		gradU=reshape((N/batch_size)*PsiU*((y_batch-fhat)/signal_var),n1,r)
		gradV=reshape((N/batch_size)*PsiV*((y_batch-fhat)/signal_var),n2,r)
	    else 
		gradU=reshape((N/batch_size)*PsiU*(y_batch-fhat)/signal_var,n1,r)-U/var_u
		gradV=reshape((N/batch_size)*PsiV*(y_batch-fhat)/signal_var,n2,r)-V/var_u
	    end
	
    
            # update w
	   #= if langevin
		w+=epsw*gradw/2+sqrt(epsw)*randn(r)
	    else w+=epsw*gradw/2
	    end=#

            # update U
	    if langevin
		if stiefel
		   momU=proj(U,sqrt(epsU)*gradU/2+randn(n1,r));	momV=proj(V,sqrt(epsU)*gradV/2+randn(n2,r))
		   U=geod(U,momU,sqrt(epsU)); V=geod(V,momV,sqrt(epsU));
		else U+=epsU*gradU/2+sqrt(epsU)*randn(n1,r);V+=epsU*gradV/2+sqrt(epsU)*randn(n2,r)
		end
	    else U+=epsU*gradU/2;V+=epsU*gradV/2
	    end
	end

	     #   w_store[:,epoch]=w
	     #   U_store[:,:,epoch]=U
	     #   V_store[:,:,epoch]=V

                trainpred = GPTinf.pred_CF(w,U,V,phiU,phiV)
		trainRMSE[epoch] = ytrainStd*norm(ytrain-trainpred)/sqrt(Ntrain)
		testpred = GPTinf.pred_CF(w,U,V,phiUtest,phiVtest)
		testRMSE[epoch] = ytrainStd*norm(ytest-testpred)/sqrt(Ntest)
		tmp1,tmp2=trainRMSE[epoch],testRMSE[epoch]
		println("epoch=$epoch, trainRMSE=$tmp1, testRMSE=$tmp2")
    end
    return trainRMSE,testRMSE
end

phiUser = eye(size(UserData,1));
phiMovie = eye(size(MovieData,1));
phiUtrain=Array(Float64,size(phiUser,1),Ntrain);phiVtrain=Array(Float64,size(phiMovie,1),Ntrain);
phiUtest=Array(Float64,size(phiUser,1),Ntest);phiVtest=Array(Float64,size(phiMovie,1),Ntest);
for i=1:Ntrain
    phiUtrain[:,i]=phiUser[:,Rating[i,1]]
    phiVtrain[:,i]=phiMovie[:,Rating[i,2]]
end
 for i=1:Ntest
	phiUtest[:,i]=phiUser[:,Rating[Ntrain+i,1]]
	phiVtest[:,i]=phiMovie[:,Rating[Ntrain+i,2]]
 end
w_store,U_store,V_store=GPT_PMF(phiUtrain, phiVtrain, ytrain,signal_var, var_u, r, m, epsw, epsU, maxepoch,param_seed,phiUtest,phiVtest,langevin=false, stiefel=false);


#=
##cross validation
##signal_var=0.001;var_u=0.01; epsU=1.0e-7;minRMSE=0.9548802146942493;true;false
testRMSE = SharedArray(Float64,maxepoch-burnin+1,myn); trainRMSE= SharedArray(Float64,maxepoch-burnin+1,myn);timer=SharedArray(Float64,myn);
@sync @parallel for k=1:myn
	signal_var,var_u,epsU=myt[k]
	phiUser = eye(size(UserData,1));
	phiMovie = eye(size(MovieData,1));
	phiUtrain=Array(Float64,size(phiUser,1),Ntrain);phiVtrain=Array(Float64,size(phiMovie,1),Ntrain);
	phiUtest=Array(Float64,size(phiUser,1),Ntest);phiVtest=Array(Float64,size(phiMovie,1),Ntest);
	for i=1:Ntrain
	    phiUtrain[:,i]=phiUser[:,Rating[i,1]]
	    phiVtrain[:,i]=phiMovie[:,Rating[i,2]]
	end
	 for i=1:Ntest
		phiUtest[:,i]=phiUser[:,Rating[Ntrain+i,1]]
		phiVtest[:,i]=phiMovie[:,Rating[Ntrain+i,2]]
	 end
	tic(); 
	w_store,U_store,V_store=GPTinf.GPT_CF(phiUtrain, phiVtrain, ytrain,signal_var, var_u, r, m, epsw, epsU, maxepoch,param_seed,true,false);
	numbatches=int(ceil(Ntrain/m));trainpred=zeros(Ntrain);testpred=zeros(Ntest)
	#make predictions for every epoch
	for epoch=burnin:maxepoch
		trainpred = (trainpred + GPTinf.pred_CF(w_store[:,epoch],U_store[:,:,epoch],V_store[:,:,epoch],phiUtrain,phiVtrain))/2
		trainRMSE[epoch-burnin+1,k]=ytrainStd*norm(ytrain-trainpred)/sqrt(Ntrain)
		testpred = (testpred + GPTinf.pred_CF(w_store[:,epoch],U_store[:,:,epoch],V_store[:,:,epoch],phiUtest,phiVtest))/2
		testRMSE[epoch-burnin+1,k]=ytrainStd*norm(ytest-testpred)/sqrt(Ntest)
	end
	timer[k] = toq(); 
	println(";signal_var=",signal_var,";var_u=",var_u,"; epsU=",epsU, ";minRMSE=",minimum(testRMSE[:,k]),";minepoch=",indmin(testRMSE[:,k]),";timer=",timer[k]);
end
=#

@everywhere signal_varvec=[0.001,0.01,0.1];
@everywhere var_uvec=[0.01,0.1,0.5];
@everywhere myn1=3*3
@everywhere t1=Iterators.product(signal_varvec,var_uvec)
@everywhere myt1=Array(Any,myn1);
@everywhere it=1;
@everywhere for prod in t1
	myt1[it]=prod;
        it+=1;
        end
testRMSEgibbs = SharedArray(Float64,maxepoch-burnin+1,myn1); trainRMSEgibbs = SharedArray(Float64,maxepoch-burnin+1,myn1);timergibbs =SharedArray(Float64,myn1)
@sync @parallel for k=1:myn1
	signal_var,var_u=myt1[k];
	phiUser = eye(size(UserData,1));
	phiMovie = eye(size(MovieData,1));
	phiUtrain=Array(Float64,size(phiUser,1),Ntrain);phiVtrain=Array(Float64,size(phiMovie,1),Ntrain);
	phiUtest=Array(Float64,size(phiUser,1),Ntest);phiVtest=Array(Float64,size(phiMovie,1),Ntest);
	for i=1:Ntrain
	    phiUtrain[:,i]=phiUser[:,Rating[i,1]]
	    phiVtrain[:,i]=phiMovie[:,Rating[i,2]]
	end
	 for i=1:Ntest
		phiUtest[:,i]=phiUser[:,Rating[Ntrain+i,1]]
		phiVtest[:,i]=phiMovie[:,Rating[Ntrain+i,2]]
	 end
	 tic();
	 wGibbs,UGibbs,VGibbs=GPTinf.GPT_CFgibbs(phiUtrain, phiVtrain,ytrain, signal_var, var_u, r, maxepoch);
         trainpred=zeros(Ntrain);testpred=zeros(Ntest)
	 for epoch=burnin:maxepoch   ##average over samples
	        trainpred = (trainpred + GPTinf.pred_CF(wGibbs[:,epoch],UGibbs[:,:,epoch],VGibbs[:,:,epoch],phiUtrain,phiVtrain))/2
		trainRMSEgibbs[epoch-burnin+1,k]=ytrainStd*norm(ytrain-trainpred)/sqrt(Ntrain)
		testpred = (testpred + GPTinf.pred_CF(wGibbs[:,epoch],UGibbs[:,:,epoch],VGibbs[:,:,epoch],phiUtest,phiVtest))/2
		testRMSEgibbs[epoch-burnin+1,k]=ytrainStd*norm(ytest-testpred)/sqrt(Ntest)
	end
	timergibbs[k]=toq(); 
	println(";signal_var=",signal_var,";var_u=",var_u,";minRMSEgibbs=",minimum(testRMSEgibbs[:,k]),";minepoch=",indmin(testRMSEgibbs[:,k]),";timergibbs=",timergibbs[k]);
end


