@everywhere using DataFrames
@everywhere using GPTinf
@everywhere using Distributions
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

@everywhere function cutoff!(pred::Vector)
	idxlow=(pred.<1); pred[idxlow]=1;
	idxhigh=(pred.>5); pred[idxhigh]=5;
end


@everywhere function GPT_fullw(Rating::Array,UserData::Array,MovieData::Array,Ratingtest::Array, signal_var::Real, var_u::Real,var_w::Real, w_init::Array, m::Integer, epsw::Real, epsU::Real,burnin::Integer, maxepoch::Integer, param_seed::Integer; langevin::Bool=false)
	N=size(Rating,1);Ntest=size(Ratingtest,1);
	n1,D1=size(UserData); 
	n2,D2=size(MovieData);
	numbatches=int(ceil(N/m));
	r=size(w_init,1);
	
	# initialise w,U,V
	srand(param_seed);
	w=w_init;
        U=randn(n1,r)*sqrt(var_u);V=randn(n2,r)*sqrt(var_u)
	testRMSEvec=zeros(maxepoch+burnin)
	testpred=zeros(Ntest)
	counter=0;

	for epoch=1:(burnin+maxepoch)
		#randomly permute training data and divide into mini_batches of size m
                perm=randperm(N)
		shuffledRatings=Rating[perm,:]
		
		#run SGLD on w and U
		for batch=1:numbatches
                      # random samples for the stochastic gradient
                       idx=(m*(batch-1)+1):min(m*batch,N);
			batch_size=length(idx);
			batch_ratings=shuffledRatings[idx,:];
			
			# compute gradients
			gradw=zeros(r,r);
			gradU=zeros(n1,r);
			gradV=zeros(n2,r);
			for ii=1:batch_size
				user=batch_ratings[ii,1]; movie=batch_ratings[ii,2]; rating=batch_ratings[ii,3];
				pred=sum((U[user,:]*w).*V[movie,:])
				gradw+=(rating-pred)*U[user,:]'*V[movie,:]/signal_var
				gradU[user,:]+=(rating-pred)*V[movie,:]*w'/signal_var
				gradV[movie,:]+=(rating-pred)*U[user,:]*w/signal_var
			end
			gradw=N/batch_size*gradw-w/var_w;
			gradU=N/batch_size*gradU-U/var_u;
			gradV=N/batch_size*gradV-V/var_u;

			# update w, U, V
			w+=epsw*gradw/2;
			U+=epsU*gradU/2; V+=epsU*gradV/2;
			if langevin w+=sqrt(epsw)*randn(r,r);U+=sqrt(epsU)*randn(n1,r); V+=sqrt(epsU)*randn(n2,r); end
		end
		
		for i=1:Ntest
			user=Ratingtest[i,1]; movie=Ratingtest[i,2];
			testpred[i]=(testpred[i]*counter+sum((U[user,:]*w).*V[movie,:]))/(counter+1)
		end
		final_testpred=testpred*ytrainStd+ytrainMean;
		cutoff!(final_testpred);
		testRMSE=sqrt(sum((ytrainStd*Ratingtest[:,3]+ytrainMean-final_testpred).^2)/Ntest)
		if epoch > burnin counter+=1 end
		testRMSEvec[epoch]=testRMSE 
		println("epoch=$epoch, testRMSE=$testRMSE")

	end
	return testRMSEvec
end



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
@everywhere n1,D1=size(UserData); 
@everywhere n2,D2=size(MovieData); 
@everywhere ytrain = Rating[1:Ntrain,3];
@everywhere ytest = Rating[Ntrain+1:Ntrain+Ntest,3];
@everywhere ytrainMean=mean(ytrain);
@everywhere ytrainStd=std(ytrain);
@everywhere ytrain=(ytrain-ytrainMean)/ytrainStd;
@everywhere ytest = (ytest-ytrainMean)/ytrainStd;
@everywhere Ratingtrain=hcat(Rating[1:Ntrain,1:2],ytrain);
@everywhere Ratingtest=hcat(Rating[Ntrain+1:Ntrain+Ntest,1:2],ytest);
@everywhere n = 30; 
@everywhere M = 5;
@everywhere r = 15
@everywhere signal_var = 0.8 #0.5;
@everywhere param_seed=17;
@everywhere m = 100;
@everywhere var_w=1;
@everywhere srand(17)
@everywhere w_init=var_w*randn(r,r)
@everywhere epsw=1.75e-6;
@everywhere epsU=2.625e-6;
@everywhere maxepoch = 120;
@everywhere burnin=50;
@everywhere var_u=0.01

#=
grid=Iterators.product(var_uvec,var_vvec)
n1,n2=length(var_uvec),length(var_vvec)
dim=*(n1,n2)
mygrid=Array(Any,dim);
it=1;
for prod in grid
	mygrid[it]=prod;
	it+=1;
end


trainRMSEvec=SharedArray(Float64,dim);testRMSEvec=SharedArray(Float64,dim);
@sync @parallel for i=1:dim
	var_u,var_v = mygrid[i]
	traintmp,testtmp=GPT_fullw(Ratingtrain,UserData,MovieData,Ratingtest,signal_var, var_u, var_w, w_init,m, epsw, epsU, burnin, maxepoch, param_seed,langevin=false);
	trainRMSEvec[i]=minimum(traintmp);	testRMSEvec[i]=minimum(testtmp);
	println("; min(testRMSE)=", testRMSEvec[i], "; minepoch =", indmin(testtmp), "; var_u=", var_u, "; var_v=", var_v)
end

=#





#######################################################################################################################################################################################
###side info###
@everywhere function GPT_fullwside(Rating::Array,UserData::Array,MovieData::Array,Ratingtest::Array, signal_var::Real, var_u::Real,var_w::Real, a::Real, b::Real, c::Real, m::Integer, epsw::Real, epsU::Real,burnin::Integer, maxepoch::Integer, param_seed::Integer; langevin::Bool=false)
	N=size(Rating,1);Ntest=size(Ratingtest,1);
	n1,D1=size(UserData); 
	n2,D2=size(MovieData);
	d1,d2=length(find(UserData[1,:])),length(find(MovieData[1,:]))
	numbatches=int(ceil(N/m));
	r=size(w_init,1);
	# initialise U,V
	srand(param_seed);
       # U=randn(n1,r)*sqrt(var_u);V=randn(n2,r)*sqrt(var_u)
        U=HDF5.h5read("/homes/xlu/Downloads/ml100k_UVhyperparams.h5","U")
        V=HDF5.h5read("/homes/xlu/Downloads/ml100k_UVhyperparams.h5","V")
        Us=randn(D1,r)*sqrt(var_u);Vs=randn(D2,r)*sqrt(var_u)
	testRMSEvec=zeros(maxepoch+burnin)
	testpred=zeros(Ntest)
	final_testpred=zeros(Ntest)
	counter=0;
	I=Array(Float64,n1,d1);
	for i=1:n1 I[i,:]=find(UserData[i,:]) end


	for epoch=1:(burnin+maxepoch)
		#randomly permute training data and divide into mini_batches of size m
                perm=randperm(N)
		shuffledRatings=Rating[perm,:]
		
		#run SGLD on and U
		for batch=1:numbatches
                      # random samples for the stochastic gradient
                       idx=(m*(batch-1)+1):min(m*batch,N);
			batch_size=length(idx);
			batch_ratings=shuffledRatings[idx,:];
			
			# compute gradients
			gradU=zeros(n1,r);gradUs=zeros(D1,r);
			gradV=zeros(n2,r);gradVs=zeros(D2,r);
			for ii=1:batch_size
				user=batch_ratings[ii,1]; movie=batch_ratings[ii,2]; rating=batch_ratings[ii,3];
				demo_idx= I[user,:][:];  genre_idx = find(MovieData[movie,:])
				pred=sum(a*(U[user,:]+b*sum(Us[demo_idx,:],1)) .* (V[movie,:]+c*sum(Vs[genre_idx,:],1)))
				tmpU=a*(rating-pred)* (V[movie,:]+c*sum(Vs[genre_idx,:],1))/signal_var
				tmpV=a*(rating-pred)* (U[user,:]+b*sum(Us[I[user,:][:],:],1)) /signal_var
				gradU[user,:]+=tmpU
				gradV[movie,:]+=tmpV
				gradUs[demo_idx,:] += b*repmat(tmpU,d1,1)
				gradVs[genre_idx,:] += c*repmat(tmpV,length(genre_idx),1)
			end
			gradU=N/batch_size*gradU-U/var_u;
			gradV=N/batch_size*gradV-V/var_u;
			gradUs=N/batch_size*gradUs-Us/var_u;
			gradVs=N/batch_size*gradVs-Vs/var_u;

			# update U, V
			U+=epsU*gradU/2; V+=epsU*gradV/2; Us+=epsU*gradUs/2; Vs+=epsU*gradVs/2;
		end
		
		for i=1:Ntest
			user=Ratingtest[i,1]; movie=Ratingtest[i,2];
			demo_idx= I[user,:][:];  genre_idx = find(MovieData[movie,:])
			testpred[i]=(testpred[i]*counter+ sum(a*(U[user,:]+b*sum(Us[demo_idx,:],1)) .* (V[movie,:]+c*sum(Vs[genre_idx,:],1))) )/(counter+1)
		end
		final_testpred=testpred*ytrainStd+ytrainMean;
		cutoff!(final_testpred);
		testRMSE=sqrt(sum((ytrainStd*Ratingtest[:,3]+ytrainMean-final_testpred).^2)/Ntest)
		if epoch > burnin counter+=1 end
		testRMSEvec[epoch]=testRMSE 
		#println("epoch=$epoch, testRMSE=$testRMSE")

	end
	return testRMSEvec,final_testpred
end

a=0.2;b=c=5;epsU=1.0e-8;signal_var=0.001
testtmpside,yfit=GPT_fullwside(Ratingtrain,UserData,MovieData,Ratingtest,signal_var, var_u, var_w, a,b,c,m, epsw, epsU, burnin, maxepoch, param_seed,langevin=false);

#=
h5open("plot/wk10/sideinfo.h5", "w") do file
    write(file, "RMSEside_init",RMSEside_init,"yfit_init",yfit_init,"RMSEside", testtmpside, "yfit", yfit, "a", a, "b", b, "c", c, "signal_var", signal_var,"var_u", var_u,"epsU",epsU)
end
=#

RMSEside,yfit,RMSEside_init,yfit_init = h5open("plot/wk10/sideinfo.h5", "r") do file
    read(file, "RMSEside","yfit","RMSEside_init","yfit_init")
end


RMSE_init,y_init=testtmp,yfit
#a=0.4,b=0.5,c=0.5,epsU=5.0e-9,signal_var=0.001,var_u=0.01,min(RMSE)=0.908 with initilization learnt from PMF, W=I

#####tuning a,b and c

@everywhere using Iterators
@everywhere epsU=1.0e-8
@everywhere signal_var=0.001
@everywhere var_u=0.01
@everywhere avec = round(linspace(0.01,1,5),2)
@everywhere bvec = round(linspace(0.01,2,5),2)
@everywhere cvec = round(linspace(0.01,2,5),2)
@everywhere n1,n2,n3=length(avec),length(bvec),length(cvec)
@everywhere grid=Iterators.product(avec,bvec,cvec)
@everywhere dim=*(n1,n2,n3)
@everywhere mygrid=Array(Any,dim);
@everywhere it=1;
@everywhere for prod in grid
	mygrid[it]=prod;
        it+=1;
        end

RMSEvec=SharedArray(Float64,maxepoch+burnin,dim);yfitvec=SharedArray(Float64,Ntest,dim);
@sync @parallel for i=1:dim
	a,b,c = mygrid[i]
	RMSEvec[:,i],yfitvec[:,i]=GPT_fullwside(Ratingtrain,UserData,MovieData,Ratingtest,signal_var, var_u, var_w, a,b,c,m, epsw, epsU, burnin, maxepoch, param_seed,langevin=false);
	println("a= ",a, "; b= ", b, "; c= ", c, "; minRMSE=", minimum(RMSEvec[:,i]), ";minepoch = ", indmin(RMSEvec[:,i]) )
end

#no side info
@everywhere epsU=5.0e-9
RMSEvec0=SharedArray(Float64,maxepoch+burnin,n1);yfitvec0=SharedArray(Float64,Ntest,n1);
@sync @parallel for i=1:n1
	a=avec[i];b=c=0;
	RMSEvec0[:,i],yfitvec0[:,i]=GPT_fullwside(Ratingtrain,UserData,MovieData,Ratingtest,signal_var, var_u, var_w, a,b,c,m, epsw, epsU, burnin, maxepoch, param_seed,langevin=false);
	println("a= ",a, "; b= ", b, "; c= ", c, "; minRMSE=", minimum(RMSEvec0[:,i]), ";minepoch = ", indmin(RMSEvec0[:,i]) )
end















#=
@everywhere beta0=2
@everywhere nu=r-1;
@everywhere W0 = eye(nu)
@everywhere srand(17)
@everywhere Prec_U = rand(Wishart(nu,W0))
@everywhere Prec_V = rand(Wishart(nu,W0))
@everywhere Sigma_U = inv(Prec_U); 
@everywhere Sigma_V = inv(Prec_V); 
@everywhere Mu_U = rand(MvNormal(zeros(nu),Sigma_U/beta0))
@everywhere Mu_V = rand(MvNormal(zeros(nu),Sigma_U/beta0))
@everywhere L_U=chol(Sigma_U,:L)
@everywhere L_V=chol(Sigma_V,:L)
@everywhere W = hcat(vcat(L_U'*L_V,Mu_U'*L_V),vcat(L_U'*Mu_V,Mu_U'*Mu_V))
@everywhere function rotation_func(r)
			Q = zeros(r,r)
			while det(Q)!=1 
				A = randn(r,r)
				Q = qr(A)[1]
			end
			return Q
	     end	
@everywhere w_init = rotation_func(r)*W
@everywhere w_init = W
x=1:(r);y=1:(r)
xgrid = repmat(x',r,1)
ygrid = repmat(y,1,r)
plot_surface(xgrid,ygrid,W)
=#
epsw=1.75e-6; epsU=2.625e-6;
#=using HDF5
h5open("w_init.h5", "w") do file
    write(file, "w_init", w_init)
end=#


#=
using HDF5
h5open("RMSEtestW.h5", "w") do file
    write(file, "learnWtestRMSEvec", learnWtestRMSEvec, "learnWtrainRMSEvec", learnWtrainRMSEvec, "fixWtestRMSEvec", fixWtestRMSEvec, "fixWtrainRMSEvec", fixWtrainRMSEvec
,"randWtestRMSEvec", randWtestRMSEvec, "randWtrainRMSEvec", randWtrainRMSEvec)
end
randWtestRMSEvec = h5open("RMSEtestW.h5", "r") do file
    read(file, "randWtestRMSEvec")
end

plot(fixWtestRMSEvec,label="fixed W"); plot(learnWtestRMSEvec,label="learning W"); plot│^CERROR: interrupt
(randWtestRMSEvec,label="random fixed W");legend() 
=#


