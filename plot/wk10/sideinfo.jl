@everywhere push!(LOAD_PATH,"/homes/xlu/Documents/GPTmodule")
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


#######################################################################################################################################################################################
###side info###
@everywhere function GPT_side(Rating::Array,UserData::Array,MovieData::Array,Ratingtest::Array,r::Real, signal_var::Real, var_u::Real, a::Real, b::Real, c::Real, m::Integer, epsU::Real,burnin::Integer, maxepoch::Integer, param_seed::Integer; langevin::Bool=false)
	N=size(Rating,1);Ntest=size(Ratingtest,1);
	n1,D1=size(UserData); 
	n2,D2=size(MovieData);
	d1,d2=length(find(UserData[1,:])),length(find(MovieData[1,:]))
	numbatches=int(ceil(N/m));
	# initialise U,V
	srand(param_seed);
        U=randn(n1,r)*sqrt(var_u);V=randn(n2,r)*sqrt(var_u)
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
				pred=sum( a*(U[user,:]+b*sum(Us[demo_idx,:],1)) .* (V[movie,:]+c*sum(Vs[genre_idx,:],1)) )
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
		println("epoch=$epoch, testRMSE=$testRMSE")

	end
	return testRMSEvec,final_testpred
end

#####tuning a,b and c

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
@everywhere signal_var = 0.001 
@everywhere param_seed=17;
@everywhere m = 100;
@everywhere maxepoch = 100;
@everywhere burnin=0;
@everywhere var_u=0.01
@everywhere var_w=1
@everywhere signal_var=0.001

@everywhere using Iterators
@everywhere avec = round(linspace(0.25,1,4),2)
@everywhere bvec = round(linspace(0,0.5,5),2)
@everywhere cvec = round(linspace(0,0.5,5),2)
#@everywhere cvec = round(linspace(0,2,5),2)
#@everywhere epsUvec = [1.0e-9,5.0e-9,1.0e-8,1.0e-7]
@everywhere epsUvec = [1.0e-8,1.0e-7]
@everywhere n1,n2,n3,n4=length(avec),length(bvec),length(cvec),length(epsUvec)
@everywhere grid=Iterators.product(avec,bvec,cvec,epsUvec)
@everywhere dim=*(n1,n2,n3,n4)
@everywhere mygrid=Array(Any,dim);
@everywhere it=1;
@everywhere for prod in grid
	mygrid[it]=prod;
        it+=1;
        end

RMSEvec=SharedArray(Float64,maxepoch+burnin,dim);yfitvec=SharedArray(Float64,Ntest,dim);
@sync @parallel for i=1:dim
	a,b,c,epsU = mygrid[i]
	RMSEvec[:,i],yfitvec[:,i]=GPT_side(Ratingtrain,UserData,MovieData,Ratingtest,r,signal_var, var_u, a,b,c,m, epsU, burnin, maxepoch, param_seed,langevin=false);
	println("a= ",a, "; b= ", b, "; c= ", c, "; epsU=", epsU, "; minRMSE=", minimum(RMSEvec[:,i]), ";minepoch = ", indmin(RMSEvec[:,i]) )
end
#=
RMSEvec=reshape(RMSEvec,maxepoch+burnin,n1,n2,n3,n4); 
h5open("RMSEside_finergrid.h5", "w") do file
    write(file, "RMSEvec", convert(Array,RMSEvec),"avec",avec,"bvec",bvec,"cvec",cvec,"epsUvec",epsUvec,"maxepoch",maxepoch,"burnin",burnin)
end
=#
###############################################################################################

#=
RMSEvec,avec,bvec,cvec,epsUvec,maxepoch,burnin = h5open("RMSEside_finergrid.h5", "r") do file
    read(file, "RMSEvec","avec","bvec","cvec","epsUvec","maxepoch","burnin")
end
=#

n1,n2,n3,n4=length(avec),length(bvec),length(cvec),length(epsUvec)
###find the indices when there is no side information
noside_id=[]
for i=1:n4
	noside_id=[noside_id,(i-1)*(*(n1,n2,n3))+1:(i-1)*(*(n1,n2,n3))+n1]
end

###examine the effect of a and epsU for no side info (W=I), data="RMSEside.h5"
RMSE_noside=reshape(reshape(RMSEvec,maxepoch+burnin,*(n1,n2,n3,n4))[:,noside_id],maxepoch+burnin,n1,n4)
for i=1:4,j=1:4                                                                                 
       a=avec[i];epsU=epsUvec[j]                                                                       
       plot(RMSE_noside[:,i,j],label="a=$a,epsU=$epsU")                                                 
end  


fig, axs = plt[:subplots](2,2)
for i=1:n1
	a=avec[i]
	for j=1:n4
		epsU=epsUvec[j]
		axs[i][:plot](RMSE_noside[:,i,j],label="epsU=$epsU")
	end
	axs[i][:set_title]("a=$a")
	axs[i][:legend](loc="best")
	axs[i][:set_xlabel]("epoch")
	axs[i][:set_ylabel]("test RMSE W=I with no side info")
end


###examine the effect of b and c and epsU, i.e. incorporating side information (W=I),data="RMSEside_finergrid.h5"
minRMSEvec=Array(Float64,maxepoch+burnin,n1,n2,n3)
for i=1:n1,j=1:n2,k=1:n3
	minid=indmin(minimum(RMSEvec[:,i,j,k,:],1))   #index of minimum RMSE, find the optimal epsU
	minRMSEvec[:,i,j,k] = RMSEvec[:,i,j,k,minid]
end
	

###for a=0.25 is optimal###
fig, axs = plt[:subplots](2,2)
for j=1:4
	b=bvec[j]
	for k=1:n3
		c=cvec[k]
		axs[j][:plot](minRMSEvec[:,1,j,k],label="c=$c")
	end
	axs[j][:set_title]("b=$b")
	axs[j][:legend](loc="best")
	axs[j][:set_xlabel]("epoch")
	axs[j][:set_ylabel]("test RMSE")
end
suptitle("RMSE for different b and c, a=0.25, W=I,SGD")

###############################################################################################################hashing##########################################

@everywhere function GPT_side(Rating::Array,UserHash::Array,MovieHash::Array,UserData::Array,MovieData::Array,Ratingtest::Array,r::Real, signal_var::Real, var_u::Real, a::Real, b::Real, c::Real, m::Integer, epsU::Real,burnin::Integer, maxepoch::Integer, param_seed::Integer; langevin::Bool=false)
	N=size(Rating,1);Ntest=size(Ratingtest,1);
	N1,N2=size(UserHash,2),size(MovieHash,2)
	n1,D1=size(UserData); 
	n2,D2=size(MovieData);
	d1,d2=length(find(UserData[1,:])),length(find(MovieData[1,:]))
	numbatches=int(ceil(N/m));
	# initialise U,V
	srand(param_seed);
        U=randn(N1,r)*sqrt(var_u);V=randn(N2,r)*sqrt(var_u)
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
				pred=sum( a*(U[user,:]+b*sum(Us[demo_idx,:],1)) .* (V[movie,:]+c*sum(Vs[genre_idx,:],1)) )
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
		println("epoch=$epoch, testRMSE=$testRMSE")

	end
	return testRMSEvec,final_testpred
end


phiUser = GPTinf.hash_feature(UserData,n,M,a,b1);
	phiMovie = GPTinf.hash_feature(MovieData,n,M,1,b2);
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




