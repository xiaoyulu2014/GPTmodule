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


@everywhere function proj(U::Array,V::Array)
    return V-U*(U'*V+V'*U)/2
end

@everywhere function geod(U::Array,mom::Array,t::Real)
    n,r=size(U)
    A=U'*mom
    temp=[A -mom'*mom;eye(r) A]
    E=expm(t*temp) #can become NaN when temp too large. Return 0 in this case with warning
    if sum(isnan(E))>0
        println("Get NaN when moving along Geodesic. Try smaller epsU") 
        return zeros(n,r)
    else
        mexp=expm(-t*A)
        tmpU=[U mom]*E[:,1:r]*mexp;
        #ensure that tmpU has cols of unit norm
        normconst=Array(Float64,1,r);
        for l=1:r
    	    normconst[1,l]=norm(tmpU[:,l])
        end
        return tmpU./repmat(normconst,n,1)
    end
end
###side info and learn full W###
@everywhere function GPT_fullwside(Rating::Array,UserData::Array,MovieData::Array,Ratingtest::Array,r::Real, signal_var::Real, var_u::Real, var_w::Real, a::Real, b::Real, c::Real, m::Integer,epsw::Real, epsU::Real,burnin::Integer, maxepoch::Integer, param_seed::Integer; langevin::Bool=false,stiefel::Bool=false)
	N=size(Rating,1);Ntest=size(Ratingtest,1);
	n1,D1=size(UserData); 
	n2,D2=size(MovieData);
	d1,d2=length(find(UserData[1,:])),length(find(MovieData[1,:]))
	numbatches=int(ceil(N/m));
	# initialise U,V
	srand(param_seed);

    if stiefel
	Z1=randn(r,n1);	Z2=randn(r,n2)
	U=transpose(\(sqrtm(Z1*Z1'),Z1))
	V=transpose(\(sqrtm(Z2*Z2'),Z2))
    else U=sqrt(var_u)*randn(n1,r);V=sqrt(var_u)*randn(n2,r)
    end

     #   U=randn(n1,r)*sqrt(var_u);V=randn(n2,r)*sqrt(var_u)
	w=var_w*randn(r,r)
	Us=randn(D1,r)*sqrt(var_u);Vs=randn(D2,r)*sqrt(var_u)
	testRMSEvec=zeros(maxepoch+burnin);#trainRMSEvec=zeros(maxepoch+burnin)
	testpred=zeros(Ntest);	trainpred=zeros(Ntrain)
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
			gradw=zeros(r,r)
			gradU=zeros(n1,r);gradUs=zeros(D1,r);
			gradV=zeros(n2,r);gradVs=zeros(D2,r);
			for ii=1:batch_size  
				user=batch_ratings[ii,1]; movie=batch_ratings[ii,2]; rating=batch_ratings[ii,3];
				demo_idx= I[user,:][:];  genre_idx = find(MovieData[movie,:])
				pred=sum( a*(U[user,:]+b*sum(Us[demo_idx,:],1)) *w .* (V[movie,:]+c*sum(Vs[genre_idx,:],1)) )
				tmpU=a*(rating-pred)* (V[movie,:]+c*sum(Vs[genre_idx,:],1)) *w' /signal_var
				tmpV=a*(rating-pred)* (U[user,:]+b*sum(Us[I[user,:][:],:],1)) *w/signal_var
				gradw+=a* (rating-pred)* (U[user,:]+b*sum(Us[I[user,:][:],:],1))'*(V[movie,:]+c*sum(Vs[genre_idx,:],1))/signal_var
				gradU[user,:]+=tmpU
				gradV[movie,:]+=tmpV
				gradUs[demo_idx,:] += b*repmat(tmpU,d1,1)
				gradVs[genre_idx,:] += c*repmat(tmpV,length(genre_idx),1)
			end
			gradw=N/batch_size*gradw-w/var_w;
			if stiefel 
				gradU=N/batch_size*gradU
				gradV=N/batch_size*gradV
			else 
				gradU=N/batch_size*gradU-U/var_u;
				gradV=N/batch_size*gradV-V/var_u;
			end
			gradUs=N/batch_size*gradUs-Us/var_u;
			gradVs=N/batch_size*gradVs-Vs/var_u;

			# update U, V
			w+=epsw*gradw/2; Us+=epsU*gradUs/2; Vs+=epsU*gradVs/2;
			if stiefel
			   momU=proj(U,sqrt(epsU)*gradU/2);	momV=proj(V,sqrt(epsU)*gradV/2)
			   U=geod(U,momU,sqrt(epsU)); V=geod(V,momV,sqrt(epsU));
			else U+=epsU*gradU/2;V+=epsU*gradV/2
			end
		end


		#=for i=1:N
			user=Rating[i,1]; movie=Rating[i,2];
			demo_idx= I[user,:][:];  genre_idx = find(MovieData[movie,:])
			trainpred[i]=(trainpred[i]*counter+ sum( a*(U[user,:]+b*sum(Us[demo_idx,:],1)) *w .* (V[movie,:]+c*sum(Vs[genre_idx,:],1))) )/(counter+1)
		end
		final_trainpred=trainpred*ytrainStd+ytrainMean;
		cutoff!(final_trainpred);
		trainRMSE=sqrt(sum((ytrainStd*Rating[:,3]+ytrainMean-final_trainpred).^2)/N)
		=#
		for i=1:Ntest
			user=Ratingtest[i,1]; movie=Ratingtest[i,2];
			demo_idx= I[user,:][:];  genre_idx = find(MovieData[movie,:])
			testpred[i]=(testpred[i]*counter+ sum( a*(U[user,:]+b*sum(Us[demo_idx,:],1)) *w .* (V[movie,:]+c*sum(Vs[genre_idx,:],1))) )/(counter+1)
		end

		final_testpred=testpred*ytrainStd+ytrainMean;
		cutoff!(final_testpred);
		testRMSE=sqrt(sum((ytrainStd*Ratingtest[:,3]+ytrainMean-final_testpred).^2)/Ntest)
		if epoch > burnin counter+=1 end
		testRMSEvec[epoch]=testRMSE 
		#trainRMSEvec[epoch]=trainRMSE	
		println("epoch=$epoch, testRMSE=$testRMSE") #, trainRMSE=$trainRMSE")

	end
	return testRMSEvec#,trainRMSEvec
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
@everywhere signal_var = 0.8
@everywhere param_seed=17;
@everywhere m = 100;
@everywhere maxepoch = 300;
@everywhere burnin=0;
@everywhere var_u=1/r #0.01
@everywhere var_w=1

@everywhere using Iterators
@everywhere avec = round(linspace(0.1,1.5,5),2)
@everywhere bvec = round(linspace(0,1,5),2)
@everywhere cvec = round(linspace(0,1,5),2)
@everywhere epswvec = 5e-6 #[1.76e-6,5e-6]
@everywhere epsUvec = 2.62e-6 #[2.62e-6,5e-6]
@everywhere n1,n2,n3,n4,n5=length(avec),length(bvec),length(cvec),length(epswvec),length(epsUvec)
@everywhere grid=Iterators.product(avec,bvec,cvec,epswvec,epsUvec)
@everywhere dim=*(n1,n2,n3,n4,n5)
@everywhere mygrid=Array(Any,dim);
@everywhere it=1;
@everywhere for prod in grid
	mygrid[it]=prod;
        it+=1;
        end

RMSEvec=SharedArray(Float64,maxepoch+burnin,dim);
@sync @parallel for i=1:dim
	a,b,c,epsw,epsU = mygrid[i]
	RMSEvec[:,i]=GPT_fullwside(Ratingtrain,UserData,MovieData,Ratingtest,r,signal_var, var_u,var_w, a,b,c,m, epsw,epsU, burnin, maxepoch, param_seed,langevin=false);
	println("a= ",a, "; b= ", b, "; c= ", c, "; epsw=", epsw, "; epsU=", epsU,"; minRMSE=", minimum(RMSEvec[:,i]), ";minepoch = ", indmin(RMSEvec[:,i]) )
end


#=
RMSEvec=reshape(RMSEvec,maxepoch+burnin,n1,n2,n3,n4,n5); 
h5open("RMSEside_W.h5", "w") do file
    write(file, "RMSEvec", convert(Array,RMSEvec),"avec",avec,"bvec",bvec,"cvec",cvec,"epswvec",epswvec,"epsUvec",epsUvec,"maxepoch",maxepoch,"burnin",burnin)
end
=#


#=
RMSEvec,avec,bvec,cvec,epsUvec,maxepoch,burnin = h5open("RMSEside_W.h5", "r") do file
    read(file, "RMSEvec","avec","bvec","cvec","epswvec","epsUvec","maxepoch","burnin")
end
=#

n1,n2,n3,n4,n5=length(avec),length(bvec),length(cvec),length(epswvec),length(epsUvec)
###find the indices when there is no side information
noside_id=[]
for i=1:n5
	noside_id=[noside_id,(i-1)*(*(n1,n2,n3,n4))+1:(i-1)*(*(n1,n2,n3,n4))+n1]
end

###examine the effect of a and epsU for no side info (W=I), data="RMSEside.h5"
RMSE_noside=Array(Float64,maxepoch+burnin,n1,n4,n5)
for i=1:n1,j=1:n4,k=1:n5                                                                             
       a=avec[i];epsw=epswvec[j];epsU=epsUvec[k];  
       RMSE_noside[:,i,j,k] = RMSEvec[:,i,1,1,j,k]                                                                    
       plot(RMSE_noside[:,i,j,k],label="a=$a,epsw=$epsw,epsU=$epsU")                                                 
end  

minRMSE_a=Array(Float64,n1)
fig, axs = plt[:subplots](2,3)
for i=1:n1
	a=avec[i]
	for j=1:n4,k=1:n5
		epsw=epswvec[j];epsU=epsUvec[k]
		axs[i][:plot](RMSE_noside[:,i,j,k],label="epsw=$epsw,epsU=$epsU")
	end
	axs[i][:set_title]("a=$a")
	axs[i][:legend](loc="best")
	axs[i][:set_xlabel]("epoch")
	axs[i][:set_ylabel]("test RMSE full W with no side info")
	minRMSE_a[i] = minimum(RMSE_noside[:,i,j,k])
end
axs[6][:plot](avec,minRMSE_a)
axs[6][:set_xlabel]("a")
axs[6][:set_ylabel]("RMSE")
axs[6][:set_title]("minimum RMSE vs a")
tmp=round(minimum(minRMSE_a),3)
annotate("min RMSE = $tmp",
    xy=[avec[indmin(minRMSE_a)]-0.5;tmp],
    xycoords="data", # Coordinates in in "data" units
)

suptitle("RMSE for different hyperparameters, full W, no side infomation,SGD")


###examine the effect of b and c and epsU
minRMSEvec=Array(Float64,maxepoch+burnin,n1,n2,n3)
for i=1:n1,j=1:n2,k=1:n3
	minid=indmin(minimum(RMSEvec[:,i,j,k,:,:],1))   #index of minimum RMSE, find the optimal epsU
	minRMSEvec[:,i,j,k] = RMSEvec[:,i,j,k,minid]
end
	

###for a=0.8 is optimal###
fig, axs = plt[:subplots](2,2)
for j=1:4
	b=bvec[j]
	for k=1:n3
		c=cvec[k]
		axs[j][:plot](minRMSEvec[:,3,j,k],label="c=$c")
	end
	axs[j][:set_title]("b=$b")
	axs[j][:legend](loc="best")
	axs[j][:set_xlabel]("epoch")
	axs[j][:set_ylabel]("test RMSE")
	axs[j][:set_ylim]([0.89,1.12])
end
suptitle("RMSE for different b and c, a=0.8, full W,SGD")
tmp=round(minimum(minRMSEvec),3)
axs[2][:annotate]("min RMSE = $tmp",xy=[100;0.9])






