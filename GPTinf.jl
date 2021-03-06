module GPTinf

using Distributions,Optim,ForwardDiff

export datawhitening,feature,samplenz,RMSE, GPTgibbs, pred, data_simulator, GPTHMC, GPT_w, GPT_SGLDERM_hyper,GPTregression,GPT_probit_SGD, GPT_SGLDERM_probit,GPNT_hyperparameters,featureNotensor,GPT_SGLDERM_probit_SEM,hash_feature,GPT_probit_SGD_hyper,GPT_probit_SGD_adam,pred_CF,GPT_CF,GPT_CFgibbs


function data_simulator(X::Array,n::Integer,r::Integer,Q::Integer,sigma::Real,length_scale::Real,sigma_RBF::Real,seed::Integer)
  N,D = size(X)
  w = randn(Q)
  scale=sqrt(n/(Q^(1/D)))  
  U=Array(Float64,n,r,D)
  for k=1:D
    Z=randn(r,n)
    U[:,:,k]=transpose(\(sqrtm(Z*Z'),Z)) 
  end
  I=samplenz(r,D,Q,seed)
  phi = feature(X,r,length_scale,sigma_RBF,seed,scale)
  temp=phidotU(U,phi)
  V=computeV(temp,I)
  return computefhat(V,w) + sigma*randn(N)
end


#extract features from tensor decomp of each row of X
function feature(X::Array,length_scale, sigma_RBF::Real,phi_scale::Real,Z::Array,b::Array)    
    N,D=size(X);
    n=size(Z,1);
    phi=Array(Float64,n,D,N)
    Zt=scale(Z,1./length_scale)
    for i=1:N
	for k=1:D
	    for j=1:n
		phi[j,k,i]=cos(X[i,k]*Zt[j,k]+b[j,k])
	    end
	end
    end
    return phi_scale*(sigma_RBF)^(1/D)*sqrt(2/n)*phi
end

#extract features from tensor decomp of each row of X
function feature_tilde(X::Array,length_scale::Array,sigma_RBF::Real,phi_scale::Integer,Z::Array,b::Array)    
    N,D=size(X)
    n=size(Z,1);
    phi=Array(Float64,n,D,N)
    for i=1:N
	for k=1:D
	    for j=1:n
		phi[j,k,i]=sin(X[i,k]*Z[j,k]/length_scale[k]+b[j,k]) * X[i,k]*Z[j,k]/(length_scale[k]^2)
	    end
	end
    end
    return phi_scale*(sigma_RBF)^(1/D)*sqrt(2/n)*phi
end

function hash_feature(X::Array,n::Integer,M::Integer,a::Real,b::Real)
    #n:number of hash features
    N,D = size(X)
    phi = zeros(n+D,N)
    for i=1:N
	idx = randperm(n)[1:M]
	phi[idx,i]= 2.*(rand(Bernoulli(),M) - 0.5)/M
	phi[:,i] = a*[phi[1:n,i], b*X[i,:]']
    end
    return phi
end


# sample the Q random non-zero locations of w
function samplenz(r::Integer,D::Integer,Q::Integer,seed::Integer)
    srand(seed)
    L=sample(0:(r^D-1),Q,replace=false)
    I=Array(Int32,Q,D)
    for q in 1:Q
        I[q,:]=digits(L[q],r,D)+1
    end
    # this way the locations are drawn uniformly from the lattice [r^D] without replacement
    # so I_qd=index of dth dim of qth non-zero
    return I
end

function proj(U::Array,V::Array)
    return V-U*(U'*V+V'*U)/2
end

function geod(U::Array,mom::Array,t::Real)
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

# define geod for Stiefel manifold - want both endpt and mom
function geodboth(U::Array,mom::Array,t::Real)
    n,r=size(U)
    A=U'*mom
    temp=[A -mom'*mom;eye(r) A]
    E=expm(t*temp) #can become NaN when temp too large. Return 0 in this case
    if sum(isnan(E))>0
        println("Get NaN when moving along Geodesic. Try smaller epsU")
        return zeros(n,r),zeros(n,r)
    else
        mexp=expm(-t*A)
        tmpU=[U mom]*E[:,1:r]*mexp;
        tmpV=[U mom]*E[:,(r+1):2r]*mexp;
        #ensure that tmpU has cols of unit norm
        normconst=Array(Float64,1,r);
        for l=1:r
    	    normconst[1,l]=norm(tmpU[:,l])
        end
        return tmpU./repmat(normconst,n,1),tmpV
    end
end

# centre and normalise data X so that each col has sd=1
function datawhitening(X::Array) 
    for i = 1:size(X,2)   
        X[:,i] = (X[:,i] - mean(X[:,i]))/std(X[:,i])   
    end
    return X
end
#compute <phi^(k)(x_i),U^(k)_{.l}> for all k,l,batch

function phidotU(U::Array,phi::Array)
    n,D,data_size=size(phi)
    r=size(U,2)
    temp=Array(Float64,D,r,data_size)
    for i=1:data_size
        for l=1:r
            for k=1:D
                temp[k,l,i]=dot(phi[:,k,i],U[:,l,k])
            end
        end
    end
    return temp
end

#compute V st V[q,i]=prod_{k=1 to D}(temp[k,I[q,k],i])

function computeV(temp::Array,I::Array)
    Q,D=size(I);
    data_size=size(temp,3)
    V=ones(Q,data_size)
    for i=1:data_size
        for q=1:Q
            for k=1:D
                V[q,i]*=temp[k,I[q,k],i];
            end
        end
    end
    return V
end

function computeDV(phi::Array,phi_tilde::Array,U::Array,I::Array)
    Q,D=size(I);
    data_size=size(phi,3)
    V=zeros(Q,data_size)
    for i=1:data_size
        for q=1:Q
            for k=1:D
                V[q,i] += dot(phi_tilde[:,k,i],U[:,I[q,k],k]) / dot(phi[:,k,i],U[:,I[q,k],k])
            end
        end
    end
    return V
end

#compute predictions fhat from V,w

function computefhat(V::Array,w::Array)
    data_size=size(V,2)
    fhat=Array(Float64,data_size)
    for i=1:data_size
        fhat[i]=dot(V[:,i],w)
    end
    return fhat
end


#compute predictions from w,U,I

function pred(w::Array,U::Array,I::Array,phitest::Array)

    # compute <phi^(k)(x_i),U^(k)_{.l}> for all k,l,test and store in temp
    temp=phidotU(U,phitest)

    # compute V st V[q,i]=prod_{k=1 to D}(temp[k,I[q,k],i])
    V=computeV(temp,I)

    # compute fhat where fhat[i]=V[:,i]'w
    return computefhat(V,w)
end

function computeU_phi(V::Array,temp::Array,I::Array)
    Q,D=size(I)
    data_size=size(V,2)
    U_phi=Array(Float64,Q,data_size,D)
    for k=1:D
        for i=1:data_size
            for q=1:Q
	        U_phi[q,i,k]=V[q,i]/temp[k,I[q,k],i]
            end
	end
    end
    return U_phi
end


function computeA(U_phi::Array,w::Array,I::Array,r::Integer)
    Q,data_size,D=size(U_phi)
    A=zeros(r,D,data_size)
    for i=1:data_size
        for k=1:D
            for l in unique(I[:,k])
                index=findin(I[:,k],l) 
                A[l,k,i]=dot(U_phi[index,i,k],w[index]) 
            end
        end
    end
    return A
end

function computePsi(A,phi)
    r,D,data_size=size(A)
    n,D,data_size=size(phi)
    Psi=Array(Float64,n*r,data_size,D)
    for i=1:data_size
        for k=1:D
            Psi[:,i,k]=kron(A[:,k,i],phi[:,k,i])
        end
    end
    return Psi
end

#work out minimum RMSE by averaging over predictions, starting from last prediction
function RMSE(w_store::Array,U_store::Array,I::Array,phitest::Array,ytest::Array)
    Ntest=length(ytest);
    T=size(w_store,2);

    meanfhat= @parallel (+) for i=1:T
        pred(w_store[:,i],U_store[:,:,:,i],I,phitest);
    end
    meanfhat=meanfhat/T;
    return norm(ytest-meanfhat)/sqrt(Ntest);

   #out = Array(Float64,T,1);
   #= for i=1:T
        tmp = pred(w_store[:,i],U_store[:,:,:,i],I,phitest)
        out[i] = norm(ytest-tmp)/sqrt(Ntest)
    end=#
end


function GPTgibbs(phi::Array,y::Array,sigma::Real,I::Array,r::Integer,Q::Integer,burnin::Integer,numiter::Integer)
    # phi is the D by n by N array of features where phi[k,:,i]=phi^(k)(x_i)
    # sigma is the s.d. of the observed values
    # sigma_w is the s.d. for the Guassian prior on w
    # epsw,epsU are the epsilons for w and U resp.
    # maxepoch is the number of sweeps through whole dataset

    n,D,N=size(phi)
    sigma_u = 0.1 #sqrt(1/r)
    sigma_w = 1 #sqrt(r^D/Q)
    # initialise w,U^(k)
    w_store=Array(Float64,Q,numiter)
    U_store=Array(Float64,n,r,D,numiter)
    w=ones(Q) #sigma_w*randn(Q)
    #println("w= ",w)
    U=Array(Float64,n,r,D)
    for k=1:D
        U[:,:,k]=sigma_u*randn(n,r)
    end


    for epoch=1:(burnin+numiter)

        # compute <phi^(k)(x_i),U^(k)_{.l}> for all k,l,batch and store in temp
       # temp=phidotU(U,phi)

        # compute V st V[q,i]=prod_{k=1 to D}(temp[k,I[q,k],i])
       # V=computeV(temp,I)

        #gibbs on w
      #  invSigma_w = 1/(sigma^2) * V * V' + (1/sigma_w^2)*eye(Q)
      #  Mu_w = \(invSigma_w,1/(sigma^2) *(V * y))
      #  w[:] = \(chol(invSigma_w,:U),randn(Q)) + Mu_w

        # compute U_phi[q,i,k]=expression in big brackets in (11)

        for k in 1:D

            # compute <phi^(k)(x_i),U^(k)_{.l}> for all k,l,batch and store in temp
            temp=phidotU(U,phi)

            # compute V st V[q,i]=prod_{k=1 to D}(temp[k,I[q,k],i])
            V=computeV(temp,I)

            # compute U_phi[q,i,k]=expression in big brackets in (11)
            U_phi=computeU_phi(V,temp,I)
            
            # compute a_l^(k)(x_i) for l=1,...,r k=1,.	.,D and store in A
            A=computeA(U_phi,w,I,r)
            
            # compute Psi as in (12)
            Psi=computePsi(A,phi)

            invSigma_U = Psi[:,:,k] * Psi[:,:,k]'/(sigma^2) + (1/sigma_u)^2 * eye(n*r)
            Mu_U = \(invSigma_U, (Psi[:,:,k] * y) / (sigma^2))
            U[:,:,k]= reshape(\(chol(invSigma_U,:U),randn(n*r)) + Mu_U,n,r)
        end

        if epoch>burnin
            w_store[:,epoch-burnin]= w
            U_store[:,:,:,epoch-burnin]=U
        end
    end
    return w_store,U_store
end

#GPTgibbs without learning U
function GPT_w(phi::Array,y::Array,sigma::Real,I::Array,r::Integer,Q::Integer)

    n,D,N=size(phi)
    sigma_u = sqrt(1/r)
    sigma_w = sqrt(r^D/Q)
    # initialise w,U^(k)
    w=sigma_w*randn(Q)
    #println("w= ",w)
    U=Array(Float64,n,r,D)
    for k=1:D
        U[:,:,k]=sigma_u*randn(n,r)
    end

    # compute <phi^(k)(x_i),U^(k)_{.l}> for all k,l,batch and store in temp
    temp=phidotU(U,phi)

    # compute V st V[q,i]=prod_{k=1 to D}(temp[k,I[q,k],i])
    V=computeV(temp,I)

    #gibbs on w
    invSigma_w = 1/(sigma^2) * V * V' + (1/sigma_w^2)*eye(Q)
    Mu_w = \(invSigma_w,1/(sigma^2) *(V * y))
    w[:] = \(chol(invSigma_w,:U),randn(Q)) + Mu_w

    return w,U
end



#HMC on Tucker Model 
function GPTHMC(phi::Array, y::Array, sigma::Real, I::Array, r::Integer, Q::Integer, epsw::Real, epsU::Real, burnin::Integer, numiter::Integer, L::Integer,param_seed::Integer)
    
    n,D,N=size(phi)
    sigma_u = sqrt(1/r)
    sigma_w = sqrt(r^D/Q)
	
    
    # initialise w,U^(k)
    w_store=Array(Float64,Q,numiter)
    U_store=Array(Float64,n,r,D,numiter)
    accept_prob=Array(Float64,numiter+burnin)
    srand(param_seed)
    w=sigma_w*randn(Q)

    U=Array(Float64,n,r,D)
    for k=1:D
        U[:,:,k]=sigma_u*randn(n,r)
    end

    for iter=1:(burnin+numiter)
        w_old=w; U_old=U;
        # initialise momentum terms and Hamiltonian
        p=randn(Q); mom=Array(Float64,n,r,D);
        for k=1:D
            mom[:,:,k]= randn(n,r)   
        end

        H_old=-sum(w.*w)/(2*sigma_w^2)-sum(U.*U)/(2*sigma_u^2)-norm(y-pred(w,U,I,phi))^2/(2*sigma^2)-sum(mom.*mom)/2-sum(p.*p)/2;
        #println("H=",H_old)
        pred_new=Array(Float64,N); # used later for computing H_new
	#half leapfrog step
        temp=phidotU(U,phi)
	V=computeV(temp,I)
	fhat=computefhat(V,w)
	gradw=V*(y-fhat)/(sigma^2)-w/(sigma_w^2)
	p+=epsw*gradw/2
	U_phi=computeU_phi(V,temp,I)
	A=computeA(U_phi,w,I,r)
	Psi=computePsi(A,phi)
	gradU=Array(Float64,n,r,D)
            for k=1:D
                gradU[:,:,k]=reshape(Psi[:,:,k]*(y-fhat)/(sigma^2),n,r)-U[:,:,k]/(sigma_u^2)
		mom[:,:,k]+=epsU*gradU[:,:,k]/2
            end

        # leapfrog step
        for l=1:L

	    ### update w,U,mom 
	    # update w, U
	    w+=epsw*p
		for k=1:D
		U[:,:,k]+=epsU*mom[:,:,k]
	    end

	    ### update p and mom
	    # compute <phi^(k)(x_i),U^(k)_{.l}> for all k,l,i and store in temp
	    temp=phidotU(U,phi)
	    # compute V st V[q,i]=prod_{k=1 to D}(temp[k,I[q,k],i])
	    V=computeV(temp,I)
	    # compute fhat where fhat[i]=V[:,i]'w
	    fhat=computefhat(V,w)
	    # now can compute gradw, the stochastic gradient of log post wrt w
	    gradw=V*(y-fhat)/(sigma^2)-w/(sigma_w^2)
	    # update p
	    p+=epsw*gradw;
	    # compute U_phi[q,i,k]=expression in big brackets in (11)
	    U_phi=computeU_phi(V,temp,I)
	    # compute a_l^(k)(x_i) for l=1,...,r k=1,..,D and store in A
	    A=computeA(U_phi,w,I,r)
	    # compute Psi as in (12)
	    Psi=computePsi(A,phi)
	    gradU=Array(Float64,n,r,D)
	    for k=1:D
		gradU[:,:,k]=reshape(Psi[:,:,k]*(y-fhat)/(sigma^2),n,r)-U[:,:,k]/(sigma_u^2)
	        mom[:,:,k]+=epsU*gradU[:,:,k]
	    end
	end
	#another half leapfrog step
	temp=phidotU(U,phi)
	V=computeV(temp,I)
	fhat=computefhat(V,w)
	gradw=V*(y-fhat)/(sigma^2)-w/(sigma_w^2)
	println("mean gradw=",round(mean(gradw),3),"mean p=",round(mean(p),3))
	p+=epsw*gradw/2
	U_phi=computeU_phi(V,temp,I)
	A=computeA(U_phi,w,I,r)
	Psi=computePsi(A,phi)
	gradU=Array(Float64,n,r,D)
        for k=1:D
              gradU[:,:,k]=reshape(Psi[:,:,k]*(y-fhat)/(sigma^2),n,r)-U[:,:,k]/(sigma_u^2)
	      mom[:,:,k]+=epsU*gradU[:,:,k]/2
        end
	println("mean gradU=",round(mean(gradU),3),"mean mom=",round(mean(mom),3))

        pred_new=fhat;
        H=-sum(w.*w)/(2*sigma_w^2)-sum(U.*U)/(2*sigma_u^2)-norm(y-pred_new)^2/(2*sigma^2)-sum(mom.*mom)/2-sum(p.*p)/2;
        u=rand(1);
        
        accept_prob[iter]=exp(H-H_old);
        println("accept_prob=",round(accept_prob[iter],3))
        
        if u[1]>accept_prob[iter] #if true, reject 
            w=w_old; U=U_old;
        end
        
	if iter>burnin
	    w_store[:,iter-burnin]=w
	    U_store[:,:,:,iter-burnin]=U
        end
    end
    return w_store,U_store,accept_prob
end

#SGLD on Tucker Model with Stiefel Manifold, Probit likelihood
function GPT_SGLDERM_probit(phi::Array, y::Array, I::Array, r::Integer, Q::Integer, m::Integer, epsw::Real, epsU::Real, burnin::Integer, maxepoch::Integer, steifel::Bool)
    # phi is the D by n by N array of features where phi[k,:,i]=phi^(k)(x_i)
    # sigma is the s.d. of the observed values
    # epsw,epsU are the epsilons for w and U resp.
    # maxepoch is the number of sweeps through whole dataset
    
    n,D,N=size(phi)
    numbatches=int(ceil(N/m))
    sigma_u = sqrt(1/r)

    # initialise w,U^(k)
    w_store=Array(Float64,Q,maxepoch*numbatches)
    U_store=Array(Float64,n,r,D,maxepoch*numbatches)


    U=Array(Float64,n,r,D)
    if steifel
	    sigma_w=1;    
	    for k=1:D
		Z=randn(r,n)
		U[:,:,k]=transpose(\(sqrtm(Z*Z'),Z)) #sample uniformly from V_{n,r}
	    end
    else    sigma_w = sqrt(r^D/Q) 
	for k=1:D
		U[:,:,k]=sigma_u*randn(n,r)
	    end
    end 

    w=sigma_w*randn(Q)

    for epoch=1:(burnin+maxepoch)
        #randomly permute training data and divide into mini_batches of size m
        perm=randperm(N)
        phi=phi[:,:,perm]; y=y[perm];
        
        # run SGLD on w and SGLDERM on U
        for batch=1:numbatches
            # random samples for the stochastic gradient
            idx=(m*(batch-1)+1):min(m*batch,N)
            phi_batch=phi[:,:,idx]; y_batch=y[idx];
            batch_size=length(idx) #this is m except for last batch
            # compute <phi^(k)(x_i),U^(k)_{.l}> for all k,l,batch and store in temp
            temp=phidotU(U,phi_batch)

	    # compute V st V[q,i]=prod_{k=1 to D}(temp[k,I[q,k],i])
            V=computeV(temp,I)
	    
            # compute fhat where fhat[i]=V[:,i]'w
            fhat=computefhat(V,w)

            # now can compute gradw, the stochastic gradient of log post wrt w
            tmp = cdf(Normal(),fhat)
            gradw = (N/batch_size)*V*( pdf(Normal(),fhat) .* (y_batch./tmp - (1-y_batch)./(1-tmp)) ) - w/(sigma_w^2)

            # compute U_phi[q,i,k]=expression in big brackets in (11)
            U_phi=computeU_phi(V,temp,I)
            
            # compute a_l^(k)(x_i) for l=1,...,r k=1,..,D and store in A
            A=computeA(U_phi,w,I,r)
            
            # compute Psi as in (12)
            Psi=computePsi(A,phi_batch)

            # SGLD step on w
            w[:]+=epsw*gradw/2 +sqrt(epsw)*randn(Q)

            gradU=Array(Float64,n,r,D)
            if steifel
		 for k=1:D
			gradU[:,:,k]=reshape((N/batch_size)*Psi[:,:,k]*( pdf(Normal(),fhat) .* (y_batch./tmp - (1-y_batch)./(1-tmp)) ),n,r)
		        mom=proj(U[:,:,k],sqrt(epsU)*gradU[:,:,k]/2+randn(n,r))
		        U[:,:,k]=geod(U[:,:,k],mom,sqrt(epsU));
		        if U[:,:,k]==zeros(n,r) #if NaN appears while evaluating G
		            return zeros(Q,maxepoch*numbatches),zeros(n,r,D,maxepoch*numbatches)
		        end
		    end
	     else 
 		    for k=1:D
			gradU[:,:,k]=reshape((N/batch_size)*Psi[:,:,k]*( pdf(Normal(),fhat) .* (y_batch./tmp - (1-y_batch)./(1-tmp)) ),n,r)- U[:,:,k]/(sigma_u^2)
		        U[:,:,k] += epsU*gradU[:,:,k]/2+sqrt(epsU)*randn(n,r)
		    end
	    end

	    if epoch>burnin
	        w_store[:,((epoch-burnin)-1)*numbatches+batch]=w
	        U_store[:,:,:,((epoch-burnin)-1)*numbatches+batch]=U
	    end
        end
    end
    return w_store,U_store
end

#SGLD on Tucker Model with Stiefel Manifold, Probit likelihood, learn hyperparameters by SEM
function GPT_SGLDERM_probit_SEM(X::Array, y::Array, I::Array, n::Real, r::Integer, Q::Integer, m::Integer, epsw::Real, epsU::Real, burnin::Integer, maxepoch::Integer, hyp_init::Array, Zmat::Array, bmat::Array)
   
    N,D=size(X)
    numbatches=int(ceil(N/m))
    sigma_w=1;
    length_scale,sigma_RBF = hyp_init
    scale=sqrt(n/(Q^(1/D)))  


    # initialise w,U^(k)
    w_store=Array(Float64,Q,maxepoch*numbatches)
    U_store=Array(Float64,n,r,D,maxepoch*numbatches)
    l_store=Array(Float64,maxepoch*numbatches)
    SigmaRBF_store=Array(Float64,maxepoch*numbatches)
    w=sigma_w*randn(Q)
    
    U=Array(Float64,n,r,D)
    for k=1:D
        Z=randn(r,n)
        U[:,:,k]=transpose(\(sqrtm(Z*Z'),Z)) #sample uniformly from V_{n,r}
    end


    for epoch=1:(burnin+maxepoch)
        #randomly permute training data and divide into mini_batches of size m
        perm=randperm(N)
        Xperm=X[perm,:]; y=y[perm];
        
        # run SGLD on w and SGLDERM on U
        for batch=1:numbatches
	    t = (epoch-1)*numbatches+batch
            # random samples for the stochastic gradient
            idx=(m*(batch-1)+1):min(m*batch,N)
            phi_batch=feature1(Xperm[idx,:],n,length_scale,sigma_RBF,scale,Zmat,bmat); 
            phi_tilde_batch=feature_tilde1(Xperm[idx,:],n,length_scale,sigma_RBF,scale,Zmat,bmat);
            y_batch=y[idx];
            batch_size=length(idx) #this is m except for last batch

            # compute <phi^(k)(x_i),U^(k)_{.l}> for all k,l,batch and store in temp
            temp=phidotU(U,phi_batch)

	    # compute V st V[q,i]=prod_{k=1 to D}(temp[k,I[q,k],i])
            V=computeV(temp,I)
	    
            # compute fhat where fhat[i]=V[:,i]'w
            fhat=computefhat(V,w)

            # now can compute gradw, the stochastic gradient of log post wrt w
            tmp = cdf(Normal(),fhat)
	    tmp1 = pdf(Normal(),fhat) .* (y_batch./tmp - (1-y_batch)./(1-tmp))
            gradw = (N/batch_size)*V*tmp1  - w/(sigma_w^2)

            # compute U_phi[q,i,k]=expression in big brackets in (11)
            U_phi=computeU_phi(V,temp,I)
            
            # compute a_l^(k)(x_i) for l=1,...,r k=1,..,D and store in A
            A=computeA(U_phi,w,I,r)
            
            # compute Psi as in (12)
            Psi=computePsi(A,phi_batch)
            
            # can now compute gradU where gradU[:,:,k]=stochastic gradient of log post wrt U^(k)
            gradU=Array(Float64,n,r,D)
            for k=1:D
                gradU[:,:,k]=reshape((N/batch_size)*Psi[:,:,k]*tmp1 ,n,r)
            end

  	    ## SEM on length scale and sigma_RBF
            # SGLD step on w
            w[:]+=epsw*gradw/2 +sqrt(epsw)*randn(Q)

            for k=1:D
                mom=proj(U[:,:,k],sqrt(epsU)*gradU[:,:,k]/2+randn(n,r))
                U[:,:,k]=geod(U[:,:,k],mom,sqrt(epsU));
                if U[:,:,k]==zeros(n,r) #if NaN appears while evaluating G
                    return zeros(Q,maxepoch*numbatches),zeros(n,r,D,maxepoch*numbatches)
                end
            end

	    theta = log(sigma_RBF);
	    DV_l = computeDV(phi_batch,phi_tilde_batch,U,I) .* V
	    gradl = (N/batch_size)*((w'*DV_l) * tmp1)[1]
          #  DV_s = V/sigma_RBF;#computeV(phidotU(U,feature(Xperm[idx,:],n,length_scale,sigma_RBF,seed,scale/sigma_RBF^(1/D))),I)
	  #  gradSrbf = (N/batch_size)*((w'* DV_s) * tmp1)[1]



 	    gradtheta =(N/batch_size)*((w'* V) * tmp1)[1] 
	    theta += 0.001*gradtheta
	    sigma_RBF = exp(theta)


            length_scale += 0.001*gradl
          #  sigma_RBF += 0.001*1/(1+t/10)*gradSrbf 
	  #  println("gradl = ", gradl, "; gradSrbf=", gradSrbf, "; update l=", 0.001*1/(1+t/10)*gradl, "; update s=", 0.001*1/(1+t/10)*gradSrbf )

            #=
	    theta = [log(length_scale),log(sigma_RBF)];
	    function lik_theta(theta::Vector)
			length_scale, sigma_RBF = exp(theta);
			phi_batch = feature(Xperm[idx,:],n,length_scale,sigma_RBF,seed,scale);
			temp = phidotU(U,phi_batch);
			V = computeV(temp,I)
			fhat=computefhat(V,w)
           		 tmp = cdf(Normal(),fhat)
            		return(-sum(y_batch.*log(tmp) + (1-y_batch).*log(1-tmp)))
	    end
		 g=ForwardDiff.gradient(lik_theta)
   		 function g!(theta::Vector,storage::Vector)
			grad=g(theta)
			for i=1:length(theta)
			    storage[i]=grad[i]
			end
		    end

		d = DifferentiableFunction(lik_theta, g!)
		theta = optimize(lik_theta,g!,theta,method=:cg).minimum
	    length_scale,sigma_RBF = exp(theta)
	    println("theta = ", theta)
	   =#
	    if epoch>burnin
	        w_store[:,((epoch-burnin)-1)*numbatches+batch]=w
	        U_store[:,:,:,((epoch-burnin)-1)*numbatches+batch]=U
		SigmaRBF_store[((epoch-burnin)-1)*numbatches+batch]=sigma_RBF
		l_store[((epoch-burnin)-1)*numbatches+batch]=length_scale
	    end
        end
    end
    return w_store,U_store,l_store,SigmaRBF_store
end


###
#SGLD on Tucker Model with Stiefel Manifold, Probit likelihood
function GPT_probit_SGD(phi::Array, y::Array, m::Integer, eps::Real, burnin::Integer, maxepoch::Integer)
   
    n,N=size(phi)
    numbatches=int(ceil(N/m))
    sigma=1;

    # initialise w,U^(k)
    theta_store=Array(Float64,n,maxepoch*numbatches)
    theta=sigma*randn(n)

    for epoch=1:(burnin+maxepoch)
        #randomly permute training data and divide into mini_batches of size m
        perm=randperm(N)
        phi=phi[:,perm]; y=y[perm];
        # run SGLD on w and SGLDERM on U
        for batch=1:numbatches
            # random samples for the stochastic gradient
            idx=(m*(batch-1)+1):min(m*batch,N)
            phi_batch=phi[:,idx]; y_batch=y[idx];
            batch_size=length(idx) 
    
            # compute fhat where fhat[i]=V[:,i]'w
            fhat= phi_batch'*theta

            # now can compute gradw, the stochastic gradient of log post wrt w
            tmp = cdf(Normal(),fhat)
            grad = (N/batch_size)*phi_batch*( pdf(Normal(),fhat) .* (y_batch./tmp - (1-y_batch)./(1-tmp)) ) - theta/(sigma^2)
	    
            # SGLD step on w
            theta[:]+=eps*grad/2
 	   # eps = eps*0.5
	    if epoch>burnin
	        theta_store[:,((epoch-burnin)-1)*numbatches+batch]=theta
	    end
        end
    end
    return theta_store
end



#SGLD on Tucker Model with Stiefel Manifold, Probit likelihood, learn hyperparameters by SEM
function GPT_probit_SGD_adam(X::Array, y::Array, I::Array, n::Real, m::Integer, alpha0::Real, burnin::Integer, maxepoch::Integer, hyp_init::Array, seed::Int)
   
    N,D=size(X)
    numbatches=int(ceil(N/m))
    sigma=1;
    length_scale,sigma_RBF = hyp_init
    beta1=0.9;beta2=0.999;eps=0.00000001
    # initialise w,U^(k)
    theta_store=Array(Float64,n,maxepoch*numbatches)
    l_store=Array(Float64,maxepoch*numbatches)
    SigmaRBF_store=Array(Float64,maxepoch*numbatches)
    theta=sigma*randn(n)
    mom = zeros(n+2); mom2 = zeros(n+2);
    log_hyp = [log(length_scale),log(sigma_RBF)]
    for epoch=1:(burnin+maxepoch)
        #randomly permute training data and divide into mini_batches of size m
        perm=randperm(N)
        Xperm=X[perm,:]; y=y[perm];
        
        # run SGLD on w and SGLDERM on U
        for batch=1:numbatches
	    t = (epoch-1)*numbatches+batch
            # random samples for the stochastic gradient
            idx=(m*(batch-1)+1):min(m*batch,N)
            phi_batch=featureNotensor(Xperm[idx,:],n,length_scale,sigma_RBF,seed);
            phi_tilde_batch=featureNotensor_tilde(Xperm[idx,:],n,length_scale,sigma_RBF,seed);
            y_batch=y[idx];
            batch_size=length(idx) #this is m except for last batch

             # compute fhat where fhat[i]=V[:,i]'w
            fhat= phi_batch'*theta

            # now can compute gradw, the stochastic gradient of log post wrt w
            tmp = cdf(Normal(),fhat)
	    tmp1 =  pdf(Normal(),fhat) .* (y_batch./tmp - (1-y_batch)./(1-tmp)) 
            grad = (N/batch_size)*phi_batch* tmp1  - theta/(sigma^2)
	    grad = [grad, (N/batch_size)*dot((phi_tilde_batch'*theta),tmp1)*exp(-length_scale)]  #grad wrt exp(l)
	    grad = [grad, (N/batch_size)*dot((phi_batch'*theta),tmp1)/sigma_RBF*exp(-sigma_RBF)]

	    mom = beta1*mom + (1-beta1)*grad
	    mom2 = beta2*mom2 + (1-beta2)*grad.^2
	    alpha = alpha0*sqrt(1-beta2^t)/(1-beta1^t)

	    tmp2 = alpha*mom./(sqrt(mom2)+eps)
            theta[:] += tmp2[1:n]
	    log_hyp += tmp2[n+1:n+2]
	   # length_scale += tmp2[n+1]; sigma_RBF += tmp2[n+2]
	     length_scale,sigma_RBF = exp(log_hyp)
	    # length_scale += eps*gradl; sigma_RBF += eps*grads
	    println("update_l = ", tmp2[n+1])
	    if epoch>burnin
	        theta_store[:,((epoch-burnin)-1)*numbatches+batch]=theta
		SigmaRBF_store[((epoch-burnin)-1)*numbatches+batch]=sigma_RBF
		l_store[((epoch-burnin)-1)*numbatches+batch]=length_scale
	    end
        end
    end
    return theta_store,l_store,SigmaRBF_store
end

# function to return the negative log marginal likelihood of No Tensor model with Gaussian likelihood
# use the Z and b that was used to compute features
function GPNT_logmarginal(X::Array,y::Array,length_scale,sigma_RBF::Real,signal_var::Real,Z::Array, b::Array)
    N,D=size(X);
    phi=featureNotensor(X,length_scale,sigma_RBF,Z,b);
    A=phi*phi'+signal_var*eye(n);
    b=phi*y;
	B=\(A,b);
	lambda=eigvals(A);
	logdetA=sum(log(lambda));
    return (N-n)*log(signal_var)/2+logdetA/2+(sum(y.*y)-sum(b.*B))/(2*signal_var)
end

# learning hyperparams signal_var,sigma_RBF,length_scale for No Tensor Model by optimising Gaussian marginal likelihood for fixed length_scale
function GPNT_hyperparameters(X::Array,y::Array,n::Integer,init_length_scale::Real,init_sigma_RBF::Real,init_signal_var::Real,Z::Array, b::Array)
	D=size(X,2);
    logmarginal(hyperparams::Vector)=GPNT_logmarginal(X,y,n,exp(hyperparams[1]),exp(hyperparams[2]),exp(hyperparams[3]),Z,b); # log marginal likelihood as a fn of hyperparams=log([length_scale,sigma_RBF,signal_var]) only.
    # exp needed to enable unconstrained optimisation, since length_scale,sigmaRBF,signal_var must be positive
    g=ForwardDiff.gradient(logmarginal)
    function g!(hyperparams::Vector,storage::Vector)
        grad=g(hyperparams)
        for i=1:length(hyperparams)
            storage[i]=grad[i]
        end
    end
    l=optimize(logmarginal,g!,log([init_length_scale,init_sigma_RBF,init_signal_var]),method=:cg,show_trace = true, extended_trace = true)
	return exp(l.minimum)
end

#learning hyperparams signal_var,sigma_RBF,length_scale for No Tensor Model by optimising Gaussian marginal likelihood for varying length_scale
function GPNT_hyperparameters(X::Array,y::Array,n::Integer,init_length_scale::Vector,init_sigma_RBF::Real,init_signal_var::Real,Z::Array, b::Array)
	D=size(X,2);
    logmarginal(hyperparams::Vector)=GPNT_logmarginal(X,y,n,exp(hyperparams[1:D]),exp(hyperparams[D+1]),exp(hyperparams[D+2]),Z::Array, b::Array); # log marginal likelihood as a fn of hyperparams=log([length_scale,sigma_RBF,signal_var]) only.
    # exp needed to enable unconstrained optimisation, since length_scale,sigmaRBF,signal_var must be positive
    g=ForwardDiff.gradient(logmarginal)
    function g!(hyperparams::Vector,storage::Vector)
        grad=g(hyperparams)
        for i=1:length(hyperparams)
            storage[i]=grad[i]
        end
    end
    l=optimize(logmarginal,g!,log([init_length_scale,init_sigma_RBF,init_signal_var]),method=:cg,show_trace = true, extended_trace = true)
	return exp(l.minimum)
end


function featureNotensor(X::Array,length_scale,sigma_RBF::Real,Z::Array,b::Array)    
    N,D=size(X)
	n=size(Z,1)
    phi=Array(Float64,n,N)
    Zt=scale(Z,1./length_scale)
    for i=1:N
		for j=1:n
        	phi[j,i]=cos(sum(X[i,:].*Zt[j,:]) + b[j])
		end
    end
    return sqrt(2/n)*sigma_RBF*phi
end

function featureNotensor_tilde(X::Array,length_scale,sigma_RBF::Real,Z::Array,b::Array)    
    N,D=size(X)
	n=size(Z,1)
    phi=Array(Float64,n,N)
    Zt=scale(Z,1./length_scale)
    Ztt=scale(Z,1./(length_scale^2))
    for i=1:N
        phi[:,i]=sin(sum(repmat(X[i,:],n,1).*Zt,2) + b) .* sum(repmat(X[i,:],n,1).*Ztt,2)
    end
    return sqrt(2/n)*sigma_RBF*phi
end



function GPTregression(phi::Array, y::Array, signal_var::Real, I::Array, r::Integer, Q::Integer, m::Integer, epsw::Real, epsU::Real, burnin::Integer, maxepoch::Integer,param_seed::Integer;langevin=true,stiefel=true)
 
    n,D,N=size(phi)
    numbatches=int(ceil(N/m))
    sigma_w=1;
    signal_u=0.01;
    
    # initialise w,U^(k)
    srand(param_seed);
    w_store=Array(Float64,Q,maxepoch*numbatches)
    U_store=Array(Float64,n,r,D,maxepoch*numbatches)
    #w=sigma_w*randn(Q)
    w=ones(r)
 
    if stiefel
	U=Array(Float64,n,r,D)
	for k=1:D
	    Z=randn(r,n)
	    U[:,:,k]=transpose(\(sqrtm(Z*Z'),Z)) #sample uniformly from V_{n,r}
	end
    else U=randn(n,r,D) #/sqrt(n)
    end

    for epoch=1:(burnin+maxepoch)
        #randomly permute training data and divide into mini_batches of size m
        perm=randperm(N)
        phi=phi[:,:,perm]; y=y[perm];
        
        # run SGLD on w and SGLDERM on U
        for batch=1:numbatches
            # random samples for the stochastic gradient
            idx=(m*(batch-1)+1):min(m*batch,N)
            phi_batch=phi[:,:,idx]; y_batch=y[idx];
            batch_size=length(idx) #this is m except for last batch

            # compute <phi^(k)(x_i),U^(k)_{.l}> for all k,l,batch and store in temp
            temp=phidotU(U,phi_batch)

	    # compute V st V[q,i]=prod_{k=1 to D}(temp[k,I[q,k],i])
            V=computeV(temp,I)
	    
            # compute fhat where fhat[i]=V[:,i]'w
            fhat=computefhat(V,w)

            # now can compute gradw, the stochastic gradient of log post wrt w
            #gradw=(N/batch_size)*V*(y_batch-fhat)/signal_var-w/(sigma_w^2)

            # compute U_phi[q,i,k]=expression in big brackets in (11)
            U_phi=computeU_phi(V,temp,I)
            
            # compute a_l^(k)(x_i) for l=1,...,r k=1,..,D and store in A
            A=computeA(U_phi,w,I,r)
            
            # compute Psi as in (12)
            Psi=computePsi(A,phi_batch)
            
            # can now compute gradU where gradU[:,:,k]=stochastic gradient of log post wrt U^(k)
            gradU=Array(Float64,n,r,D)
	    if stiefel
		    for k=1:D
		        gradU[:,:,k]=reshape((N/batch_size)*Psi[:,:,k]*(y_batch-fhat)/signal_var,n,r)
		    end
	    else  for k=1:D
		        gradU[:,:,k]=reshape((N/batch_size)*Psi[:,:,k]*(y_batch-fhat)/signal_var,n,r)-U[:,:,k]/signal_u
		    end
	    end
	    
            # update w
	    #if langevin
	#	w+=epsw*gradw/2+sqrt(epsw)*randn(Q)
	   # else w+=epsw*gradw/2
	   # end

            # update U
	    if langevin
		if stiefel
		    for k=1:D
		        mom=proj(U[:,:,k],sqrt(epsU)*gradU[:,:,k]/2+randn(n,r))
		        U[:,:,k]=geod(U[:,:,k],mom,sqrt(epsU));
		        if U[:,:,k]==zeros(n,r) #if NaN appears while evaluating G
		            return zeros(Q,maxepoch*numbatches),zeros(n,r,D,maxepoch*numbatches)
		        end
		    end
		else U+=epsU*gradU/2+sqrt(epsU)*randn(n,r,D)
		end
	    else
		if stiefel
			for k=1:D
		        mom=proj(U[:,:,k],sqrt(epsU)*gradU[:,:,k]/2)
		        U[:,:,k]=geod(U[:,:,k],mom,sqrt(epsU));
		        if U[:,:,k]==zeros(n,r) #if NaN appears while evaluating G
		            return zeros(Q,maxepoch*numbatches),zeros(n,r,D,maxepoch*numbatches)
		        end
		    end
		else U+=epsU*gradU/2
		end
	    end

	    if epoch>burnin
	        w_store[:,((epoch-burnin)-1)*numbatches+batch]=w
	        U_store[:,:,:,((epoch-burnin)-1)*numbatches+batch]=U
	    end
        end
    end
    return w_store,U_store
end


function pred_CF(w::Array,U::Array,V::Array,phiU::Array,phiV::Array)
    PhidotU=U'*phiU; PhidotV=V'*phiV ;fw=PhidotU.*PhidotV;
    return fw'*w
end

function GPT_CF(phiU::Array, phiV::Array,y::Array, signal_var::Real,var_u::Real, r::Integer, m::Integer, epsw::Real, epsU::Real, maxepoch::Integer,param_seed::Integer,langevin::Bool,stiefel::Bool)
 
    n1,N=size(phiU); n2=size(phiV,1)
    numbatches=int(ceil(N/m))
    var_w=1;
    
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

    for epoch=1:maxepoch
        #randomly permute training data and divide into mini_batches of size m
        perm=randperm(N)
        phiU=phiU[:,perm]; phiV=phiV[:,perm];y=y[perm];

        # run SGLD on w and SGLDERM on U
        for batch=1:numbatches
            # random samples for the stochastic gradient
            idx=(m*(batch-1)+1):min(m*batch,N)
            phi_batchU=phiU[:,idx];phi_batchV=phiV[:,idx]; y_batch=y[idx];
            batch_size=length(idx) 

	    PhidotU=U'*phi_batchU; PhidotV=V'*phi_batchV ;fw=PhidotU.*PhidotV;
	    fhat=fw'*w;
            # now can compute gradw, the stochastic gradient of log post wrt w
            #gradw=(N/batch_size)*fw*(y_batch-fhat)/signal_var-w/var_w

          
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
	  #=  if langevin
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
	        w_store[:,epoch]=w
	        U_store[:,:,epoch]=U
	        V_store[:,:,epoch]=V
    end
    return w_store,U_store,V_store
end


function GPT_CFgibbs(phiU::Array, phiV::Array,y::Array, signal_var::Real, var_u::Real, var_w::Real, r::Integer, maxepoch::Integer)
 
    n1,N=size(phiU); n2=size(phiV,1)
    
    # initialise w,U^(k)
    w_store=Array(Float64,r,maxepoch)
    U_store=Array(Float64,n1,r,maxepoch)
    V_store=Array(Float64,n2,r,maxepoch)
    w=sqrt(var_w)*randn(r)
   # w=ones(r)
    U=sqrt(var_u)*randn(n1,r);
    V=sqrt(var_u)*randn(n2,r)


    for epoch=1:maxepoch
	    PhidotU=U'*phiU; PhidotV=V'*phiV ;fw=PhidotU.*PhidotV;
	    fhat=fw'*w;
 
            #gibbs on w
            invSigma_w = 1/(signal_var) * fw * fw' + (1/var_w)*eye(r)
            Mu_w = \(invSigma_w,1/(var_w) *(fw * y))
            w[:] = \(chol(invSigma_w,:U),randn(r)) + Mu_w
    
	    PsiU=Array(Float64,n1*r,N);PsiV=Array(Float64,n2*r,N);
	    for i=1:N
		PsiU[:,i]=kron(w.*PhidotV[:,i],phiU[:,i])
		PsiV[:,i]=kron(w.*PhidotU[:,i],phiV[:,i])
	    end

	    invSigma_U = PsiU * PsiU'/signal_var + (1/var_u) * eye(n1*r)	
	    invSigma_V = PsiV * PsiV'/signal_var + (1/var_u) * eye(n2*r)
            Mu_U = \(invSigma_U, (PsiU * y) / signal_var)
            Mu_V = \(invSigma_V, (PsiV * y) / signal_var)
            U= reshape(\(chol(invSigma_U,:U),randn(n1*r)) + Mu_U,n1,r)	
            V= reshape(\(chol(invSigma_V,:U),randn(n2*r)) + Mu_V,n2,r)			
 
	         w_store[:,epoch]=w
	         U_store[:,:,epoch]=U
	         V_store[:,:,epoch]=V
    end
    return w_store,U_store,V_store
end



#SGLD on Tucker Model with Stiefel Manifold, learning hyperparameters as well
function GPT_SGLDERM_adam(X::Array, y::Array, I::Array, n::Integer,r::Integer, Q::Integer, m::Integer, epsw::Real, epsU::Real, burnin::Integer, maxepoch::Integer,seed::Integer,hyp_init::Array)
    sigma_w=1;
    sigma_hyper = [1,1,1];
    beta_1,beta_2,alpha0,epsilon = 0.9,0.999,0.001,0.00000001
    N,D=size(X)
    scale=sqrt(n/(Q^(1/D)))  
    length_scale,sigma_RBF,signal_var = hyp_init #ones(3); #exp(sqrt(sigma_hyper).*randn(3))
    numbatches=int(ceil(N/m))
   
    # initialise w,U^(k)
    w_store=Array(Float64,Q,maxepoch*numbatches)
    U_store=Array(Float64,n,r,D,maxepoch*numbatches)
    w=sigma_w*randn(Q)

    U=Array(Float64,n,r,D)
    for k=1:D
        Z=randn(r,n)
        U[:,:,k]=transpose(\(sqrtm(Z*Z'),Z)) #sample uniformly from V_{n,r}
    end

    # initialise w,U^(k)
    w_store=Array(Float64,Q,maxepoch*numbatches)
    U_store=Array(Float64,n,r,D,maxepoch*numbatches) 
    l_store=Array(Float64,maxepoch*numbatches)
    SigmaRBF_store=Array(Float64,maxepoch*numbatches)
    SignalVar_store=Array(Float64,maxepoch*numbatches)
   
    #initialise 1st and 2nd moment vectors for ADAM
    moment1 = zeros(3) ; moment2 = zeros(3) #moments for w and hyperparameters

    for epoch=1:(burnin+maxepoch)
        #randomly permute training data and divide into mini_batches of size m
        perm=randperm(N)
        Xperm = X[perm,:];  yperm=y[perm];
        # run SGLD on w and SGLDERM on U
        for batch=1:numbatches
            t = (epoch-1)*numbatches+batch
            alpha = alpha0 * sqrt(1-beta_2^t) / (1-beta_1^t)
            # random samples for the stochastic gradient
            idx=(m*(batch-1)+1):min(m*batch,N)
            phi_batch=feature(Xperm[idx,:],n,length_scale,sigma_RBF,seed,scale); y_batch=yperm[idx];
            batch_size=length(idx) #this is m except for last batch

            # compute <phi^(k)(x_i),U^(k)_{.l}> for all k,l,batch and store in temp
            temp=phidotU(U,phi_batch)

	    # compute V st V[q,i]=prod_{k=1 to D}(temp[k,I[q,k],i])
            V=computeV(temp,I)
	    
            # compute fhat where fhat[i]=V[:,i]'w
            fhat=computefhat(V,w)

            # now can compute negative gradw, the stochastic gradient of log post wrt w
            gradw= (N/batch_size)*V*(y_batch-fhat)/signal_var-w/(sigma_w^2)

            # compute U_phi[q,i,k]=expression in big brackets in (11)
            U_phi=computeU_phi(V,temp,I)
            
            # compute a_l^(k)(x_i) for l=1,...,r k=1,..,D and store in A
            A=computeA(U_phi,w,I,r)
            
            # compute Psi as in (12)
            Psi=computePsi(A,phi_batch)
            
            # can now compute negative gradU where gradU[:,:,k]=stochastic gradient of log post wrt U^(k)
            gradU=Array(Float64,n,r,D)
            for k=1:D
                gradU[:,:,k]=reshape((N/batch_size)*Psi[:,:,k]*(y_batch-fhat)/signal_var,n,r)
            end

            ## SGLD on length scale and sigma_RBF
		theta = [log(length_scale),log(sigma_RBF),log(signal_var)];
		#write log likelihood as a function of log length scale
		function lik_theta(theta::Vector)
			length_scale, sigma_RBF = exp(theta);
			phi_batch = feature(Xperm[idx,:],n,length_scale,sigma_RBF,seed,scale);
			temp = phidotU(U,phi_batch);
			V = computeV(temp,I)
			fhat=computefhat(V,w)
			return(sum((y_batch-fhat).^2))
		end

		g = ForwardDiff.gradient(lik_theta)(theta)
                gradl=  -( (N/batch_size)*g[1]/(2*signal_var) )#+ theta[1]/(sigma_hyper[1]^2) )
		gradrbf=  -( (N/batch_size)*g[2]/(2*signal_var) )#+ theta[2]/(sigma_hyper[2]^2) )
                gradtau= (N/batch_size)*exp(-theta[3])*sum((y_batch-fhat).^2)/2 - N/2 #- theta[3]/(sigma_hyper[3]^2) 

      	    grad_hyper = [gradl,gradrbf,gradtau]
            moment1 = beta_1*moment1 + (1-beta_1)*grad_hyper
            moment2 = beta_2*moment2 + (1-beta_2)*grad_hyper.^2
 	    # SGLD step on w
            w[:]+=epsw*gradw/2 +sqrt(epsw)*randn(Q)
            # SGLDERM step on U
            for k=1:D
                mom=proj(U[:,:,k],sqrt(epsU)*gradU[:,:,k]/2+randn(n,r))
                U[:,:,k]=geod(U[:,:,k],mom,sqrt(epsU));
                if U[:,:,k]==zeros(n,r) 
                    return zeros(Q,maxepoch*numbatches),zeros(n,r,D,maxepoch*numbatches)
                end
            end
  		
		delta_theta = alpha*moment1./(sqrt(moment2) + epsilon)
		stepsize = abs(2*delta_theta./grad_hyper)
		theta += delta_theta #+ sqrt(stepsize).*randn(3)

		length_scale,sigma_RBF,signal_var = exp(theta)
		#println("grad_hyper[1] = ", grad_hyper[1], "; update[1] = ", exp(alpha*moment1./(sqrt(moment2) + epsilon)[1]))
	    if epoch>burnin
	        w_store[:,((epoch-burnin)-1)*numbatches+batch]=w
	        U_store[:,:,:,((epoch-burnin)-1)*numbatches+batch]=U
                l_store[((epoch-burnin)-1)*numbatches+batch] = length_scale
		SigmaRBF_store[((epoch-burnin)-1)*numbatches+batch]=sigma_RBF
		SignalVar_store[((epoch-burnin)-1)*numbatches+batch]=signal_var
	    end
		
        end
    end
	println("length_scale = ", length_scale, "; SigmaRBF = ", sigma_RBF, "; SignalVar = ", signal_var)
    return w_store,U_store, l_store, SigmaRBF_store, SignalVar_store
end
	




#SGLD on Tucker Model with Stiefel Manifold, learning hyperparameters as well
function GPT_SGLDERM_hyper(X::Array, y::Array, I::Array, n::Integer,r::Integer, Q::Integer, m::Integer, epsw::Real, epsU::Real, burnin::Integer, maxepoch::Integer,
				epslnl::Real, epslnSrbf::Real, epstau::Real, seed::Integer)
    # phi is the D by n by N array of features where phi[k,:,i]=phi^(k)(x_i)
    # signal_var is the variance of the observed values
    # epsw,epsU are the epsilons for w and U resp.
    # maxepoch is the number of sweeps through whole dataset
    sigma_w=1;
    sigma_lnl = 1;
    sigma_lnSrbf = 1;
    sigma_lnSvar = 1;
    N,D=size(X)
    scale=sqrt(n/(Q^(1/D)))
    length_scale = exp(randn(sigma_lnl))[1]; sigma_RBF = exp(randn(sigma_lnSrbf))[1]; signal_var = var(y);
    phi=feature(X,n,length_scale,sigma_RBF,seed,scale);
    n,D,N=size(phi)
    numbatches=int(ceil(N/m))
    # initialise w,U^(k)
    w_store=Array(Float64,Q,maxepoch*numbatches)
    U_store=Array(Float64,n,r,D,maxepoch*numbatches) 
    l_store=Array(Float64,maxepoch*numbatches)
    SigmaRBF_store=Array(Float64,maxepoch*numbatches)
    SignalVar_store=Array(Float64,maxepoch*numbatches)
    w=sigma_w*randn(Q)

    U=Array(Float64,n,r,D)
    for k=1:D
        Z=randn(r,n)
        U[:,:,k]=transpose(\(sqrtm(Z*Z'),Z)) #sample uniformly from V_{n,r}
    end


    for epoch=1:(burnin+maxepoch)
        #randomly permute training data and divide into mini_batches of size m
        perm=randperm(N)
        X = X[perm,:]; y=y[perm];
        
        # run SGLD on w and SGLDERM on U
        for batch=1:numbatches

    	    phi=feature(X,n,length_scale,sigma_RBF,seed,scale);
            # random samples for the stochastic gradient
            idx=(m*(batch-1)+1):min(m*batch,N)
            phi_batch=phi[:,:,idx]; y_batch=y[idx];
            batch_size=length(idx) #this is m except for last batch

            # compute <phi^(k)(x_i),U^(k)_{.l}> for all k,l,batch and store in temp
            temp=phidotU(U,phi_batch)

	    # compute V st V[q,i]=prod_{k=1 to D}(temp[k,I[q,k],i])
            V=computeV(temp,I)
	    
            # compute fhat where fhat[i]=V[:,i]'w
            fhat=computefhat(V,w)

            # now can compute gradw, the stochastic gradient of log post wrt w
            gradw=(N/batch_size)*V*(y_batch-fhat)/signal_var-w/(sigma_w^2)

            # compute U_phi[q,i,k]=expression in big brackets in (11)
            U_phi=computeU_phi(V,temp,I)
            
            # compute a_l^(k)(x_i) for l=1,...,r k=1,..,D and store in A
            A=computeA(U_phi,w,I,r)
            
            # compute Psi as in (12)
            Psi=computePsi(A,phi_batch)
            
            # can now compute gradU where gradU[:,:,k]=stochastic gradient of log post wrt U^(k)
            gradU=Array(Float64,n,r,D)
            for k=1:D
                gradU[:,:,k]=reshape((N/batch_size)*Psi[:,:,k]*(y_batch-fhat)/signal_var,n,r)
            end
	    
 
            ## SGLD on length scale and sigma_RBF
		theta = [log(length_scale),log(sigma_RBF),log(signal_var)];
		#write log likelihood as a function of log length scale
		function lik_theta(theta::Vector,)
			length_scale, sigma_RBF = exp(theta);
			phi = feature(X,n,length_scale,sigma_RBF,seed,scale);
			phi_batch=phi[:,:,idx]
			temp = phidotU(U,phi_batch);
			V = computeV(temp,I)
			fhat=computefhat(V,w)
			return(sum((y_batch-fhat).^2))
		end

		g = ForwardDiff.gradient(lik_theta)(theta)

 	    # SGLD step on w
            w[:]+=epsw*gradw/2 +sqrt(epsw)*randn(Q)
            # SGLDERM step on U
            for k=1:D
                mom=proj(U[:,:,k],sqrt(epsU)*gradU[:,:,k]/2+randn(n,r))
                U[:,:,k]=geod(U[:,:,k],mom,sqrt(epsU));
                if U[:,:,k]==zeros(n,r) #if NaN appears while evaluating G
                    return zeros(Q,maxepoch*numbatches),zeros(n,r,D,maxepoch*numbatches)
                end
            end

                gradl=-(N/batch_size)*g[1]/(2*signal_var)-theta[1]/(sigma_lnl^2)
		theta[1] += epslnl*gradl/2 + sqrt(epslnl)*randn(1)[1]
		gradrbf=-(N/batch_size)*g[2]/(2*signal_var)-theta[2]/(sigma_lnSrbf^2)
		theta[2] += epslnSrbf*gradrbf/2 + sqrt(epslnSrbf)*randn(1)[1]
                gradtau=(N/batch_size)*exp(-theta[3])*sum((y_batch-fhat).^2)/2 - N/2 -theta[3]/(sigma_lnSvar^2)
		theta[3] += epstau*gradtau/2 + sqrt(epstau)*randn(1)[1]
		length_scale,sigma_RBF,signal_var = exp(theta)

	    if epoch>burnin
	        w_store[:,((epoch-burnin)-1)*numbatches+batch]=w
	        U_store[:,:,:,((epoch-burnin)-1)*numbatches+batch]=U
                l_store[((epoch-burnin)-1)*numbatches+batch] = length_scale
		SigmaRBF_store[((epoch-burnin)-1)*numbatches+batch]=sigma_RBF
		SignalVar_store[((epoch-burnin)-1)*numbatches+batch]=signal_var
	    end
		#tmp = ((0.01+(epoch-1)*numbatches+batch)/(0.01+(epoch-1)*numbatches+batch-1))^(-0.5)
		#epslnl *= tmp; epslnSrbf*= tmp; epstau*= tmp
        end
    end
    return w_store,U_store, l_store, SigmaRBF_store, SignalVar_store
end
end
