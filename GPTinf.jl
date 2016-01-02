module GPTinf

using Distributions

export datawhitening,feature,samplenz,RMSE, GPTgibbs, GPTSGLD, RMSESGLD, GPNHT_SGLDERM, RMSESGLDvec, pred, data_simulator

function datawhitening(X::Array)
    for i = 1:size(X,2)
        X[:,i] = (X[:,i] - mean(X[:,i]))/std(X[:,i])
    end
    return X
end

function data_simulator(X::Array,n::Integer,r::Integer,Q::Integer,sigma::Real,length_scale::Real,seed::Integer)
  N,D = size(X)
  w = randn(Q)
  U=Array(Float64,n,r,D)
  for k=1:D
    Z=randn(r,n)
    U[:,:,k]=transpose(\(sqrtm(Z*Z'),Z)) 
  end
  I=samplenz(r,D,Q,seed)
  phi = feature(X,r,length_scale,seed,1)
  temp=phidotU(U,phi)
  V=computeV(temp,I)
  return computefhat(V,w) + sigma*randn(N)
end

#extract features from tensor decomp of each row of X
function feature(X::Array,n::Integer,length_scale::Real,seed::Integer,scale::Real)
    N,D=size(X)
    phi=Array(Float64,n,D,N)
    srand(seed)
    Z=randn(n,D)/length_scale
    b=rand(n,D)*2*pi
    for i=1:N
    	for k=1:D
	      for j=1:n
      		phi[j,k,i]=cos(X[i,k]*Z[j,k]+b[j,k])
	      end
    	end
    end
    return scale*sqrt(2/n)*phi
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

# define geod for Stiefel manifold - just want endpt
function geod(U::Array,mom::Array,t::Real)
    n,r=size(U)
    A=U'*mom
    temp=[A -mom'*mom;eye(r) A]
    E=expm(t*temp) #can become NaN when temp too large. Return 0 in this case
    if sum(isnan(E))>0
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
function geodboth(U::Array,mom::Array,t::Real)
    n,r=size(U)
    A=U'*mom
    temp=[A -mom'*mom;eye(r) A]
    E=expm(t*temp) #can become NaN when temp too large. Return 0 in this case
    if sum(isnan(E))>0
        return zeros(n,r),zeros(n,r)
    else
        mexp=expm(-t*A)
        tmpU=[U mom]*E[:,1:r]*mexp;
        tmpV=[U mom]*E[:,(r+1):end]*mexp;
        #ensure that tmpU has cols of unit norm
        normconst=Array(Float64,1,r);
        for l=1:r
    	    normconst[1,l]=norm(tmpU[:,l])
        end
        return tmpU./repmat(normconst,n,1),tmpV
    end
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

#write RMSE to filename
#=function SDexp(phitrain::Array,phitest::Array,ytrain::Array,ytest::Array,ytrainStd::Real,seed::Integer,sigma::Real,
    I::Array, length_scale::Real,r::Integer,Q::Integer,burnin::Integer,numiter::Integer,filename::ASCIIString)
    n=size(phitrain,1);D=size(phitrain,2);
    w_store,U_store=GPTgibbs(phitrain,ytrain,sigma,I,r,Q,burnin,numiter);
        predRMSE=RMSE(w_store,U_store,I,phitest,ytest);
    outfile=open(filename,"a") #append to file
    println(outfile,"RMSE=",ytrainStd*predRMSE,";seed=",seed,";sigma=",sigma,";length_scale=",length_scale,";n=",n,
    ";r=",r,";Q=",Q,";burnin=",burnin,";numiter=",numiter);
    close(outfile)
    return w_store,U_store
end=#


function GPTgibbs(phi::Array,y::Array,sigma::Real,I::Array,r::Integer,Q::Integer,burnin::Integer,numiter::Integer)
    # phi is the D by n by N array of features where phi[k,:,i]=phi^(k)(x_i)
    # sigma is the s.d. of the observed values
    # sigma_w is the s.d. for the Guassian prior on w
    # epsw,epsU are the epsilons for w and U resp.
    # maxepoch is the number of sweeps through whole dataset

    n,D,N=size(phi)
    sigma_u = sqrt(1/r)
    sigma_w = sqrt(r^D/Q)
    # initialise w,U^(k)
    w_store=Array(Float64,Q,numiter)
    U_store=Array(Float64,n,r,D,numiter)
    w=sigma_w*randn(Q)
    #println("w= ",w)
    U=Array(Float64,n,r,D)
    for k=1:D
        U[:,:,k]=sigma_u*randn(n,r)
    end


    for epoch=1:(burnin+numiter)

        # compute <phi^(k)(x_i),U^(k)_{.l}> for all k,l,batch and store in temp
        temp=phidotU(U,phi)

        # compute V st V[q,i]=prod_{k=1 to D}(temp[k,I[q,k],i])
        V=computeV(temp,I)

        #gibbs on w
        invSigma_w = 1/(sigma^2) * V * V' + (1/sigma_w^2)*eye(Q)
        Mu_w = \(invSigma_w,1/(sigma^2) *(V * y))
        w[:] = \(chol(invSigma_w,:U),randn(Q)) + Mu_w

        # compute U_phi[q,i,k]=expression in big brackets in (11)

        for k in 1:D
            U_phi=Array(Float64,Q,N)
            temp=phidotU(U,phi)
            V=computeV(temp,I)

            for i=1:N
                for q=1:Q
                    U_phi[q,i]=V[q,i]/temp[k,I[q,k],i]
                end
            end
        # now compute a_l^(k)(x_i) for l=1,...,r k=1,..,D and store in A
            A=zeros(r,N)
            for i=1:N
                    for l in unique(I[:,k])
                        index=findin(I[:,k],l) #I_l
                        A[l,i]=dot(U_phi[index,i],w[index])
                    end
            end

            # compute Psi as in (12)
            Psi=Array(Float64,n*r,N)
            for i=1:N
                    Psi[:,i]=kron(A[:,i],phi[:,k,i])
            end

            invSigma_U = Psi * Psi'/(sigma^2) + (1/sigma_u)^2 * eye(n*r)
            Mu_U = \(invSigma_U, (Psi * y) / (sigma^2))
            U[:,:,k]= reshape(\(chol(invSigma_U,:U),randn(n*r)) + Mu_U,n,r)
        end

        if epoch>burnin
            w_store[:,epoch-burnin]= w
            U_store[:,:,:,epoch-burnin]=U
        end
    end
    return w_store,U_store
end

function GPTSGLD(phi::Array,y::Array,sigma::Real,I::Array,r::Integer,Q::Integer,m::Integer,epsw::Real,epsU::Real,burnin::Integer,maxepoch::Integer)
    # phi is the D by n by N array of features where phi[k,:,i]=phi^(k)(x_i)
    # sigma is the s.d. of the observed values
    # epsw,epsU are the epsilons for w and U resp.
    # maxepoch is the number of sweeps through whole dataset

    n,D,N=size(phi)
    numbatches=int(ceil(N/m))
    sigma_w = sqrt(r^D/Q);

    # initialise w,U^(k)
    w_store=Array(Float64,Q,maxepoch*numbatches)
    U_store=Array(Float64,n,r,D,maxepoch*numbatches)
    w=sigma_w*randn(Q)

    U=Array(Float64,n,r,D)
    for k=1:D
        U[:,:,k]= sqrt(1/r) * randn(n,r) #sample uniformly from V_{n,r}
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
            gradw=(N/batch_size)*V*(y_batch-fhat)/(sigma^2)-w/(sigma_w^2)

            # compute U_phi[q,i,k]=expression in big brackets in (11)
            U_phi=Array(Float64,Q,batch_size,D)
            for k=1:D
	          	for i=1:batch_size
		            for q=1:Q
		            	U_phi[q,i,k]=V[q,i]/temp[k,I[q,k],i]
		            end
	          	end
            end
            # now compute a_l^(k)(x_i) for l=1,...,r k=1,..,D and store in A
            A=zeros(r,D,batch_size)
            for i=1:batch_size
                for k=1:D
                    for l in unique(I[:,k])
                        index=findin(I[:,k],l) #I_l
                        A[l,k,i]=dot(U_phi[index,i,k],w[index])
                    end
                end
            end
            # compute Psi as in (12)
            Psi=Array(Float64,n*r,batch_size,D)
            for i=1:batch_size
                for k=1:D
                    Psi[:,i,k]=kron(A[:,k,i],phi_batch[:,k,i])
                end
            end
            # can now compute gradU where gradU[:,:,k]=stochastic gradient of log post wrt U^(k)
            gradU=Array(Float64,n,r,D)
            for k=1:D
                gradU[:,:,k]=reshape((N/batch_size)*Psi[:,:,k]*(y_batch-fhat)/(sigma^2),n,r)
            end

            # SGLD step on w
            w[:]+=epsw*gradw/2 +sqrt(epsw)*randn(Q)
	    #if batch==1
	    #	println("mean epsgradw_half=",mean(epsw*gradw/2)," std =",std(epsw*gradw/2))
	    #	println("meansqrtepsgradU_half=",mean(sqrt(epsU)*gradU/2), " std=",std(sqrt(epsU)*gradU/2))
	    #end
            # SGLDERM step on U
            for k=1:D
               U[:,:,k]+= epsU* gradU[:,:,k]/2 +sqrt(epsU)*randn(n,r)
            end
	          if epoch>burnin
	            w_store[:,((epoch-burnin)-1)*numbatches+batch]=w
	            U_store[:,:,:,((epoch-burnin)-1)*numbatches+batch]=U
	        end
        end
        epsw = epsw * (epoch/(epoch + 1))^(0.55)
    end
    return w_store,U_store
end


function RMSESGLDvec(w_store::Array,U_store::Array,I::Array,phitest::Array,ytest::Array)
    Ntest=length(ytest);
    T=size(w_store,2);
   #= meanfhat= @parallel (+) for i=1:T
        pred(w_store[:,i],U_store[:,:,:,i],I,phitest);
    end
    meanfhat=meanfhat/T;=#
    yfit = Array(Float64,Ntest,T); RMSEvec = Array(Float64,1,T);
    for i = 1:T
      yfit[:,i] = pred(w_store[:,i],U_store[:,:,:,i],I,phitest);
      RMSEvec[1,i] = norm(ytest-yfit[:,i])/sqrt(Ntest);
        end
    meanfhat=mean(yfit,2)
#return norm(ytest-meanfhat)/sqrt(Ntest),RMSEvec;
return RMSEvec
end

function RMSESGLD(w_store::Array,U_store::Array,I::Array,phitest::Array,ytest::Array)
    Ntest=length(ytest);
    T=size(w_store,2);
    meanfhat= @parallel (+) for i=1:T
        pred(w_store[:,i],U_store[:,:,:,i],I,phitest);
    end
    meanfhat=meanfhat/T;
return norm(ytest-meanfhat)/sqrt(Ntest);
end

function GPNHT_SGLDERM(phi::Array,y::Array,sigma::Real,I::Array,r::Integer,Q::Integer,m::Integer,epsw::Real,epsU::Real,
    burnin::Integer,maxepoch::Integer,L::Integer,A_w::Real, A_U::Real)
    # phi is the D by n by N array of features where phi[k,:,i]=phi^(k)(x_i)
    # sigma is the s.d. of the observed values
    # epsw,epsU are the epsilons for w and U resp.
    # maxepoch is the number of sweeps through whole dataset

    n,D,N=size(phi)
    numbatches=int(ceil(N/m))
    sigma_w=1;

    # initialise w,U^(k)
    w_store=Array(Float64,Q,maxepoch*numbatches)
    U_store=Array(Float64,n,r,D,maxepoch*numbatches)
    w=sigma_w*randn(Q)

    U=Array(Float64,n,r,D)
    for k=1:D
        Z=randn(r,n)
        U[:,:,k]=transpose(\(sqrtm(Z*Z'),Z)) #sample uniformly from V_{n,r}
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

            p = randn(Q); zeta_w = A_w;  zeta_U = A_U*ones(D); V_U = randn(n,r,D)
            for leapfrog = 1:L

            # compute <phi^(k)(x_i),U^(k)_{.l}> for all k,l,batch and store in temp
                temp=phidotU(U,phi_batch)

                # compute V st V[q,i]=prod_{k=1 to D}(temp[k,I[q,k],i])
                V=computeV(temp,I)

                # compute fhat where fhat[i]=V[:,i]'w
                fhat=computefhat(V,w)

                # compute U_phi[q,i,k]=expression in big brackets in (11)
                U_phi=Array(Float64,Q,batch_size,D)
                for k=1:D
                    for i=1:batch_size
                        for q=1:Q
                            U_phi[q,i,k]=V[q,i]/temp[k,I[q,k],i]
                        end
                    end
                end
                # now compute a_l^(k)(x_i) for l=1,...,r k=1,..,D and store in A
                A=zeros(r,D,batch_size)
                for i=1:batch_size
                    for k=1:D
                        for l in unique(I[:,k])
                            index=findin(I[:,k],l) #I_l
                            A[l,k,i]=dot(U_phi[index,i,k],w[index])
                        end
                    end
                end
                # compute Psi as in (12)
                Psi=Array(Float64,n*r,batch_size,D)
                for i=1:batch_size
                    for k=1:D
                        Psi[:,i,k]=kron(A[:,k,i],phi_batch[:,k,i])
                    end
                end

                # now can compute gradw, the stochastic gradient of log post wrt w
                gradw=(N/batch_size)*V*(y_batch-fhat)/(sigma^2)-w/(sigma_w^2)

                # can now compute gradU where gradU[:,:,k]=stochastic gradient of log post wrt U^(k)
                gradU=Array(Float64,n,r,D)
                for k=1:D
                    gradU[:,:,k]=reshape((N/batch_size)*Psi[:,:,k]*(y_batch-fhat)/(sigma^2),n,r)
                end

                # thermostats
                p += sqrt(epsw)*(gradw - zeta_w*p + sqrt(2*A_w*sqrt(epsw))*randn(Q))

                w[:] += sqrt(epsw)*p[:];
                for k = 1:D
                    V_U[:,:,k] = proj(U[:,:,k],sqrt(epsU)*(gradU[:,:,k] - zeta_U[k]* V_U[:,:,k]
                                + sqrt(2*A_U*sqrt(epsU))*randn(n,r)) +  V_U[:,:,k])
                    U[:,:,k], V_U[:,:,k]=geodboth(U[:,:,k],V_U[:,:,k],sqrt(epsU))
                end

                zeta_w += sqrt(epsw)*((p'*p)[1]/Q - 1)
                for k in 1:D
                    zeta_U[k] += sqrt(epsU)*(trace(V_U[:,:,k]'*V_U[:,:,k])/(n*r) - 1)
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


end
