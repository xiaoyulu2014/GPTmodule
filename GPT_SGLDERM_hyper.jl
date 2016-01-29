function GPT_SGLDERM_hyper(X::Array, y::Array, I::Array, n::Integer,r::Integer, Q::Integer, m::Integer, epsilon::Real, alpha::Real, burnin::Integer, maxepoch::Integer,seed::Integer)
    # phi is the D by n by N array of features where phi[k,:,i]=phi^(k)(x_i)
    # signal_var is the variance of the observed values
    # epsw,epsU are the epsilons for w and U resp.
    # maxepoch is the number of sweeps through whole dataset
    sigma_w=1;
    sigma_lnl = 1;
    sigma_lnSrbf = 1;
    tau_a = 1; tau_b = 1;
    N,D=size(X)
    scale=sqrt(n/(Q^(1/D)))
    length_scale = exp(randn(sigma_lnl))[1]; sigma_RBF = exp(randn(sigma_lnSrbf))[1]; tau = rand(Gamma(tau_a,tau_b))[1]; signal_var = 1/tau;
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
    # 
    gw=zeros(Q)
    gU=zeros(n,r,D)
    gl=zeros(1); gs=zeros(1); gvar=zeros(1)

    lambda=1e-5 #smoothing value for numerical convenience in RMSprop

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

            # now can compute gradw, the unnormalised stochastic gradient of log lik wrt w
            gradw=(1/batch_size)*V*(y_batch-fhat)/signal_var

            # update gw and compute step size for w
            gw=alpha*gw+(1-alpha)*(gradw.^2) 
            epsw=epsilon./(sqrt(gw)+lambda)

            # normalise stochastic grad and add grad of log prior to make grad of log post
            gradw=N*gradw-w/(sigma_w^2)

            # compute U_phi[q,i,k]=expression in big brackets in (11)
            U_phi=computeU_phi(V,temp,I)
            
            # compute a_l^(k)(x_i) for l=1,...,r k=1,..,D and store in A
            A=computeA(U_phi,w,I,r)
            
            # compute Psi as in (12)
            Psi=computePsi(A,phi_batch)
            
            # can now compute gradU where gradU[:,:,k]=stochastic gradient of log post wrt U^(k)
            gradU=Array(Float64,n,r,D)
            for k=1:D
                gradU[:,:,k]=reshape((1/batch_size)*Psi[:,:,k]*(y_batch-fhat)/signal_var,n,r)
            end
	    
            gU=alpha*gU+(1-alpha)*(gradU.^2)
            epsU=epsilon./(sqrt(gU)+lambda)
            meanepsU=vec(mean(epsU,[1,2]));

            gradU*=N

            ## SGLD on length scale and sigma_RBF
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

		theta = [log(length_scale),log(sigma_RBF)];  tau = 1/signal_var
		g = ForwardDiff.gradient(lik_theta)(theta)
                gradl=-(1/batch_size)*g[1]/(2*signal_var)
		gl=alpha*gl+(1-alpha)*(gradl.^2); epslnl=epsilon./(sqrt(gl)+lambda)
                gradl=N*gradl-theta[1]/(sigma_lnl^2)
		gradrbf=-(1/batch_size)*g[2]/(2*signal_var)-theta[2]/(sigma_lnSrbf^2)
		gs=alpha*gs+(1-alpha)*(gradrbf.^2); epslnSrbf=epsilon./(sqrt(gs)+lambda)
                gradrbf=N*gradrbf-theta[2]/(sigma_lnSrbf^2)
                gradtau=-(1/batch_size)*tau*sum(y_batch-fhat)
		gvar=alpha*gvar+(1-alpha)*(gradtau.^2); epstau=epsilon./(sqrt(gvar)+lambda)
                gradtau=N*gradtau+(tau_a-1)*log(tau) - tau/tau_b

            # SGLD step on w,Ul,sigma_rbf and signal_var
            w[:]+=epsw.*gradw/2 +sqrt(epsw).*randn(Q)
            for k=1:D
                mom=proj(U[:,:,k],sqrt(meanepsU[k])*gradU[:,:,k]/2+randn(n,r))
                U[:,:,k]=geod(U[:,:,k],mom,sqrt(meanepsU[k]));
                if U[:,:,k]==zeros(n,r) #if NaN appears while evaluating G
                    return zeros(Q,maxepoch*numbatches),zeros(n,r,D,maxepoch*numbatches)
                end
            end

		theta[1] += (epslnl*gradl/2 + sqrt(epslnl).*randn(1))[1]
                length_scale = exp(theta[1][1])
		theta[2] += (epslnSrbf*gradrbf/2 + sqrt(epslnSrbf).*randn(1))[1]
                sigma_RBF = exp(theta[2][1])
		tau += (epstau*gradtau/2 + sqrt(epstau).*randn(1))[1]
		signal_var = 1/tau

	    if epoch>burnin
	        w_store[:,((epoch-burnin)-1)*numbatches+batch]=w
	        U_store[:,:,:,((epoch-burnin)-1)*numbatches+batch]=U
                l_store[((epoch-burnin)-1)*numbatches+batch] = length_scale
		SigmaRBF_store[((epoch-burnin)-1)*numbatches+batch]=sigma_RBF
		SignalVar_store[((epoch-burnin)-1)*numbatches+batch]=signal_var
	    end
        end
    end
    return w_store,U_store, l_store, SigmaRBF_store, SignalVar_store
end
