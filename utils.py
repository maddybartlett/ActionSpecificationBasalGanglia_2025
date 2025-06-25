import numpy as np
import scipy 
from scipy.stats import multivariate_normal, beta
from scipy.special import softmax, log_softmax
from statsmodels.stats.weightstats import DescrStatsW
import matplotlib.pyplot as plt


def create_targets(M, n_seeds):
    ## create a list of target peaks so that there's an equal number of each
    targets_list = []
    for m in range(M):
        if int(n_seeds/M) == 0:
            targets_list.append([m] * 1)
        else:
            targets_list.append([m] * int(n_seeds/M))
            
    ## flatten the list
    temp_list = [x for list in targets_list for x in list]
            
    ## make sure len(targets_list) = n_seeds when n_seeds%M != 0
    if len(temp_list) != n_seeds:
        for i in range(n_seeds-len(temp_list)):
            targets_list.append([0])
            
    ## flatten the list
    targets_list = [x for list in targets_list for x in list]
    
    return targets_list


## Helper Functions
## Mu is the bundle
def born_prob(mu, phi_xs):
    ## compute the square of the dot product phi(x) dot mu
    num = (phi_xs @ mu.T)**2
    ## divide by eta (the sum of the squared similarity)
    eta = np.prod(1)*np.sum(num)
    return np.asarray(num/eta)
    #return np.einsum("d,nd -> n", mu, phi_xs)

def entropy(p_x):
    log_p_x = np.log(p_x)
    return np.sum(p_x*log_p_x)*-1

## we'll use a softmax function in the hidden layer 
def softmax_nonlinearity(t, x):
    #e_x = np.exp(x)
    #return e_x/np.sum(e_x)
    return softmax(x)
    

def standard_deviation(xs, Pxs):
    std = DescrStatsW(xs, weights=Pxs).std
    return std
    
    
def gen_discrete_dist(target_idx, patterns):
    ## randomly change some elements of a trained pattern to create a test pattern
    s = patterns[target_idx]*0.8
    short_patterns = np.delete(patterns, target_idx, axis=0)
    for i in range(len(short_patterns)):
        s += (short_patterns[i]*0.5) 
    return s

def gen_gaus_cont_dist(mean, covariance, domain, domain_phis):
    rvs = multivariate_normal(mean, covariance)
    ## generate samples from the probability density function
    Ps = rvs.pdf(domain)
    B = np.einsum('n,nd->d',Ps, domain_phis)
    ## weight each ssp by the probability and add them together
    # B = np.sum(np.asarray(domain_phis)*Ps.reshape(-1,1), axis=0)
    return B

def gen_beta_cont_dist(Ps, domain_phis):
    ## generate samples from the a beta probability density function
    B = np.einsum('n,nd->d',Ps, domain_phis)
    return B

def gen_mean_dist(mean, covariance, domain, domain_phis):
    rvs = multivariate_normal(mean, covariance)
    ## generate samples from the probability density function
    Ps = rvs.pdf(domain)
    B = np.einsum('n,nd->d',Ps, domain_phis)
    ## weight each ssp by the probability and add them together
    # B = np.sum(np.asarray(domain_phis)*Ps.reshape(-1,1), axis=0)
    dist_phis = np.asarray(domain_phis)*Ps.reshape(-1,1)
    mu = np.mean(dist_phis, axis=0)
    return mu, rvs
    
## Convert sparsity parameter to neuron bias/intercept
def sparsity_to_x_intercept(dim, proportion):
    sign = 1
    if proportion > 0.5:
        proportion = 1.0 - proportion
        sign = -1
    return sign * np.sqrt(1-scipy.special.betaincinv((dim-1)/2.0, 0.5, 2*proportion))

def get_data(domain, domain_phis, out_probe, in_probe, n_tests, deepatt = False):
    if deepatt:
        out_data=out_probe[0]
        in_data=in_probe[0]
    else:
        out_data=out_probe
        in_data=in_probe

    sim_per_inp = len(in_data)/n_tests
    out_std = []
    out_peak = []
    test_std = []
    test_peak = []
    out_entropy = []
    test_entropy = []
    sims_before_list = []
    sims_after_list = []

    ## MEASURE PERFORMANCE ##    
    ## STANDARD DEVIATION AND ENTROPY ##
    ## for each timestep, find the standard deviation, entropy and the peak of the output distribution
    for i in range(n_tests):
        ## get the first pattern from the input probe - the input bundle
        # print(f'len input data: {len(in_data)}')
        # print(f'index for input pattern: {int(i*(int(sim_per_inp)))}')
        test_pattern = in_data[int(i*(int(sim_per_inp)))].reshape(1,-1)
        ## decode the input bundle using similarity decoding
        sims_before = born_prob(test_pattern, domain_phis).astype(np.float32)
        ## add to the data list
        sims_before_list.append(sims_before)        
        ## record the standard deviation
        test_std.append(standard_deviation(domain, sims_before))
        ## and record where the peak is
        test_peak.append(domain[np.argmax(sims_before)])
        ## and record the entropy
        test_entropy.append(entropy(sims_before))

        ## get the last pattern from the output probe
        # print(f'output index: {int(((i+1)*(sim_per_inp))-1)}')
        sims_after = born_prob(out_data[int(((i+1)*(sim_per_inp))-1)].reshape(1,-1), domain_phis)
        sims_after_list.append(sims_after)
        ## record the standard deviation
        out_std.append(standard_deviation(domain, sims_after))
        ## and record where the peak is
        out_peak.append(domain[np.argmax(sims_after)])
        ## and record the entropy
        out_entropy.append(entropy(sims_after))

    return out_std, test_std, out_peak, test_peak, out_entropy, test_entropy, sims_before_list, sims_after_list

## function for generating training bundles and ssp targets
def get_training_data_discrete(n_samples, target_ssps, blur=False, dist=None):
    n_targets = len(target_ssps)
    bundles_ = []
    targets_ = []
    
    if blur:
        assert dist != None
    
    ## function for generation target, bundle pairs
    def gen_data_(a, n_targets, target_ssps, blur):
        sals = np.random.uniform(0,1.0, n_targets).tolist()
        ## rotate saliences until the max salience is in the right index position
        while sals[a] != max(sals):
            sals.append(sals.pop(0))

        ## choose a target 
        target = target_ssps[np.argmax(sals)]
        ## choose whether or not to use blurred targets
        if blur:
            target = gen_blurred_target_(target, a, dist, target_ssps)
        
        ## create bundle by weighting ssps according to a 
        ## uniform distribution with the largest salience at the target point
        #if space_type == 'continuous':        
        #bundle, rvs = gen_cont_dist(mean=domain[a], covariance=0.5, domain=domain, domain_phis=target_ssps)
        #elif space_type == 'discrete':
        bundle = np.sum([np.asarray(target_ssps[i])*sals[i] for i in range(len(target_ssps))], axis=0)
            
        return bundle, target
    
    ## repeat as many times n_targets fits into n_samples
    for i in range(int(n_samples/n_targets)):
        for a in range(n_targets):
            bundle, target = gen_data_(a, n_targets, target_ssps, blur)
            
            targets_.append(target)
            
            bundles_.append(bundle)
    
    ## now carry on for the remainder of n_samples
    for a in range(n_samples%n_targets):
        bundle, target = gen_data_(a, n_targets, target_ssps, blur)
        
        targets_.append(target)
        
        bundles_.append(bundle)
            
    return bundles_, targets_

def gen_blurred_target_(target, idx, dist, target_ssps): 
    '''
    function that takes the target ssp and blurs it
    
    This is achieved by taking the target ssp and N=dist neighbouring ssps,
    weighting the neighbours according to their (normalized) similarity
    to the target, and then summing all the ssps together to form a bundle. 
    '''
    centre = target
    dist = dist+1   
    
    ## calculate the start and end indexes for grabbing the neighbour ssps
    if idx >= dist:
        start = idx-dist
    else:
        start = 0
    end = idx+dist+1
    
    ## grab the neighbours
    surround = np.concatenate((target_ssps[start:idx], target_ssps[idx+1:end]))
    
    ## now weight each SSP by its similarity to the centre
    sims = (surround @ centre)
    weights = (sims-sims.min())/(1-sims.min()) # normalised similarity
    wphis = surround.T * weights

    ## then sum the ssps together into a blurred ssp
    blurred = np.sum(wphis, axis=1) + centre
        
    return blurred

def get_training_data_cont(n_samples, ssp_encoder, M, domain, domain_phis, dist_type):
    targets_ = []
    bundles_ = []
    for i in range(n_samples):
        if dist_type == 'gaussian':
            mean = np.random.uniform(0,M,1)
            cov = np.random.uniform(0.1,1.5,1)
            
            sample, rvs = np.array([gen_gaus_cont_dist(mean, cov, domain, domain_phis)][0])
            target = np.array([ssp_encoder.encode(mean)][0])

        elif dist_type == 'beta':
            a = np.random.uniform(1,10,1)
            b = np.random.uniform(1,10,1)
            
            Ps = beta.pdf(domain, a,b, scale=domain[-1])
            sample = np.einsum('nm,nd->d',Ps, domain_phis)

            target = np.array([ssp_encoder.encode(domain[np.argmax(Ps)])][0])
            
        bundles_.append(sample)
        targets_.append(target)
        
    return bundles_, targets_

def get_training_data_decode(n_samples, domain, M):
    '''
    Function for generating the training data needed to train the attractor
    with a decode layer. 
    '''
    targets_ = []
    bundles_ = []
    for i in range(n_samples):
        mean = np.random.uniform(0,M,1)
        cov = np.random.uniform(0.1,1.5,1)
        rvs = multivariate_normal(mean, cov)
        ## generate samples from the probability density function
        sample = rvs.pdf(domain)

        rvs_t = multivariate_normal(mean, 0.01)
        target = rvs_t.pdf(domain)
    
        targets_.append(target)
        bundles_.append(sample)
        
    return bundles_, targets_
    
def get_dl_testing_data(n_tests, domain, domain_phis, ssp_encoder, dist_type):
    ## create input patterns for testing the network 
    test_patterns = []
    test_labels = []
    for i in range(n_tests):
        if dist_type == 'gaussian':
            mean = np.random.uniform(0,M,1)
            cov = np.random.uniform(0.1,1.5,1)
            
            sample, rvs = np.array([gen_gaus_cont_dist(mean, cov, domain, domain_phis)][0])
            target = np.array([ssp_encoder.encode(mean)][0])

        elif dist_type == 'beta':
            a = np.random.uniform(1,10,1)
            b = np.random.uniform(1,10,1)
            
            Ps = beta.pdf(domain, a,b, scale=domain[-1])
            sample = np.einsum('nm,nd->d',Ps, domain_phis)

            target = np.array([ssp_encoder.encode(domain[np.argmax(Ps)])][0])
            
        ## Tile the pattern so that we can feed it in for a full second 
        inpt_pattern = np.tile(sample, (1000, 250))
        inpt_pattern = inpt_pattern.reshape(1000,250,512) 
        ## Tile the target label so that we can feed it in for a full second 
        label = np.tile(target, (1000, 250))
        label = label.reshape(1000,250,512) 

        test_patterns.append(inpt_pattern)
        test_labels.append(label)

    return test_patterns, test_labels

def get_dt_training_data(batch_size, domain, domain_phis, total_samples, N):
    n_training_pts = batch_size
    bundles = []
    for i in range(n_training_pts):
        mean=np.random.uniform(0,4,1)
        covs = np.linspace(0.01, 1.0, 500)

        RVs = [multivariate_normal(mean, cov) for cov in covs]
        RVs = reversed(RVs)
        Ps = [rvs.pdf(domain) for rvs in RVs]

        bundles.append([np.einsum('n,nd->d',p, domain_phis) for p in Ps])

        #bundles.append(Bs)
    train_labels_ = np.asarray(bundles)
    train_labels_ = train_labels_.reshape(total_samples,N)[:, None, :]
    
    ## next we need to generate the training pattern array by repeating the indexed ssps 500 times 
    pat_idx = np.arange(0,total_samples,500)
    train_patterns = np.asarray([train_labels_[idx] for idx in pat_idx])
    train_patterns = train_patterns.repeat(500, 1)
    train_patterns_ = train_patterns.reshape(total_samples,N)[:, None, :]
    
    return train_patterns_, train_labels_

def normalize(x, axis=-1, order=2):
    '''
    copied from https://github.com/keras-team/keras/blob/v3.3.3/keras/src/utils/numerical_utils.py#L8
    '''
    norm = np.atleast_1d(np.linalg.norm(x, order, axis))
    norm[norm == 0] = 1

    # axis cannot be `None`
    axis = axis or -1
    return x / np.expand_dims(norm, axis)
    
def cos_sim(y_true, y_pred, axis=-1):
    '''
    copied from https://github.com/keras-team/keras/blob/v3.3.3/keras/src/losses/losses.py#L1320
    '''
    y_pred = normalize(y_pred, axis=axis)
    y_true = normalize(y_true, axis=axis)
    return -np.sum(y_true * y_pred, axis=axis)

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)
