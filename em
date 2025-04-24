import numpy as np
from scipy.stats import norm

def expectation_maximization(X, num_clusters=2, max_iters=100, tol=1e-4):
   
    np.random.seed(42)
    mu = np.random.choice(X, num_clusters)  
    sigma = np.full(num_clusters, np.var(X))  
    pi = np.full(num_clusters, 1 / num_clusters) 

    X = np.array(X) 

    for iteration in range(max_iters):
        
        responsibilities = np.zeros((len(X), num_clusters))
        for j in range(num_clusters):
            responsibilities[:, j] = pi[j] * norm.pdf(X, mu[j], np.sqrt(sigma[j]))
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)  

        
        N_k = responsibilities.sum(axis=0) 
        new_mu = np.sum(responsibilities * X[:, np.newaxis], axis=0) / N_k
        new_sigma = np.sum(responsibilities * (X[:, np.newaxis] - new_mu) ** 2, axis=0) / N_k
        new_pi = N_k / len(X)

       
        if np.allclose(mu, new_mu, atol=tol) and np.allclose(sigma, new_sigma, atol=tol):
            break

        mu, sigma, pi = new_mu, new_sigma, new_pi  
    return mu, sigma, pi, responsibilities


X = [2.0, 2.2, 1.8, 6.0, 5.8, 6.2]


mu_final, sigma_final, pi_final, responsibilities_final = expectation_maximization(X)


print(f"Final Means: {mu_final}")
print(f"Final Variances: {sigma_final}")
print(f"Final Mixing Coefficients: {pi_final}")
print("Final Responsibilities:")
print(responsibilities_final)
