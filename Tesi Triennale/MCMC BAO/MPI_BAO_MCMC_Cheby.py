import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Define constants
c = 299792.458  # Speed of light in km/s

# MCMC parameters
initial_theta = np.array([10000, 0.32, 1, 0,0,0])  # Initial guess for parameters (H0rd, Omega_m, c1,c2,c3,c4)
target_accepted = 80000  # Number of MCMC accepted iterations
adapt_interval = 1000 # N.iteration after whic print acceptance rate

###############################################################################à
# Define the equation of state w(z) for the w0-wa parametrization

def x(z):
    z_max = 3.5
    return 1 -2*(z_max-z)/(z_max)
   
def w(z,c1,c2,c3,c4):
    return -1*(c1+c2*x(z)+c3*(2*x(z)**2-1)+c4*(4*x(z)**3-3*x(z)))

# Define the integrand for f_DE(z)
def integrand(z,c1,c2,c3,c4):
    return 3 * (1 + w(z,c1,c2,c3,c4)) / (1 + z)

# Define the function to compute f_DE(z) using numerical integration
def f_de(z,c1,c2,c3,c4):
    integral, _ = quad(integrand, 0, z, args=(c1,c2,c3,c4))
    return np.exp(integral)   

s=9   

#################################################################################

# Define the Hubble parameter H(z) based on the Friedmann equation
def hubble_parameter(z, omega_m0, omega_r0, hrdrag, c1,c2,c3,c4):
    # Omega_de0 from the flat universe condition
    omega_de0 = 1 - omega_m0 - omega_r0
    # Hubble parameter H(z)
    return np.sqrt(omega_m0 * (1 + z)**3 + omega_r0 * (1 + z)**4 + omega_de0 * f_de(z, c1,c2,c3,c4))

# Calculate D_H(z) / r_d based on Eq. 2.8
def D_H_over_rd(z, hrdrag, omega_m0, omega_r0, c1,c2,c3,c4):
    hz = hubble_parameter(z, omega_m0, omega_r0, hrdrag, c1,c2,c3,c4) * hrdrag 
    return c / hz

# Calculate D_M(z) / r_d based on Eq. 2.9
def D_M_over_rd(z, hrdrag, omega_m0, omega_r0,  c1,c2,c3,c4):
    def integrand(z_prime):
        return 1 / hubble_parameter(z_prime, omega_m0, omega_r0, hrdrag, c1,c2,c3,c4)
    
    integral, _ = quad(integrand, 0, z)
    return c * integral / hrdrag


# Calculate D_V(z) based on D_H and D_M
def D_V_over_rd(z, hrdrag, omega_m0, omega_r0, c1,c2,c3,c4):
    DH = D_H_over_rd(z, hrdrag, omega_m0, omega_r0, c1,c2,c3,c4)
    DM = D_M_over_rd(z, hrdrag, omega_m0, omega_r0, c1,c2,c3,c4)
    return (z * DM**2 * DH)**(1/3)

# Define the likelihood function
def likelihood(theta, data, quantity_array, covariance):
    # Unpack parameters
    hrdrag, omega_m, c1,c2,c3,c4 = theta
    
    # Get model predictions based on the parameters for each z
    model_prediction,dev = model(hrdrag, omega_m, c1,c2,c3,c4, data, quantity_array)
    hyper = theta[2:6]
    hyper_prior = np.sum(-0.5 /s * (hyper - np.array([1, 0, 0, 0]))**2)  # Gaussian prior
    # Calculate the likelihood (assuming Gaussian)
    diff = data[:, 1] - model_prediction
    
    return -0.5 * np.dot(diff.T, np.linalg.solve(covariance, diff)) + hyper_prior #−0.5⋅diff^T⋅cov^(−1)⋅diff + ...

# Define your model function
def model(hrdrag, omega_m, c1,c2,c3,c4, data, quantity_array):
    # Initialize an array to hold model predictions
    predictions = np.zeros(data.shape[0])
    
    omega_r0 = 0  # Radiation density, can be updated as needed
    
    for i in range(data.shape[0]):
        z = data[i, 0]  # Redshift value
        quantity = quantity_array[i]  # Use the quantity from the separate array
        
        # Compute D_H, D_M, D_V based on the quantity
        if quantity == 'DH_over_rs':
            predictions[i] = D_H_over_rd(z, hrdrag, omega_m, omega_r0, c1,c2,c3,c4)
        elif quantity == 'DM_over_rs':
            predictions[i] = D_M_over_rd(z, hrdrag, omega_m, omega_r0, c1,c2,c3,c4)
        elif quantity == 'DV_over_rs':
            predictions[i] = D_V_over_rd(z, hrdrag, omega_m, omega_r0, c1,c2,c3,c4)
        else:
            raise ValueError(f"Unknown quantity: {quantity}")
    
    return predictions
        
# Load data and covariance matrix from files
def load_data_and_covariance(data_path, cov_path):
    # Load data, treating the first two columns as floats and the third as strings
    data = np.loadtxt(data_path, dtype={'names': ('z', 'value', 'quantity'), 'formats': ('f4', 'f4', 'U20')}, comments='#')
    covariance = np.loadtxt(cov_path)
    
    # Create a numerical data array (first two columns)
    data_array = np.zeros((data.shape[0], 2))
    data_array[:, 0] = data['z']
    data_array[:, 1] = data['value']
    
    # Create a separate array for the quantity strings
    quantity_array = data['quantity']
    
    return data_array, quantity_array, covariance

# Paths to the data and covariance matrix
data_path = '/home/domenicogiardino/Documents/cobaya_project/cobaya_env/cobaya/cobaya/data/bao_data/desi_2024_gaussian_bao_ALL_GCcomb_mean.txt'
cov_path = '/home/domenicogiardino/Documents/cobaya_project/cobaya_env/cobaya/cobaya/data/bao_data/desi_2024_gaussian_bao_ALL_GCcomb_cov.txt'

# Load data and covariance matrix
data, quantity_array, covariance = load_data_and_covariance(data_path, cov_path)

#w proposal
proposal_scale = np.array([[ 1.86995815e+04, -8.48095466e-01,  1.96398785e+01, -5.22548102e+00, -3.51952351e+01, -3.82757798e+01],
                   [-8.48095466e-01,  2.15630344e-04,  2.65775205e-03,  7.82907317e-04, -4.26284950e-03, -1.97848945e-03],
                   [ 1.96398785e+01,  2.65775205e-03,  2.03899678e-01,  4.53271806e-02, -3.42767148e-01, -2.64638470e-01],
                   [-5.22548102e+00,  7.82907317e-04,  4.53271806e-02,  3.02572636e-02, -5.16580130e-02, -4.09145827e-02],
                   [-3.51952351e+01, -4.26284950e-03, -3.42767148e-01, -5.16580130e-02,  6.20266718e-01,  4.73761818e-01],
                   [-3.82757798e+01, -1.97848945e-03, -2.64638470e-01, -4.09145827e-02,  4.73761818e-01,  3.83458696e-01]]);


from mpi4py import MPI
import numpy as np

def mcmc_sampling(data, quantity_array, covariance, initial_theta, proposal_scale, target_accepted, adapt_interval):
    theta = np.random.multivariate_normal(initial_theta, proposal_scale)  # Adjust rescaling
    samples = []  # Initialize as a list
    acceptance_count = 0
    accepted_count = 0
    iteration = 0

    while (accepted_count < target_accepted):
    

        # Propose a new sample
        theta_prime = np.random.multivariate_normal(theta, proposal_scale)
        iteration += 1
        # Apply priors: 
        if not (0.01 <= theta_prime[1] <= 0.98 and 3650 <= theta_prime[0] <= 18250):
            samples.append(theta)  # Append the current sample
            continue  # Skip if proposal is out of bounds
        # Calculate likelihoods (local to each process)
        log_likelihood_current = likelihood(theta, data, quantity_array, covariance)
        log_likelihood_proposed= likelihood(theta_prime, data, quantity_array, covariance)

        # Calculate acceptance ratio
        acceptance_ratio = np.exp(log_likelihood_proposed - log_likelihood_current)

        if np.random.rand() < acceptance_ratio:
            theta = theta_prime
            acceptance_count += 1
            accepted_count += 1
        samples.append(theta)  # Append the current sample
        # Adapt the proposal scale (only after a certain number of accepted samples)
        if (accepted_count % adapt_interval == 0) and (acceptance_count > 0):
            acceptance_rate = acceptance_count / iteration
            
            print(f"Process {rank}: Accepted {accepted_count} samples. Current acceptance rate = {acceptance_rate:.3f}")
            acceptance_count = 0
            iteration = 0
            if accepted_count >= 100:
                # Calculate covariance of the samples for adjusting proposal scale
                variances = np.cov(np.array(samples), rowvar=False)  # Convert to array for covariance calculation
                proposal_scale = variances * (1)  # Example scaling, adjust multiplier as needed
            # Write local samples to a file (for each MPI process)
            if rank == 0:
                np.savetxt("samples/chain.1.txt", samples, header="H0rd Omega_m c1 c2 c3 c4",   fmt='%15.8f', delimiter=' ')
            elif rank == 1:
                np.savetxt("samples/chain.2.txt", samples, header="H0rd Omega_m c1 c2 c3 c4",   fmt='%15.8f', delimiter=' ')
            elif rank == 2:
                np.savetxt("samples/chain.3.txt", samples, header="H0rd Omega_m c1 c2 c3 c4",   fmt='%15.8f', delimiter=' ')
            elif rank == 3:
                np.savetxt("samples/chain.4.txt", samples, header="H0rd Omega_m c1 c2 c3 c4",   fmt='%15.8f', delimiter=' ')
            # Calculate the Gelman-Rubin statistic

    return np.array(samples)  # Convert to NumPy array before returning

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # Get current process rank
size = comm.Get_size()  # Get number of processes

# Each process runs its own MCMC sampling
local_samples = mcmc_sampling(data, quantity_array, covariance, initial_theta, proposal_scale, target_accepted // size, adapt_interval)


# Gather the samples from all processes on the root process
all_samples = comm.gather(local_samples, root=0)

if rank == 0:
    # Define the file paths for the chains
    files = [
        'samples/chain.1.txt',
        'samples/chain.2.txt',
        'samples/chain.3.txt',
        'samples/chain.4.txt',
    ]

    # Step 1: Load the data into arrays
    chains = []
    for file in files:
        data_printed = np.loadtxt(file, skiprows=1)  # Skip the header
        chains.append(data_printed)

    # Step 2: Calculate the Gelman-Rubin statistic
    def gelman_rubin(chains):
        num_chains = 4
        num_samples = len(chains[0])  # sample x chain

        # Calculate means of each chain
        chain_means = np.array([np.mean(chain, axis=0) for chain in chains])
        overall_mean = np.mean(chain_means, axis=0)

        # Between-chain variance
        B = np.sum((chain_means - overall_mean) ** 2, axis=0) / (num_chains - 1)

        # Within-chain variance
        W = np.mean([np.var(chain, ddof=1, axis=0) for chain in chains])

        # Gelman-Rubin statistic
        R_hat = np.sqrt((((num_samples - 1) / num_samples * W + B) / W))

        return R_hat

    # Calculate and print the Gelman-Rubin statistic
    R_m1 = np.max(gelman_rubin(chains)) - 1
    print(f"Gelman-Rubin statistic (R-1): {round(R_m1, 3)}")

    from getdist import MCSamples
    names = ['H0rd', 'Omega_m', 'c1', 'c2','c3','c4']
    samples = MCSamples(samples=chains, names=names)
    for i, name in enumerate(names):
        mean = samples.mean(names[i])
        std = round(samples.std(names[i]), 3)
        print("", samples.getInlineLatex(names[i], limit=1), f"std is : {std}")  # lim 1 68, lim 2 95


