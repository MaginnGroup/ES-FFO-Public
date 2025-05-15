import gpflow
import numpy as np
import tensorflow as tf
from tensorflow_probability import bijectors as tfb
from gpflow.utilities import print_summary
import warnings
# tf.config.run_functions_eagerly(True)
import copy

def buildGP(x_train, y_train, gpConfig, retrain = 0):

    """Create and train a GPFlow model

    Parameters
    ----------
    x_Train : numpy array (N,K)
        Training features, where N is the number of data points and K is the
        number of independent features (e.g., sigma profile bins).
    y_Train : numpy array (N,1)
        Training labels (e.g., property of a given molecule).
    gpConfig : dictionary, optional
        Dictionary containing the configuration of the GP. If a key is not
        present in the dictionary, its default value is used.
        Keys:
            . kernel : string
                Kernel to be used. One of:
                    . 'RBF' - gpflow.kernels.RBF()
                    . 'RQ' - gpflow.kernels.RationalQuadratic()
                    . 'Matern32' - gpflow.kernels.Matern32()
                    . 'Matern52' - gpflow.kernels.Matern52()
                The default is 'RBF'.
            . useWhiteKernel : boolean
                Whether to use a White kernel (gpflow.kernels.White).
                The default is True.
            . trainLikelihood : boolean
                Whether to treat the variance of the likelihood of the modeal
                as a trainable (or fitting) parameter. If False, this value is
                fixed at 10^-5.
                The default is True.
        The default is {}.

    Raises
    ------
    UserWarning
        Warning raised if the optimization (fitting) fails to converge.

    Returns
    -------
    model : gpflow.models.gpr.GPR object
        GP model.

    """
    # Unpack gpConfig
    kernel=gpConfig.get('kernel','RBF')
    useWhiteKernel=gpConfig.get('useWhiteKernel','True')
    trainLikelihood=gpConfig.get('trainLikelihood','True')
    typeMeanFunc=gpConfig.get('mean_function','Zero')
    anisotropy=gpConfig.get('anisotropic','True')
    
    #Get hyperparameters
    hypers = get_init_hypers(retrain, x_train, anisotropy= anisotropy)
    lengthscale_, variance_, alpha_, white_var = hypers
    # print("lengthscale_", lengthscale_)
    # print("variance_", variance_)
    # print("alpha_", alpha_)
    # print("white_var", white_var)
    # Select and initialize kernel
    if kernel=='RBF':
        gpKernel=gpflow.kernels.SquaredExponential(variance=variance_, lengthscales=lengthscale_)
    elif kernel=='RQ':
        gpKernel=gpflow.kernels.RationalQuadratic(variance=variance_, lengthscales=lengthscale_, alpha=alpha_)
    elif kernel=='Matern32':
        gpKernel=gpflow.kernels.Matern32(variance=variance_, lengthscales=lengthscale_)
    elif kernel=='Matern52':
        gpKernel=gpflow.kernels.Matern52(variance=variance_, lengthscales=lengthscale_)
    else:
        raise ValueError('Invalid kernel type')
    
    # Add White kernel
    if useWhiteKernel: 
        gpKernel=gpKernel+gpflow.kernels.White(variance = white_var)
            
    # Add Mean function
    if typeMeanFunc == 'Zero':
        mf = None
    elif typeMeanFunc == 'Linear':
        mf = gpflow.mean_functions.Linear(
                A=np.zeros(x_train.shape[1]).reshape(-1, 1)
            )
    else:
        raise ValueError('Invalid mean function type')

    # Build GP model    
    model=gpflow.models.GPR((x_train,y_train.reshape(-1,1)),gpKernel,mean_function=mf, noise_variance=10**-5)
    # model_pretrain = copy.deepcopy(model)
    # print(gpflow.utilities.print_summary(model_pretrain))
    condition_number = np.linalg.cond(model.kernel(x_train))
    # Select whether the likelihood variance is trained
    gpflow.utilities.set_trainable(model.likelihood.variance,trainLikelihood)
    # Build optimizer
    optimizer=gpflow.optimizers.Scipy()

    # Fit GP to training data
    aux=optimizer.minimize(model.training_loss,
                           model.trainable_variables,
                           options={'maxiter':10**9},
                           method="L-BFGS-B")
    train_loss = model.training_loss().numpy()
    # print(gpflow.utilities.print_summary(model))
    return model, aux, condition_number

def get_init_hypers(retrain, x_train, anisotropy):
    lenscl_bnds = [0.00001, 1000.0]
    var_bnds = [0.00001, 100.0]
    alpha_bnds = [0.0001, 3000.0]
    white_var_bnds = [0.00001, 10.0]

    if retrain == 0:
        lengthscale_1 = bounded_parameter(lenscl_bnds[0], lenscl_bnds[1], 1.0)
        if anisotropy == True:
            lengthscale_ = np.ones(x_train.shape[1])*lengthscale_1
        else:
            lengthscale_ = lengthscale_1
        variance_ = bounded_parameter(var_bnds[0], var_bnds[1], 1.0)
        alpha_ = bounded_parameter(alpha_bnds[0], alpha_bnds[1], 1.0)
        white_var = 1.0
    else:
        seed_ = int(retrain)
        np.random.seed(seed_)
        tf.random.set_seed(seed_)

        if anisotropy == True:
            initial_values = np.array(np.random.uniform(0.1, 100.0, x_train.shape[1]), dtype='float64')
            lengthscale_ = bounded_parameter(lenscl_bnds[0], lenscl_bnds[1], initial_values)
        else:
            lengthscale_ = bounded_parameter(lenscl_bnds[0], lenscl_bnds[1], 
                                             np.array(np.random.uniform(0.1, 100.0), dtype='float64'))
        variance_ = bounded_parameter(var_bnds[0], var_bnds[1], np.array(np.random.lognormal(0.0, 1.0), dtype='float64'))
        alpha_ = bounded_parameter(alpha_bnds[0], alpha_bnds[1], np.array(np.random.uniform(0.01, 100), dtype='float64'))
        white_var = bounded_parameter(white_var_bnds[0], white_var_bnds[1], np.array(np.random.uniform(0.05, 10), dtype='float64'))

    return lengthscale_, variance_, alpha_, white_var

def bounded_parameter(low, high, initial_value):
    sigmoid = tfb.Sigmoid(low=tf.cast(low, dtype=tf.float64), high=tf.cast(high, dtype=tf.float64))
    return gpflow.Parameter(initial_value, transform=sigmoid, dtype=tf.float64)

def run_gpflow_scipy(x_train, y_train, gpConfig, restarts = 1):
    """Create and train a GPFlow model

    x_Train : numpy array (N,K)
        Training features, where N is the number of data points and K is the
        number of independent features (e.g., sigma profile bins).
    y_Train : numpy array (N,1)
        Training labels (e.g., property of a given molecule).
    gpConfig : dictionary, optional
        Dictionary containing the configuration of the GP. If a key is not
        present in the dictionary, its default value is used.
        Keys:
            . kernel : string
                Kernel to be used. One of:
                    . 'RBF' - gpflow.kernels.RBF()
                    . 'RQ' - gpflow.kernels.RationalQuadratic()
                    . 'Matern32' - gpflow.kernels.Matern32()
                    . 'Matern52' - gpflow.kernels.Matern52()
                The default is 'RBF'.
            . useWhiteKernel : boolean
                Whether to use a White kernel (gpflow.kernels.White).
                The default is True.
            . trainLikelihood : boolean
                Whether to treat the variance of the likelihood of the modeal
                as a trainable (or fitting) parameter. If False, this value is
                fixed at 10^-5.
                The default is True.
        The default is {}.
    restarts : int, optional
        Number of restarts for the optimization. 
        The default is 1.
    """
    # Train the model multiple times and keep track of the model with the lowest minimum training loss
    best_minimum_loss = float('inf')
    best_model = None
    first_model = None
    best_model_success = False

    #While you still have retrains left
    for i in range(restarts):
        model_params = buildGP(x_train, y_train, gpConfig, retrain = i)
        gp_model, aux, condition_number = model_params
        # Compute the training loss of the model
        training_loss = gp_model.training_loss().numpy()
        #If we succeeded in training the model
        if aux.success:
            # Check if this model has the best minimum training loss
            if training_loss < best_minimum_loss:
                #If it does, set success to true and save the model
                best_minimum_loss = training_loss
                best_model = gp_model
                best_model_success = True
        #Otherwise if this is the 1st iteration and it failed, save the model
        #This model will be used as a backup in the case that all models fail
        elif aux.success == False and i == 0:
            first_model = gp_model
            first_minimum_loss = training_loss
            
    #If we have no good models, use the first model
    if best_model is None:
        best_model = first_model
        best_minimum_loss = first_minimum_loss
        warnings.warn('GP optimizer failed to converge after' + str(restarts) + ' attempts. Using the first model as a backup.')

    # gpflow.utilities.print_summary(best_model)
    return best_model

# def run_gpflow_scipy(x_train, y_train, kernel, mean_function="linear", fmt="notebook", seed= None, restarts = 1):
#     """Create and train a GPFlow model

#     Parameters
#     ----------
#     x_train : np.ndarray, shape=(n_samples, n_parameters)
#         The x training data
#     y_train : np.ndarray, shape=(n_samples, 1)
#         The y training data
#     kernel : string
#         Kernel to use for the GP model
#     mean_function: string or None, default = "linear"
#         Type of mean function for the GP model
#         Options are "linear", or None
#     fmt : string, optional, default="notebook"
#         The formatting type for the GPFlow print_summary
#     """
#     if seed != None:
#         np.random.seed(seed)
#         tf.compat.v1.get_default_graph()
#         tf.compat.v1.set_random_seed(seed)
#         tf.random.set_seed(seed)

#     if mean_function is not None:
#         if mean_function == "linear":
#             mean_function = gpflow.mean_functions.Linear(
#                 A=np.zeros(x_train.shape[1]).reshape(-1, 1)
#             )
#         elif mean_function.lower() == "none":
#             mean_function = None
#         else:
#             raise ValueError(
#                 "Only supported mean functions are 'linear' and 'none'"
#             )

#     # Create the model
#     model = gpflow.models.GPR(
#         data=(x_train, y_train.reshape(-1, 1)),
#         kernel=kernel,
#         mean_function=mean_function
#     )

#     # Print initial values
#     # print_summary(model, fmt=fmt)

#     # Optimize model with scipy
#     optimizer = gpflow.optimizers.Scipy()
#     optimizer.minimize(model.training_loss, model.trainable_variables)

#     # Print the optimized values
#     # print_summary(model, fmt="notebook")

#     # Return the model
#     return model