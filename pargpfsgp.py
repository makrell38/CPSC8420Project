from argparse import Namespace
import copy
import numpy as np
import tensorflow as tf
from gpflow import kernels
from gpflow.config import default_float as floatx
from gpflow.utilities import print_summary

from bayesian_algorithm_execution.bax.models.simple_gp import SimpleGp
from bayesian_algorithm_execution.bax.models.gpfs.models import PathwiseGPR
from bayesian_algorithm_execution.bax.models.gp.gp_utils import kern_exp_quad, gp_post
from bayesian_algorithm_execution.bax.util.base import Base
from bayesian_algorithm_execution.bax.util.misc_util import dict_to_namespace, suppress_stdout_stderr
from bayesian_algorithm_execution.bax.util.domain_util import unif_random_sample_domain

from bayesian_algorithm_execution.bax.models.pgps.pssgp.kernels.matern import Matern52, Matern32
from bayesian_algorithm_execution.bax.models.pgps.pssgp.kernels.rbf import RBF
from bayesian_algorithm_execution.bax.models.pgps.pssgp.model import StateSpaceGP

class ParGpfsGp(Base):

    def __init__(self, params=None, data=None, verbose=True):
        super().__init__(params, verbose)
        self.set_data(data)

    def set_params(self, params):
        """Set self.params, the parameters for this model."""
        super().set_params(params)
        params = dict_to_namespace(params)

        # Set self.params
        self.params.name = getattr(params, 'name', 'ParGpfsGp')
        self.params.n_bases = getattr(params, 'n_bases', 1000)
        self.params.n_dimx = getattr(params, 'n_dimx', 1)
        self.params.ls = getattr(params, 'ls', 3.7)
        self.params.alpha = getattr(params, 'alpha', 1.85)
        self.params.sigma = getattr(params, 'sigma', 1e-2)
        self.set_kernel(params)

    def set_kernel(self, params):
        """Set GPflow kernel."""
        self.params.kernel_str = getattr(params, 'kernel_str', 'rbf')

        ls = self.params.ls
        kernvar = self.params.alpha**2

        if self.params.kernel_str == 'rbf':
            gpf_kernel = RBF(variance=kernvar, lengthscales=ls)
            kernel = getattr(params, 'kernel', kern_exp_quad)
        elif self.params.kernel_str == 'matern52':
            gpf_kernel = Matern52(variance=kernvar, lengthscales=ls)
            kernel = getattr(params, 'kernel', kern_exp_quad)
            #raise Exception('Matern 52 kernel is not yet supported.')
        elif self.params.kernel_str == 'matern32':
            gpf_kernel = Matern32(variance=kernvar, lengthscales=ls)
            kernel = getattr(params, 'kernel', kern_exp_quad)
            #raise Exception('Matern 32 kernel is not yet supported.')

        self.params.gpf_kernel = gpf_kernel
        self.params.kernel = kernel

    def set_data(self, data):
        """Set self.data."""
        if data is None:
            # Initialize self.data to be empty
            self.data = Namespace()
            self.data.x = []
            self.data.y = []
        else:
            data = dict_to_namespace(data)
            self.data = copy.deepcopy(data)
        self.tf_data = Namespace()
        self.tf_data.x = tf.convert_to_tensor(np.array(self.data.x))
        self.tf_data.y = tf.convert_to_tensor(
            np.array(self.data.y).reshape(-1, 1)
        )
        self.set_model()

    def set_model(self):
        """Set GPFlowSampling as self.model."""
        data = (self.tf_data.x, self.tf_data.y)
        self.params.model = StateSpaceGP(
            data,
            kernel=self.params.gpf_kernel,
            noise_variance=self.params.sigma**2,
            parallel=True,
            max_parallel=1000
        )

    def initialize_function_sample_list(self, n_samp=1):
        """Initialize a list of n_samp function samples."""
        n_bases = self.params.n_bases
        paths = self.params.model.generate_paths(num_samples=n_samp, num_bases=n_bases)
        _ = self.params.model.set_paths(paths)

        Xinit = tf.random.uniform(
            [n_samp, self.params.n_dimx], minval=0.0, maxval=0.1, dtype=floatx()
        )
        Xvars = tf.Variable(Xinit)
        self.fsl_xvars = Xvars

    @tf.function
    def call_fsl_on_xvars(self, model, xvars, sample_axis=0):
        """Call fsl on fsl_xvars."""
        #print(model.predict_f(Xnew=xvars, sample_axis=sample_axis))
        fvals, _ =  model.predict_f(Xnew=xvars, sample_axis=sample_axis)
        
        return fvals

    def call_function_sample_list(self, x_list):
        """Call a set of posterior function samples on respective x in x_list."""

        # Replace Nones in x_list with first non-None value
        x_list = self.replace_x_list_none(x_list)

        # Set fsl_xvars as x_list, call fsl, return y_list
        self.fsl_xvars.assign(x_list)
        y_tf = self.call_fsl_on_xvars(self.params.model, self.fsl_xvars)
        y_list = list(np.asarray(y_tf).reshape(-1))
        #y_list = list(y_tf.numpy().reshape(-1))
        return y_list

    def replace_x_list_none(self, x_list):
        """Replace any Nones in x_list with first non-None value and return x_list."""

        # Set new_val as first non-None element of x_list
        new_val = next(x for x in x_list if x is not None)

        # Replace all Nones in x_list with new_val
        x_list_new = [new_val if x is None else x for x in x_list]

        return x_list_new

    def get_post_mu_cov(self, x_list, full_cov=False):
        """See SimpleGp. Returns a list of mu, and a list of cov/std."""
        mu_list, cov_list = self.gp_post_wrapper(x_list, self.data, full_cov)
        return mu_list, cov_list

    def gp_post_wrapper(self, x_list, data, full_cov=True):
        """See SimpleGp. Returns a list of mu, and a list of cov/std."""
        if len(data.x) == 0:
            return self.get_prior_mu_cov(x_list, full_cov)
        # If data is not empty:
        mu, cov = gp_post(
            data.x,
            data.y,
            x_list,
            self.params.ls,
            self.params.alpha,
            self.params.sigma,
            self.params.kernel,
            full_cov=full_cov,
        )

        # Return mean and cov matrix (or std-dev array if full_cov=False)
        return mu, cov
