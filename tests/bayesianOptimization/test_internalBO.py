import pickle
import numpy as np
from numpy.random import default_rng
import pickle
import unittest

from bo.gprInterface import InternalGPR
from bo.utils import Fn, compute_robustness
from bo.sampling import uniform_sampling
from bo.gprInterface import internalGPR
from bo.bayesianOptimization.internalBO import InternalBO
from matplotlib import pyplot as plt


class Test_internalBO(unittest.TestCase):
    def test1_internalBO(self):
        def internal_function(X):
            return X[0] ** 2
            # return X[0] ** 2 + X[1] ** 2 + X[2] ** 2

        rng = default_rng(12345)
        # region_support = np.array([[-1, 1], [-2, 2], [-3, 3]])
        region_support = np.array([[-1, 1]])

        func1 = Fn(internal_function)
        in_samples_1 = uniform_sampling(20, region_support, 1, rng)
        out_samples_1 = compute_robustness(in_samples_1, func1)

        gpr_model = InternalGPR()
        bo = InternalBO()

        x_complete, y_complete = bo.sample(
            func1, 50, in_samples_1, out_samples_1, region_support, gpr_model, rng
        )
        
        
        
        with open("tests\\bayesianOptimization\\goldResources\\test_1_internalBO.pickle", "rb") as f:
            # pickle.dump([x_complete, y_complete], f)
            gr_x_complete, gr_y_complete = pickle.load(f)
        

        np.testing.assert_array_equal(x_complete, gr_x_complete)
        np.testing.assert_array_equal(y_complete, gr_y_complete)


        assert x_complete.shape[0] == 70
        assert x_complete.shape[1] == 1
        assert y_complete.shape[0] == 70
        # assert y_complete.shape[1] == 1

    def test2_internalBO(self):
        def internal_function(X):
            return X[0] ** 2 + X[1] ** 2
            # return X[0] ** 2 + X[1] ** 2 + X[2] ** 2

        rng = default_rng(12345)
        # region_support = np.array([[-1, 1], [-2, 2], [-3, 3]])
        region_support = np.array([[-1, 1], [-1, 1]])

        func1 = Fn(internal_function)
        in_samples_1 = uniform_sampling(20, region_support, 2, rng)
        out_samples_1 = compute_robustness(in_samples_1, func1)

        gpr_model = InternalGPR()
        bo = InternalBO()

        x_complete, y_complete = bo.sample(
            func1, 50, in_samples_1, out_samples_1, region_support, gpr_model, rng
        )

        with open("tests\\bayesianOptimization\\goldResources\\test_2_internalBO.pickle", "rb") as f:
            # pickle.dump([x_complete, y_complete], f)
            gr_x_complete, gr_y_complete = pickle.load(f)
        

        np.testing.assert_array_equal(x_complete, gr_x_complete)
        np.testing.assert_array_equal(y_complete, gr_y_complete)

        assert x_complete.shape[0] == 70
        assert x_complete.shape[1] == 2
        assert y_complete.shape[0] == 70

    def test3_internalBO(self):
        def internal_function(X):
            return X[0] ** 2 + X[1] ** 2 + X[2] ** 2

        rng = default_rng(12345)
        region_support = np.array([[-1, 1], [-2, 2], [-3, 3]])
        # region_support = np.array([[-1, 1], [-1, 1]])

        func1 = Fn(internal_function)
        in_samples_1 = uniform_sampling(20, region_support, 3, rng)
        out_samples_1 = compute_robustness(in_samples_1, func1)

        gpr_model = InternalGPR()
        bo = InternalBO()

        x_complete, y_complete = bo.sample(
            func1, 50, in_samples_1, out_samples_1, region_support, gpr_model, rng
        )

        with open("tests\\bayesianOptimization\\goldResources\\test_3_internalBO.pickle", "rb") as f:
            # pickle.dump([x_complete, y_complete], f)
            gr_x_complete, gr_y_complete = pickle.load(f)
        

        np.testing.assert_array_equal(x_complete, gr_x_complete)
        np.testing.assert_array_equal(y_complete, gr_y_complete)

        assert x_complete.shape[0] == 70
        assert x_complete.shape[1] == 3
        assert y_complete.shape[0] == 70


if __name__ == "__main__":
    unittest.main()
