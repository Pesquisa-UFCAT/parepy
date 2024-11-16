import sys
sys.path.append('./parepy_toolbox')
from common_library import calc_pf_beta, beta_equation, fbf, sampling

import unittest
import pandas as pd

class TestCalcPfBeta(unittest.TestCase):
    def setUp(self):
        data = {'I_0': [0] * 98 + [1] * 2}
        self.df = pd.DataFrame(data)

    def test_calc_pf_beta(self):
        pf_df, beta_df = calc_pf_beta(self.df)
        expected_pf = 2 / 100 
        self.assertAlmostEqual(pf_df.iloc[0]['I_0'], expected_pf, places=4)
        expected_beta = beta_equation(expected_pf)
        self.assertAlmostEqual(beta_df.iloc[0]['I_0'], expected_beta, places=4)


class TestFBF(unittest.TestCase):
    def setUp(self):
        self.algorithm = 'MCS_TIME'
        self.n_constraints = 2
        self.time_analysis = 3

        data = {
            'I_0_t=0': [0, 0, 1],
            'I_0_t=1': [0, 1, 0],
            'I_0_t=2': [1, 0, 0],
            'I_1_t=0': [0, 1, 1],
            'I_1_t=1': [0, 0, 0],
            'I_1_t=2': [1, 0, 0]
        }
        self.results_about_data = pd.DataFrame(data)

        expected_data = {
            'I_0_t=0': [0, 0, 1],
            'I_0_t=1': [0, 1, 1],
            'I_0_t=2': [1, 1, 1],
            'I_1_t=0': [0, 1, 1],
            'I_1_t=1': [0, 1, 1],
            'I_1_t=2': [1, 1, 1]
        }
        self.expected_results = pd.DataFrame(expected_data)

    def test_fbf_with_mcs_time(self):
        processed_data = fbf(self.algorithm, self.n_constraints, self.time_analysis, self.results_about_data)
        pd.testing.assert_frame_equal(processed_data, self.expected_results)

    def test_fbf_with_different_algorithm(self):
        algorithm = 'OTHER_ALGORITHM'
        processed_data = fbf(algorithm, self.n_constraints, self.time_analysis, self.results_about_data)
        pd.testing.assert_frame_equal(processed_data, self.results_about_data)


class TestSamplingFunction(unittest.TestCase):

    def setUp(self):
        self.n_samples = 10
        self.d = 3
        self.model = {
            'model sampling': 'MCS TIME',
            'time steps': 5
        }
        self.variables_setup = [
            {
                'type': 'GAUSSIAN',
                'seed': 42,
                'stochastic variable': True,
                'loc': 0,
                'scale': 1
            },
            {
                'type': 'GUMBEL MAX',
                'seed': 42,
                'stochastic variable': True,
                'loc': 0,
                'scale': 1
            },
            {
                'type': 'WEIBULL',
                'seed': 42,
                'stochastic variable': True,
                'shape': 1.5,
                'loc': 0,
                'scale': 1
            }
        ]

    def test_sampling_shape(self):
        result = sampling(self.n_samples, self.d, self.model, self.variables_setup)
        expected_shape = (self.n_samples * self.model['time steps'], self.d + 1)
        self.assertEqual(result.shape, expected_shape)
        
if __name__ == '__main__':
    unittest.main(argv = [''], verbosity = 2)