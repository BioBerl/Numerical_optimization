import unittest
from unconstrained_min import LineSearch
#from utils import *
#from examples import *



class TestLineSearch(unittest.TestCase):

    # def test_q1(self):
    #     print("worked!")

    def run_quadratic_test(self, f, q_num, calc_hessian, step_len=None, wolfe_c1=0.01):
        lsearch = LineSearch(verbose=True, wolfe_c1=wolfe_c1, calc_hessian=calc_hessian, max_iter=10000)
        opt_type = "Hessian" if calc_hessian else "Gradient Descent"
        wolfe_text = " Wolfe" if wolfe_c1 > 0 else ""
        print("\nMinimizing quadratic function {} with {}{}:\nQ={}\n"
              .format(q_num, opt_type, wolfe_text, f.q2) + "=" * 50)
        
        res = lsearch.minimize(f=f, x0=[1, 1], step_len=step_len, print_every=1 if calc_hessian else None)
        print(res, flush=True)
        self.assertLess(res['objective'], 1e-6)
        self.assertLess(np.linalg.norm(res['location']), 1e-4)
        if calc_hessian:
            self.assertEqual(res['num_iter'], 2)
        return lsearch


if __name__ == '__main__':
    unittest.main()