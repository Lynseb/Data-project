from types import SimpleNamespace
import numpy as np
from scipy import optimize

class ConsumerClass:
    """ a consumer with nested CES preferences over three goods """

    def __init__(self, par=None):
        self.setup()
        if par is not None:
            for k, v in par.items():
                self.par.__dict__[k] = v

    def setup(self):
        par = self.par = SimpleNamespace()
        self.sol = SimpleNamespace()

        # a. Præferencevægte
        par.alpha = 0.60  # Vægt på mad
        par.beta = 0.50   # Vægt på bus

        # b. Substitution
        par.sigma_A = 0.80  # Mellem mad og transport (øvre nest)
        par.sigma_B = 0.40  # Mellem bus og tog (nedre nest)

        # c. Priser og indkomst
        par.p1 = 1.0
        par.p2 = 1.0
        par.p3 = 1.5
        par.I = 10.0

        # d. Numeriske indstillinger
        par.s_min = 1e-12

    def __str__(self):
        par = self.par
        lines = ['ConsumerClass']
        lines.append(f'  alpha = {par.alpha:.4f}, beta = {par.beta:.4f}')
        lines.append(f'  sigma_A = {par.sigma_A:.4f}, sigma_B = {par.sigma_B:.4f}')
        lines.append(f'  p1 = {par.p1:.4f}, p2 = {par.p2:.4f}, p3 = {par.p3:.4f}')
        lines.append(f'  I = {par.I:.4f}')
        return '\n'.join(lines)

    # 1. CES Nest
    def ces(self, z1, z2, w, sigma):
        par = self.par
        assert not np.isclose(sigma, 1.0), 'sigma = 1 giver rho = 0'

        z1 = np.maximum(z1, par.s_min)
        z2 = np.maximum(z2, par.s_min)
        rho = 1.0 - 1.0 / sigma
        return (w * z1**rho + (1.0 - w) * z2**rho)**(1.0 / rho)

    def utility(self, x1, x2, x3):
        par = self.par
        # Kombiner gode 2 og 3 til transport
        travel = self.ces(x2, x3, par.beta, par.sigma_B)
        # Kombiner mad og transport til samlet nytte
        u = self.ces(x1, travel, par.alpha, par.sigma_A)
        return u

    # 2. Budgetandele og mængder
    def shares(self, s1, w):
        return s1, (1.0 - s1) * w, (1.0 - s1) * (1.0 - w)

    def quantities(self, s1, w):
        par = self.par
        s1, s2, s3 = self.shares(s1, w)
        return s1 * par.I / par.p1, s2 * par.I / par.p2, s3 * par.I / par.p3

    def value_of_choice(self, s1, w):
        x1, x2, x3 = self.quantities(s1, w)
        return self.utility(x1, x2, x3)

    def objective(self, s):
        return -self.value_of_choice(s[0], s[1])

    # 3. Løsningsmetoder
    def solve_grid(self, N=200, do_print=True):
        opt = SimpleNamespace()

        # a. Gitre over enhedskvadratet
        s1_vec = np.linspace(0.0, 1.0, N)
        w_vec = np.linspace(0.0, 1.0, N)
        opt.s1_grid, opt.w_grid = np.meshgrid(s1_vec, w_vec, indexing='ij')

        # b. Beregn nytte i alle punkter
        x1, x2, x3 = self.quantities(opt.s1_grid, opt.w_grid)
        opt.u_grid = self.utility(x1, x2, x3)

        # c. Find maksimum
        idx = np.unravel_index(np.argmax(opt.u_grid), opt.u_grid.shape)
        opt.s1 = opt.s1_grid[idx]
        opt.w = opt.w_grid[idx]
        opt.s1, opt.s2, opt.s3 = self.shares(opt.s1, opt.w)
        opt.u = opt.u_grid[idx]

        if do_print:
            print(f"Grid search (N={N}): s1={opt.s1:.4f}, s2={opt.s2:.4f}, s3={opt.s3:.4f}, u={opt.u:.4f}")

        return opt

    def solve(self, s0=None, do_print=True, **kwargs):
        opt = SimpleNamespace()

        if s0 is None:
            s0 = np.array([0.5, 0.5])
        s0 = np.asarray(s0, dtype=float)

        path = [s0.copy()]
        bounds = ((0.0, 1.0), (0.0, 1.0))

        res = optimize.minimize(
            self.objective,
            s0,
            method='L-BFGS-B',
            bounds=bounds,
            callback=lambda x: path.append(x.copy()),
            **kwargs
        )

        opt.s1, opt.w = res.x
        opt.s1, opt.s2, opt.s3 = self.shares(opt.s1, opt.w)
        opt.u = -res.fun
        opt.path = np.array(path)
        opt.res = res

        if do_print:
            print(f"L-BFGS-B: s1={opt.s1:.4f}, s2={opt.s2:.4f}, s3={opt.s3:.4f}, w={opt.w:.4f}, u={opt.u:.4f}")

        return opt