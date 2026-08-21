from types import SimpleNamespace
import numpy as np
from scipy import optimize
from Consumer import ConsumerClass


class GovernmentClass(ConsumerClass):
    """a government raising revenue from the consumer in Consumer.py"""

    def __init__(self, par=None):
        # a. initialiser basis-parametre fra ConsumerClass
        self.setup()
        
        # b. initialiser skatteparametre
        self.setup_government()

        # c. opdater parametre hvis custom dictionary gives
        if par is not None:
            for k, v in par.items():
                self.par.__dict__[k] = v

        # d. synkroniser før-skat priser og indkomst
        self.sync_pre_tax()

    def setup_government(self):
        """ initialiserer skatteinstrumenter til 0 """
        par = self.par
        par.T = 0.0
        par.tau1 = 0.0
        par.tau2 = 0.0
        par.tau3 = 0.0

    def sync_pre_tax(self):
        """ gemmer før-skat værdier som referencepunkter """
        par = self.par
        par.p1_pre = par.p1
        par.p2_pre = par.p2
        par.p3_pre = par.p3
        par.I_pre = par.I

    def set_taxes(self, T=0.0, tau1=0.0, tau2=0.0, tau3=0.0):
        """ opdaterer skattesatser og de resulterende forbrugerpriser og indkomst """
        par = self.par
        par.T = T
        par.tau1 = tau1
        par.tau2 = tau2
        par.tau3 = tau3

        # Forbrugerpriser efter skat
        par.p1 = (1.0 + tau1) * par.p1_pre
        par.p2 = (1.0 + tau2) * par.p2_pre
        par.p3 = (1.0 + tau3) * par.p3_pre
        
        # Disponibel indkomst efter lump-sum skat
        par.I = par.I_pre - T

    def tax_revenue(self, opt=None):
        """
        Beregner det samlede skatteprovenu (Ligning 5):
        R = T + sum_{j=1}^3 tau_j * p_j^{pre} * x_j^*
        """
        par = self.par
        if opt is None:
            opt = self.solve(do_print=False)

        # Mængder forbrugt under gældende priser og indkomst
        x1, x2, x3 = self.quantities(opt.s1, opt.w)

        # Skatteindtægt fra vareskatter
        rev_food = par.tau1 * par.p1_pre * x1
        rev_bus = par.tau2 * par.p2_pre * x2
        rev_train = par.tau3 * par.p3_pre * x3

        # Samlet provenu R
        R = par.T + rev_food + rev_bus + rev_train
        return R

    def revenue_and_utility(self, tau, goods=(2,)):
        """ evaluerer provenu og nytte ved en given vareskat tau på en delmængde af goder """
        tau1 = tau if 1 in goods else 0.0
        tau2 = tau if 2 in goods else 0.0
        tau3 = tau if 3 in goods else 0.0

        self.set_taxes(T=0.0, tau1=tau1, tau2=tau2, tau3=tau3)
        opt = self.solve(do_print=False)
        R = self.tax_revenue(opt=opt)
        u = opt.u
        return R, u

    def revenue_and_utility_lump_sum(self, T):
        """ evaluerer provenu og nytte ved en lump-sum skat T """
        self.set_taxes(T=T, tau1=0.0, tau2=0.0, tau3=0.0)
        opt = self.solve(do_print=False)
        R = self.tax_revenue(opt=opt)
        u = opt.u
        return R, u

    def max_revenue(self, goods=(2,), tau_max=10.0, N=1001):
        """ finder toppunktet på Laffer-kurven for et givet skatteinstrument """
        tau_grid = np.linspace(0.0, tau_max, N)
        rev_grid = np.zeros(N)

        for i, tau in enumerate(tau_grid):
            R, _ = self.revenue_and_utility(tau=tau, goods=goods)
            rev_grid[i] = R

        best_idx = np.argmax(rev_grid)
        return tau_grid[best_idx], rev_grid[best_idx]

    def find_tax_rate(self, R_target, goods=(2,), bracket=(1e-10, 1.0)):
        """ finder den præcise skattesats tau der leverer et målsat provenu R_target """
        def objective(tau):
            R, _ = self.revenue_and_utility(tau=tau, goods=goods)
            return R - R_target

        try:
            res = optimize.root_scalar(objective, bracket=bracket, method="brentq")
            return res.root
        except (ValueError, RuntimeError):
            return np.nan