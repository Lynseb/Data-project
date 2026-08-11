import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def simulate_lifecycle(seed=23, N=50000):
    """
    Simulates the life-cycle income model from age 18 to 65.
    
    Returns a dictionary containing income, human capital, employment status,
    and initial parameters for analysis.
    """
    # Initialize random number generator for reproducibility
    rng = np.random.default_rng(seed)
    T = 65 - 18  # 47 years (from age 18 to 65)

    # Parameters from the model specification
    p_e = [0.40, 0.35, 0.25]  # Education probabilities [short, medium, long]
    S_e = [1, 3, 5]  # Years of education for each track
    h_e0 = [1.00, 1.20, 1.55]  # Initial human capital for each education level
    Delta_e = [0.010, 0.020, 0.030]  # Growth rate of human capital for each education level
    delta = 0.06  # Depreciation rate of human capital when unemployed
    sigma_psi = 0.10  # Standard deviation of the lognormal shock to human capital
    lambda_p = 0.60  # Job-finding probability for unemployed individuals
    sigma_p = 0.05  # Job-separation (layoff) probability for employed individuals

    y_SU = 0.45  # Student grant (income during education)
    rho = 0.60  # Unemployment benefit replacement rate (60% of previous wage)
    y_underline = 0.35  # Minimum benefit floor (social assistance for never-employed)

    # 1. Randomly assign education levels to individuals based on probabilities p_e
    edu_idx = rng.choice([0, 1, 2], size=N, p=p_e)

    # Map education-specific parameters to each individual based on their assigned education level
    S_i = np.array([S_e[i] for i in edu_idx])
    h_0_i = np.array([h_e0[i] for i in edu_idx])
    Delta_i = np.array([Delta_e[i] for i in edu_idx])

    # 2. Matrices to store income, human capital, and employment status for each individual over time (N x T)
    income = np.zeros((N, T))
    h = np.zeros((N, T))
    employed = np.zeros((N, T), dtype=bool)

    # Arrays to track the last job income and whether an individual has ever held a job (for benefit calculation)
    last_job_income = np.zeros(N)
    has_been_employed = np.zeros(N, dtype=bool)

    # 3. Main simulation loop running year by year (t = 0 corresponds to age 18, t = 46 to age 64)
    for t in range(T):
        # Draw a mean-one lognormal shock to human capital for each individual
        # The term -0.5 * sigma_psi**2 corrects the mean of the lognormal distribution to exactly 1.0
        psi = rng.lognormal(-0.5 * sigma_psi**2, sigma_psi, size=N)

        # -----------------------------------------------------------------------------
        # STATE 1: Individuals currently in education (t < S_i)
        # Income equals student grant y_SU, human capital remains unchanged at initial h_0,
        # and individuals are outside the labor market (employed = False).
        # -----------------------------------------------------------------------------
        in_edu = t < S_i
        income[in_edu, t] = y_SU
        h[in_edu, t] = h_0_i[in_edu]
        employed[in_edu, t] = False

        # -----------------------------------------------------------------------------
        # STATE 2: First year on the labor market after graduation (t == S_i)
        # Everyone enters the labor market as unemployed (employed = False) with initial
        # education-specific human capital h_0 intact (no growth or depreciation yet).
        # Since they have never held a job, they receive the social assistance floor y_underline.
        # -----------------------------------------------------------------------------
        entering = t == S_i
        if np.any(entering):
            income[entering, t] = y_underline  # Benefit floor for never-employed
            h[entering, t] = h_0_i[entering]
            employed[entering, t] = False  # Enters as unemployed

        # -----------------------------------------------------------------------------
        # STATE 3: Active on the labor market in subsequent years (t > S_i)
        # Employment transitions follow a persistent Markov chain, human capital evolves,
        # and income is assigned based on current employment status.
        # -----------------------------------------------------------------------------
        on_market = t > S_i
        if np.any(on_market):
            # Fetch previous year's employment status for active market participants
            prev_emp = employed[on_market, t - 1]

            # Draw random uniform values between 0 and 1 to determine Markov transition outcomes
            draws = rng.random(size=np.sum(on_market))

            # Markov transition logic:
            # - If previously employed (prev_emp == True): Remains employed if draw > sigma_p (95% chance to stay employed).
            # - If previously unemployed (prev_emp == False): Finds job if draw < lambda_p (60% chance to find job).
            now_emp = np.where(prev_emp, draws > sigma_p, draws < lambda_p)
            employed[on_market, t] = now_emp

            # Fetch previous human capital, education growth rate, and current shock
            prev_h = h[on_market, t - 1]
            delta_val = Delta_i[on_market]
            psi_val = psi[on_market]

            # Update human capital:
            # - If employed: Grows by education-specific rate (1 + Delta_e) multiplied by shock psi.
            # - If unemployed: Depreciates by rate (1 - delta) multiplied by shock psi.
            h_new = np.where(
                now_emp,
                prev_h * (1 + delta_val) * psi_val,
                prev_h * (1 - delta) * psi_val,
            )
            # Store updated human capital in the matrix (acts as prev_h for year t + 1)
            h[on_market, t] = h_new

            # Update employment history and reference wage for currently employed individuals
            # This saves their wage to calculate unemployment benefits in case of future layoffs
            emp_indices = np.where(on_market)[0][now_emp]
            last_job_income[emp_indices] = h_new[now_emp]
            has_been_employed[emp_indices] = True

            # Calculate unemployment benefits for active individuals:
            # - If previously employed: Receives replacement rate rho (60%) of last job income.
            # - If never employed before: Receives social assistance benefit floor y_underline (0.35).
            unemp_benefit = np.where(
                has_been_employed[on_market],
                rho * last_job_income[on_market],
                y_underline,
            )

            # Assign final income for the current year:
            # Full wage (h_new) if currently employed, or unemployment benefit if unemployed
            income[on_market, t] = np.where(now_emp, h_new, unemp_benefit)

    # Return all simulation data for analysis and visualization
    return {
        'income': income,
        'h': h,
        'employed': employed,
        'edu_idx': edu_idx,
        'p_e': p_e,
        'sigma_p': sigma_p,
        'lambda_p': lambda_p,
    }