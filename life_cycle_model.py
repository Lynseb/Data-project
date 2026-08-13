import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def simulate_lifecycle(
    seed=23,
    N=50000,
    sigma_p=0.05,  # Job-separation (layoff) probability (can be scalar or array for Task 2.5)
    lambda_p=0.60,  # Job-finding probability (can be scalar or array for Task 2.5)

     # -----------------------------------------------------------------------------
        # TASK 2.4: The following switches off specific features of the model for alternative scenarios.
        # To calculate the effect of each feature,
        # -----------------------------------------------------------------------------

    no_edu_diff=False,  # Turn off education differences
    no_shocks=False,  # Turn off productivity shocks
    no_depr=False,  # Turn off human capital depreciation during unemployment
    no_unemp=False,  # Turn off unemployment
):
    """Simulates the life-cycle income model from age 18 to 65."""


    # Initialize random number generator for reproducibility
    rng = np.random.default_rng(seed)
    T = 65 - 18  # 47 years (from age 18 to 65)

    # Parameters from the model specification
    p_e = [0.40, 0.35, 0.25]  # Education probabilities [short, medium, long]
    S_e = [1, 3, 5]  # Years of education for each track
    h_e0 = [1.00, 1.20, 1.55]  # Initial human capital for each education level
    Delta_e = [
        0.010,
        0.020,
        0.030,
    ]  # Growth rate of human capital for each education level
    delta = 0.06  # Depreciation rate of human capital when unemployed
    sigma_psi = (
        0.10  # Standard deviation of the lognormal shock to human capital
    )

    # -----------------------------------------------------------------------------
    # TASK 2.5 EXTENSION: Support time-varying labor market shocks (Recessions/External Shocks)
    # Convert scalar probabilities to vectors across time T if single values are passed.
    # -----------------------------------------------------------------------------
    sigma_p_vec = (
        np.array(sigma_p)
        if isinstance(sigma_p, (list, np.ndarray))
        else np.full(T, sigma_p)
    )
    lambda_p_vec = (
        np.array(lambda_p)
        if isinstance(lambda_p, (list, np.ndarray))
        else np.full(T, lambda_p)
    )

    y_SU = 0.45  # Student grant (income during education)
    rho = 0.60  # Unemployment benefit replacement rate (60% of previous wage)
    y_underline = (
        0.35  # Minimum benefit floor (social assistance for never-employed)
    )

    # -----------------------------------------------------------------------------
    # TASK 2.4 SWITCHES: Overwrite parameters if alternative scenarios are active
    # -----------------------------------------------------------------------------
    if no_shocks:
        sigma_psi = 0.0  # Turn off productivity shocks

    if no_depr:
        delta = 0.0  # Turn off human capital depreciation during unemployment

    if no_unemp:
        sigma_p_vec = np.zeros(T)  # No layoffs (nobody loses their job)
        lambda_p_vec = np.ones(
            T
        )  # Immediate job finding for unemployed (lambda = 1.0)

    if no_edu_diff:
        # Equalize initial human capital, growth rates, and education duration
        h_e0 = [1.20, 1.20, 1.20]
        Delta_e = [0.020, 0.020, 0.020]
        S_e = [0, 0, 0]  # Everyone starts working at the same age

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
        # Fetch current year parameters (Supports Task 2.5 time-varying shocks)
        current_sigma_p = sigma_p_vec[t]
        current_lambda_p = lambda_p_vec[t]

        # Draw a mean-one lognormal shock to human capital for each individual
        # The term -0.5 * sigma_psi**2 corrects the mean of the lognormal distribution to exactly 1.0
        if sigma_psi > 0:
            psi = rng.lognormal(-0.5 * sigma_psi**2, sigma_psi, size=N)
        else:
            psi = np.ones(N)  # No shock (multiplier = 1.0)

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
            if no_unemp:
                # If unemployment is turned off, individuals start directly in employment
                income[entering, t] = h_0_i[entering]
                h[entering, t] = h_0_i[entering]
                employed[entering, t] = True
                has_been_employed[entering] = True
                last_job_income[entering] = h_0_i[entering]
            else:
                income[entering, t] = (
                    y_underline  # Benefit floor for never-employed
                )
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

            # Markov transition logic (Using Task 2.5 time-varying probabilities current_sigma_p & current_lambda_p):
            # - If previously employed (prev_emp == True): Remains employed if draw > current_sigma_p.
            # - If previously unemployed (prev_emp == False): Finds job if draw < current_lambda_p.
            if no_unemp:
                now_emp = np.ones(np.sum(on_market), dtype=bool)
            else:
                now_emp = np.where(
                    prev_emp, draws > current_sigma_p, draws < current_lambda_p
                )

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
        "income": income,
        "h": h,
        "employed": employed,
        "edu_idx": edu_idx,
        "p_e": p_e,
        "sigma_p": sigma_p_vec,
        "lambda_p": lambda_p_vec,
    }