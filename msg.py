import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Portfolio Optimisation App", layout="centered")

st.title("Sustainable Finance Portfolio Optimisation")
st.write("Enter the asset characteristics and investor preferences below.")

# ------------------------------
# Inputs from the user
# ------------------------------
st.header("Inputs")

r_h = st.number_input("Asset 1 Expected Return (%)", value=5.0, step=0.1) / 100
sd_h = st.number_input("Asset 1 Standard Deviation (%)", value=9.0, step=0.1) / 100

r_f = st.number_input("Asset 2 Expected Return (%)", value=12.0, step=0.1) / 100
sd_f = st.number_input("Asset 2 Standard Deviation (%)", value=20.0, step=0.1) / 100

rho_hf = st.slider("Correlation between Asset 1 and Asset 2", min_value=-1.0, max_value=1.0, value=-0.2, step=0.01)

r_free = st.number_input("Risk-Free Rate (%)", value=2.0, step=0.1) / 100
gamma = st.number_input("Risk Aversion (γ)", value=5.0, step=0.1, min_value=0.1)

# ------------------------------
# Functions
# ------------------------------
def portfolio_ret(w1, r1, r2):
    return w1 * r1 + (1 - w1) * r2

def portfolio_sd(w1, sd1, sd2, rho):
    variance = (
        w1**2 * sd1**2
        + (1 - w1)**2 * sd2**2
        + 2 * rho * w1 * (1 - w1) * sd1 * sd2
    )
    return np.sqrt(max(variance, 0))

# ------------------------------
# Run optimisation
# ------------------------------
if st.button("Calculate Optimal Portfolio"):
    weights = np.linspace(0, 1, 1000)
    sharpe_ratios = []

    for w in weights:
        ret = portfolio_ret(w, r_h, r_f)
        sd = portfolio_sd(w, sd_h, sd_f, rho_hf)
        if sd > 0:
            sharpe = (ret - r_free) / sd
            sharpe_ratios.append(sharpe)
        else:
            sharpe_ratios.append(-np.inf)

    max_idx = np.argmax(sharpe_ratios)
    w1_tangency = weights[max_idx]
    w2_tangency = 1 - w1_tangency

    ret_tangency = portfolio_ret(w1_tangency, r_h, r_f)
    sd_tangency = portfolio_sd(w1_tangency, sd_h, sd_f, rho_hf)

    # Optimal portfolio
    if sd_tangency > 0:
        w_tangency_optimal = (ret_tangency - r_free) / (gamma * sd_tangency**2)
    else:
        w_tangency_optimal = 0

    w1_optimal = w_tangency_optimal * w1_tangency
    w2_optimal = w_tangency_optimal * w2_tangency
    w_rf_optimal = 1 - w_tangency_optimal

    ret_optimal = r_free + w_tangency_optimal * (ret_tangency - r_free)
    sd_optimal = abs(w_tangency_optimal) * sd_tangency

    # ------------------------------
    # Display results
    # ------------------------------
    st.header("Optimal Portfolio Weights")
    st.write(f"**Risk-Free Asset:** {w_rf_optimal * 100:.2f}%")
    st.write(f"**Asset 1:** {w1_optimal * 100:.2f}%")
    st.write(f"**Asset 2:** {w2_optimal * 100:.2f}%")
    st.write(f"**Expected Return:** {ret_optimal * 100:.2f}%")
    st.write(f"**Portfolio Risk (Std Dev):** {sd_optimal * 100:.2f}%")

    # ------------------------------
    # Plot Efficient Frontier
    # ------------------------------
    weights_plot = np.linspace(0, 1, 200)
    returns_frontier = [portfolio_ret(w, r_h, r_f) for w in weights_plot]
    sds_frontier = [portfolio_sd(w, sd_h, sd_f, rho_hf) for w in weights_plot]

    fig, ax = plt.subplots(figsize=(8, 5))

    # Efficient frontier
    ax.plot(sds_frontier, returns_frontier, linewidth=2, label="Efficient Frontier")

    # Capital Market Line
    if sd_tangency > 0:
        sd_max = max(sds_frontier) * 1.2
        sd_cml = np.linspace(0, sd_max, 100)
        ret_cml = r_free + (ret_tangency - r_free) / sd_tangency * sd_cml
        ax.plot(sd_cml, ret_cml, "--", linewidth=2, label="Capital Market Line")

    # Tangency portfolio
    ax.scatter(sd_tangency, ret_tangency, s=100, marker="*", label="Tangency Portfolio")

    # Optimal portfolio
    ax.scatter(sd_optimal, ret_optimal, s=100, marker="D", label="Optimal Portfolio")

    # Risk-free asset
    ax.scatter(0, r_free, s=80, marker="s", label="Risk-Free Asset")

    ax.set_xlabel("Risk (Standard Deviation)")
    ax.set_ylabel("Expected Return")
    ax.set_title("Portfolio Optimization")
    ax.legend()
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)
