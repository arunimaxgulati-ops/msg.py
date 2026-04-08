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
st.subheader("Investor Preference Presets")

preset = st.selectbox(
    "Choose an investor profile",
    ["Custom", "Profit Motivated", "Balanced", "ESG Focused"])
if preset == "Profit Motivated":
    default_gamma = 2.0
    default_lambda = 0.01
elif preset == "Balanced":
    default_gamma = 5.0
    default_lambda = 0.05
elif preset == "ESG Focused":
    default_gamma = 7.0
    default_lambda = 0.10
else:
    default_gamma = 5.0
    default_lambda = 0.05
gamma = st.number_input(
    "Risk Aversion (γ)",
    value=float(default_gamma),
    step=0.1,
    min_value=0.1
)

lambda_esg = st.number_input(
    "ESG Preference (λ)",
    value=float(default_lambda),
    step=0.01,
    min_value=0.0
)

st.subheader("ESG Inputs")
st.subheader("ESG Inputs (E, S, G Breakdown)")

st.write("Asset 1 ESG Scores")
e1 = st.number_input("Asset 1 - Environmental Score", value=70.0)
s1 = st.number_input("Asset 1 - Social Score", value=65.0)
g1 = st.number_input("Asset 1 - Governance Score", value=75.0)

st.write("Asset 2 ESG Scores")
e2 = st.number_input("Asset 2 - Environmental Score", value=80.0)
s2 = st.number_input("Asset 2 - Social Score", value=85.0)
g2 = st.number_input("Asset 2 - Governance Score", value=78.0)

# Combine ESG scores (simple average)
esg1 = (e1 + s1 + g1) / 3
esg2 = (e2 + s2 + g2) / 3

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
def portfolio_esg(w1, esg1, esg2):
    return w1 * esg1 + (1 - w1) * esg2

def traditional_utility(ret, sd, gamma):
    return ret - (gamma / 2) * (sd ** 2)

def utility_function(ret, sd, gamma, lambda_esg, esg_score):
    return ret - (gamma / 2) * (sd ** 2) + lambda_esg * (esg_score / 100)
# ------------------------------
# Run optimisation
# ------------------------------
if st.button("Calculate Optimal Portfolio"):
    weights = np.linspace(0, 1, 1000)

    returns = []
    risks = []
    esg_scores = []
    traditional_utilities = []
    esg_utilities = []

    for w in weights:
        ret = portfolio_ret(w, r_h, r_f)
        sd = portfolio_sd(w, sd_h, sd_f, rho_hf)
        esg_score = portfolio_esg(w, esg1, esg2)

        trad_u = traditional_utility(ret, sd, gamma)
        esg_u = utility_function(ret, sd, gamma, lambda_esg, esg_score)

        returns.append(ret)
        risks.append(sd)
        esg_scores.append(esg_score)
        traditional_utilities.append(trad_u)
        esg_utilities.append(esg_u)

    # Traditional optimal portfolio
    trad_idx = np.argmax(traditional_utilities)
    w1_trad = weights[trad_idx]
    w2_trad = 1 - w1_trad
    ret_trad = returns[trad_idx]
    sd_trad = risks[trad_idx]
    esg_trad = esg_scores[trad_idx]
    utility_trad = traditional_utilities[trad_idx]

    # ESG-adjusted optimal portfolio
    esg_idx = np.argmax(esg_utilities)
    w1_esg = weights[esg_idx]
    w2_esg = 1 - w1_esg
    ret_esg = returns[esg_idx]
    sd_esg = risks[esg_idx]
    esg_portfolio = esg_scores[esg_idx]
    utility_esg = esg_utilities[esg_idx]
    
    # Percentage differences: ESG-adjusted vs Traditional
    if esg_trad != 0:
        esg_improvement_pct = ((esg_portfolio - esg_trad) / esg_trad) * 100
    else:
        esg_improvement_pct = 0

    if ret_trad != 0:
        return_change_pct = ((ret_esg - ret_trad) / ret_trad) * 100
    else:
        return_change_pct = 0

    if sd_trad != 0:
        risk_change_pct = ((sd_esg - sd_trad) / sd_trad) * 100
    else:
        risk_change_pct = 0

    st.header("Portfolio Comparison")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Traditional Portfolio")
        st.write(f"**Asset 1 Weight:** {w1_trad * 100:.2f}%")
        st.write(f"**Asset 2 Weight:** {w2_trad * 100:.2f}%")
        st.write(f"**Expected Return:** {ret_trad * 100:.2f}%")
        st.write(f"**Portfolio Risk:** {sd_trad * 100:.2f}%")
        st.write(f"**Portfolio ESG Score:** {esg_trad:.2f}")
        st.write(f"**Utility:** {utility_trad:.4f}")

    with col2:
        st.subheader("ESG-Adjusted Portfolio")
        st.write(f"**Asset 1 Weight:** {w1_esg * 100:.2f}%")
        st.write(f"**Asset 2 Weight:** {w2_esg * 100:.2f}%")
        st.write(f"**Expected Return:** {ret_esg * 100:.2f}%")
        st.write(f"**Portfolio Risk:** {sd_esg * 100:.2f}%")
        st.write(f"**Portfolio ESG Score:** {esg_portfolio:.2f}")
        st.write(f"**Utility:** {utility_esg:.4f}")

        st.subheader("ESG Improvement vs Traditional")

    st.write(f"**Change in Portfolio ESG Score:** {esg_improvement_pct:.2f}%")
    st.write(f"**Change in Expected Return:** {return_change_pct:.2f}%")
    st.write(f"**Change in Portfolio Risk:** {risk_change_pct:.2f}%")

    if esg_improvement_pct > 0 and return_change_pct < 0:
        st.write(
            "The ESG-adjusted portfolio improves sustainability, but this comes with a reduction in expected return."
        )
    elif esg_improvement_pct > 0 and return_change_pct >= 0:
        st.write(
            "The ESG-adjusted portfolio improves sustainability without reducing expected return."
        )
    else:
        st.write(
            "Including ESG preferences does not materially improve the ESG profile under the current inputs."
        )

    # Interpretation
    st.subheader("Interpretation")
    if esg_portfolio > esg_trad:
        st.write(
            "Including ESG preferences shifts the recommended portfolio toward the asset mix with a higher sustainability score."
        )
    else:
        st.write(
            "Including ESG preferences does not materially change the portfolio for these inputs."
        )

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(risks, returns, linewidth=2, label="Efficient Frontier")
    ax.scatter(sd_trad, ret_trad, s=120, marker="*", label="Traditional Portfolio")
    ax.scatter(sd_esg, ret_esg, s=100, marker="D", label="ESG-Adjusted Portfolio")

    ax.set_xlabel("Risk (Standard Deviation)")
    ax.set_ylabel("Expected Return")
    ax.set_title("Traditional vs ESG-Adjusted Portfolio")
    ax.legend()
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)
