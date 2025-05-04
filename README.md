# Game Analytics – A/B Testing and Blockchain Feature Analysis
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://abtestgame.streamlit.app/)

## Overview

This project simulates a controlled A/B test for a blockchain-integrated game feature, with a focus on player engagement, monetization, and interaction with decentralized in-game assets. It is designed to demonstrate core competencies in experimental design, applied statistics (frequentist and Bayesian), and in-game economy analysis.

## Scope

- Simulated data representing 1,000 players split into control and variant groups.
- Includes metrics for revenue, XP progression, gameplay events (stunts/crashes), token usage, and NFT purchases.
- Applies both traditional and Bayesian A/B testing methods to evaluate feature performance.
- Produces actionable business insights based on statistical outcomes and key KPIs.

## Analysis Highlights

- **Frequentist Testing**: Welch’s t-test to detect differences in mean revenue between groups.
- **Bayesian Estimation**: Posterior probability simulation to assess likelihood of improvement.
- **Causal Inference**: Assumes randomized group assignment; includes notes on validity and confounding risks.
- **KPIs Tracked**:
  - ARPU (Average Revenue Per User)
  - NFT Purchase Rate
  - XP Accumulation
  - Token Spend
  - Event-Driven Engagement

## Interactive Dashboard

This project includes a Streamlit application (`app.py`) located in the `dashboards` folder, which provides an interactive way to explore the A/B test results. The dashboard allows you to:

- Visualize key metrics for both the control and variant groups.
- See the results of the frequentist and Bayesian statistical analyses.
- Filter and compare different segments of the player data.
- Dynamically adjust analysis parameters such as the significance level and the number of Bayesian samples.
- Gain a more nuanced and interactive understanding of the feature's impact through various plots and metrics.

**Key Features of the Dashboard:**

- **Data Exploration:** View key performance indicators at a glance for both groups.
- **Visualizations:** Interactive plots for revenue distribution, token usage, NFT purchase rates, XP distribution, and event impact.
- **A/B Testing Results:** Clear presentation of frequentist t-test results and Bayesian probability of improvement.
- **User Controls:** Sidebar for easy adjustment of visualization groups, Bayesian sampling size, and significance level.
- **Data Filtering:** Option to filter the data based on different columns for more granular analysis.

## Tools Used

- Python, Pandas, NumPy, SciPy
- Matplotlib, Seaborn (for visualization)
- Jupyter Notebook (for exploratory workflow)
- Streamlit (for interactive dashboard - see below)

## How to Use

- **Clone the Repository:**
   Begin by cloning the project repository to your local machine.

- **Install Dependencies:**
   Before running any code, ensure you have all the required libraries installed. Navigate to the project's root directory in your terminal and run:
   ```bash
   pip install -r requirements.txt

- **Explore with the Jupyter Notebook:**
    * Open the ab_test_analysis.ipynb file in your Jupyter environment.
    * Execute each cell sequentially. This will run the simulation, perform the statistical analysis, and generate the visualizations.
    * Once the notebook has finished running, review the final summary section for key findings and business recommendations.


- **Launch the Interactive Dashboard:**
    * Open your terminal and navigate to the dashboard directory:
   ```bash
   cd dashboards
   ```
    * Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```
