import os
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import pandas as pd

# Define the models with varying substances, rate constants, and ODE functions
models = {
    "Modelo1": {
        "substances": ["CFA", "CPFA", "CiPrFE", "CiPrL", "CW"],
        "rate_constants": ["k1f", "k1r", "k2f", "k2r", "k3f", "k3r"],
        "ode_function": lambda t, y, k: [
            -k["k1f"] * y[0]**2 + k["k1r"] * y[1] * y[4]- k["k2f"] * y[0] + k["k2r"] * y[2] * y[4],
            k["k1f"] * y[0]**2 - k["k1r"] * y[1] * y[4],
            k["k2f"] * y[0] - k["k2r"] * y[2] * y[4] - k["k3f"] * y[2] * y[4] + k["k3r"] * y[3],
            k["k3f"] * y[2] * y[4] - k["k3r"] * y[3],
            k["k1f"] * y[0]**2 - k["k1r"] * y[1] * y[4] + k["k2f"] * y[0] - k["k2r"] * y[2] * y[4] - k["k3f"] * y[2] * y[4] + k["k3r"] * y[3],
        ],
        "initial_conditions": [0.358817533, 0, 0, 0, 0],
        "k_guess": [0.1, 0.01, 0.05, 0.01, 0.3, 0.2],
    },
    "Modelo2": {
        "substances": ["CFA", "CPFA", "CiPrFE", "CiPrL"],
        "rate_constants": ["k1f", "k1r", "k2f", "k2r", "k3f", "k3r"],
        "ode_function": lambda t, y, k: [
            -k["k1f"] * y[0]**2 + k["k1r"] * y[1] - k["k2f"] * y[0] + k["k2r"] * y[2] ,
            k["k1f"] * y[0]**2 - k["k1r"] * y[1],
            k["k2f"] * y[0] - k["k2r"] * y[2] - k["k3f"] * y[2] + k["k3r"] * y[3],
            k["k3f"] * y[2]  - k["k3r"] * y[3],
        ],
        "initial_conditions": [0.358817533, 0, 0, 0],
        "k_guess": [0.1, 0.01, 0.05, 0.01, 0.3, 0.2],
    },
    "Modelo3": {
        "substances": ["CFA", "CPFA", "CiPrFE", "CiPrL", "CW", "LOH"],
        "rate_constants": ["k1f", "k1r", "k2f", "k2r", "k3f", "k3r", "k4f", "k4r"],
        "ode_function": lambda t, y, k: [
            -k["k1f"] * y[0]**2 + k["k1r"] * y[1] * y[4]- k["k2f"] * y[0] + k["k2r"] * y[2] * y[4],
            k["k1f"] * y[0]**2 - k["k1r"] * y[1] * y[4],
            k["k2f"] * y[0] - k["k2r"] * y[2] * y[4] - k["k3f"] * y[2] * y[4] + k["k3r"] * y[3],
            k["k3f"] * y[2] * y[4] - k["k3r"] * y[3] - k["k4f"] * y[3] * y[4] + k["k4r"] * y[5],
            k["k1f"] * y[0]**2 - k["k1r"] * y[1] * y[4] + k["k2f"] * y[0] - k["k2r"] * y[2] * y[4] - k["k3f"] * y[2] * y[4] + k["k3r"] * y[3],
            k["k4f"] * y[3] * y[4] - k["k4r"] * y[5],
        ],
        "initial_conditions": [0.358817533, 0, 0, 0, 0, 0],
        "k_guess": [0.1, 0.01, 0.05, 0.01, 0.3, 0.2, 0.0, 0.0],
    },
    "Modelo4": {
        "substances": ["CFA", "CPFA", "CiPrFE", "CiPrL", "LOH"],
        "rate_constants": ["k1f", "k1r", "k2f", "k2r", "k3f", "k3r", "k4f", "k4r"],
        "ode_function": lambda t, y, k: [
            -k["k1f"] * y[0]**2 + k["k1r"] * y[1] - k["k2f"] * y[0] + k["k2r"] * y[2] ,
            k["k1f"] * y[0]**2 - k["k1r"] * y[1],
            k["k2f"] * y[0] - k["k2r"] * y[2] - k["k3f"] * y[2] + k["k3r"] * y[3],
            k["k3f"] * y[2]  - k["k3r"] * y[3] - k["k4f"] * y[3] + k["k4r"] * y[4],
            k["k4f"] * y[3] - k["k4r"] * y[4],
        ],
        "initial_conditions": [0.358817533, 0, 0, 0, 0],
        "k_guess": [0.1, 0.01, 0.05, 0.01, 0.3, 0.2, 0.0, 0.0],
    },
    "Modelo5": {
        "substances": ["CFA", "CiPrFE", "CiPrL", "CW"],
        "rate_constants": ["k2f", "k2r", "k3f", "k3r"],
        "ode_function": lambda t, y, k: [
            - k["k2f"] * y[0] + k["k2r"] * y[1] * y[3],
            k["k2f"] * y[0] - k["k2r"] * y[1] * y[3] - k["k3f"] * y[1] * y[3] + k["k3r"] * y[3],
            k["k3f"] * y[1] * y[3] - k["k3r"] * y[2],
            k["k2f"] * y[0] - k["k2r"] * y[1] * y[3] - k["k3f"] * y[1] * y[3] + k["k3r"] * y[2],
        ],
        "initial_conditions": [0.358817533, 0, 0, 0],
        "k_guess": [0.1, 0.01, 0.05, 0.01],
    },
    "Modelo6": {
        "substances": ["CFA", "CiPrFE", "CiPrL"],
        "rate_constants": ["k2f", "k2r", "k3f", "k3r"],
        "ode_function": lambda t, y, k: [
            - k["k2f"] * y[0] + k["k2r"] * y[1] ,
            k["k2f"] * y[0] - k["k2r"] * y[1] - k["k3f"] * y[1] + k["k3r"] * y[2],
            k["k3f"] * y[1]  - k["k3r"] * y[2],
        ],
        "initial_conditions": [0.358817533, 0, 0],
        "k_guess": [0.1, 0.01, 0.05, 0.01],
    },
    "Modelo7": {
        "substances": ["CFA", "CiPrFE", "CiPrL", "CW", "LOH"],
        "rate_constants": ["k2f", "k2r", "k3f", "k3r", "k4f", "k4r"],
        "ode_function": lambda t, y, k: [
            - k["k2f"] * y[0] + k["k2r"] * y[1] * y[3],
            k["k2f"] * y[0] - k["k2r"] * y[1] * y[3] - k["k3f"] * y[1] * y[3] + k["k3r"] * y[2],
            k["k3f"] * y[1] * y[3] - k["k3r"] * y[2] - k["k4f"] * y[2] * y[3] + k["k4r"] * y[4],
            k["k2f"] * y[0] - k["k2r"] * y[1] * y[3] - k["k3f"] * y[1] * y[3] + k["k3r"] * y[2],
            k["k4f"] * y[2] * y[3] - k["k4r"] * y[4],
        ],
        "initial_conditions": [0.358817533, 0, 0, 0, 0],
        "k_guess": [0.05, 0.01, 0.3, 0.2, 0.0, 0.0],
    },
    "Modelo8": {
        "substances": ["CFA", "CiPrFE", "CiPrL", "LOH"],
        "rate_constants": ["k2f", "k2r", "k3f", "k3r", "k4f", "k4r"],
        "ode_function": lambda t, y, k: [
             k["k2f"] * y[0] + k["k2r"] * y[1] ,
            k["k2f"] * y[0] - k["k2r"] * y[1] - k["k3f"] * y[1] + k["k3r"] * y[2],
            k["k3f"] * y[1]  - k["k3r"] * y[2] - k["k4f"] * y[2] + k["k4r"] * y[3],
            k["k4f"] * y[2] - k["k4r"] * y[3],
        ],
        "initial_conditions": [0.358817533, 0, 0, 0],
        "k_guess": [0.05, 0.01, 0.3, 0.2, 0.0, 0.0],
    },
    "Modelo9": {
        "substances": ["CFA", "CPFA", "CiPrFE", "CiPrL", "CW"],
        "rate_constants": ["k1f", "k1r", "k2f", "k2r", "k3f", "k3r"],
        "ode_function": lambda t, y, k: [
            -k["k1f"] * y[0] + k["k1r"] * y[1] * y[4]- k["k2f"] * y[0] + k["k2r"] * y[2] * y[4],
            k["k1f"] * y[0] - k["k1r"] * y[1] * y[4],
            k["k2f"] * y[0] - k["k2r"] * y[2] * y[4] - k["k3f"] * y[2] * y[4] + k["k3r"] * y[3],
            k["k3f"] * y[2] * y[4] - k["k3r"] * y[3],
            k["k1f"] * y[0]- k["k1r"] * y[1] * y[4] + k["k2f"] * y[0] - k["k2r"] * y[2] * y[4] - k["k3f"] * y[2] * y[4] + k["k3r"] * y[3],
        ],
        "initial_conditions": [0.358817533, 0, 0, 0, 0],
        "k_guess": [0.1, 0.01, 0.05, 0.01, 0.3, 0.2],
    },
    "Modelo10": {
        "substances": ["CFA", "CPFA", "CiPrFE", "CiPrL"],
        "rate_constants": ["k1f", "k1r", "k2f", "k2r", "k3f", "k3r"],
        "ode_function": lambda t, y, k: [
            -k["k1f"] * y[0] + k["k1r"] * y[1] - k["k2f"] * y[0] + k["k2r"] * y[2] ,
            k["k1f"] * y[0] - k["k1r"] * y[1],
            k["k2f"] * y[0] - k["k2r"] * y[2] - k["k3f"] * y[2] + k["k3r"] * y[3],
            k["k3f"] * y[2]  - k["k3r"] * y[3],
        ],
        "initial_conditions": [0.358817533, 0, 0, 0],
        "k_guess": [0.1, 0.01, 0.05, 0.01, 0.3, 0.2],
    },
    "Modelo11": {
        "substances": ["CFA", "CPFA", "CiPrFE", "CiPrL", "CW", "LOH"],
        "rate_constants": ["k1f", "k1r", "k2f", "k2r", "k3f", "k3r", "k4f", "k4r"],
        "ode_function": lambda t, y, k: [
            -k["k1f"] * y[0] + k["k1r"] * y[1] * y[4]- k["k2f"] * y[0] + k["k2r"] * y[2] * y[4],
            k["k1f"] * y[0] - k["k1r"] * y[1] * y[4],
            k["k2f"] * y[0] - k["k2r"] * y[2] * y[4] - k["k3f"] * y[2] * y[4] + k["k3r"] * y[3],
            k["k3f"] * y[2] * y[4] - k["k3r"] * y[3] - k["k4f"] * y[3] * y[4] + k["k4r"] * y[5],
            k["k1f"] * y[0] - k["k1r"] * y[1] * y[4] + k["k2f"] * y[0] - k["k2r"] * y[2] * y[4] - k["k3f"] * y[2] * y[4] + k["k3r"] * y[3],
            k["k4f"] * y[3] * y[4] - k["k4r"] * y[5],
        ],
        "initial_conditions": [0.358817533, 0, 0, 0, 0, 0],
        "k_guess": [0.1, 0.01, 0.05, 0.01, 0.3, 0.2, 0.0, 0.0],
    },
    "Modelo12": {
        "substances": ["CFA", "CPFA", "CiPrFE", "CiPrL", "LOH"],
        "rate_constants": ["k1f", "k1r", "k2f", "k2r", "k3f", "k3r", "k4f", "k4r"],
        "ode_function": lambda t, y, k: [
            -k["k1f"] * y[0] + k["k1r"] * y[1] - k["k2f"] * y[0] + k["k2r"] * y[2] ,
            k["k1f"] * y[0] - k["k1r"] * y[1],
            k["k2f"] * y[0] - k["k2r"] * y[2] - k["k3f"] * y[2] + k["k3r"] * y[3],
            k["k3f"] * y[2]  - k["k3r"] * y[3] - k["k4f"] * y[3] + k["k4r"] * y[4],
            k["k4f"] * y[3] - k["k4r"] * y[4],
        ],
        "initial_conditions": [0.358817533, 0, 0, 0, 0],
        "k_guess": [0.1, 0.01, 0.05, 0.01, 0.3, 0.2, 0.0, 0.0],
    },
}

# Experimental data
time_exp = np.array([0, 15, 30, 60, 90, 120, 150, 180, 210])
data_exp = {
    "CFA": np.array([0.358817533, 0.035662968, 0.002839061, 0.000000000, 0.000000000, 0.000000000, 0.000000000, 0.000000000, 0.000000000]),
    "CiPrFE": np.array([0, 0.017226560, 0.027195006, 0.025357552, 0.068633591, 0.025737597, 0.000000000, 0.000000000, 0.000000000]),
    "CiPrL": np.array([0, 0.016224873, 0.052487389, 0.185125347, 0.357775333, 0.338530099, 0.305112130, 0.257521427, 0.195757988]),
    "time": time_exp,
}

def validate_inputs(model):
    substances = model["substances"]
    initial_conditions = model["initial_conditions"]
    k_guess = model["k_guess"]

    if len(initial_conditions) != len(substances):
        raise ValueError("Initial conditions length does not match substances length.")

    if len(k_guess) != len(model["rate_constants"]):
        raise ValueError("Rate constants length does not match initial guess length.")

def solve_ode(ode_function, initial_conditions, k, time_span):
    try:
        return solve_ivp(
            ode_function,
            time_span,
            initial_conditions,
            t_eval=data_exp["time"],
            args=(k,),
            method='LSODA',
            rtol=1e-10,
            atol=1e-11
        )
    except Exception as e:
        print(f"ODE solving error: {e}")
        raise

def plot_residue_graphs(residuals_dict, time_points, model_name):
    plt.figure(figsize=(12, 2 * len(residuals_dict)))
    for i, (substance, residuals) in enumerate(residuals_dict.items()):
        std_dev = np.std(residuals)

        plt.subplot(len(residuals_dict), 1, i + 1)
        plt.plot(time_points, residuals, "o", label=f"{substance} Resíduos")
        plt.axhline(3 * std_dev, color="r", linestyle="--", label="+3sey")
        plt.axhline(-3 * std_dev, color="r", linestyle="--", label="-3sey")
        plt.title(f"{substance} Gráfico de resíduos do {model_name}")
        plt.xlabel("Tempo / min")
        plt.ylabel("Resíduo")
        plt.legend()
        plt.grid()

    plt.tight_layout()

    # Save the plot instead of showing it interactively
    output_plot_file = f"{model_name}_residual_plot.png"
    plt.savefig(output_plot_file)
    print(f"Residual plot saved to: {output_plot_file}")

    # Optionally show the plot and close
    plt.show()
    plt.close()


def export_results(solution, substances, model_name):
    results = {"Time (min)": solution.t}
    for i, substance in enumerate(substances):
        results[substance] = solution.y[i]

    results_df = pd.DataFrame(results)
    output_file = f"{model_name}_simulation_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")

def solve_model(model_name, time_span):
    model = models[model_name]
    validate_inputs(model)

    substances = model["substances"]
    rate_constants = model["rate_constants"]
    ode_function = model["ode_function"]
    initial_conditions = model["initial_conditions"]
    k_guess = model["k_guess"]

    def cost_function(k_values):
        try:
            k = dict(zip(rate_constants, k_values))
            solution = solve_ode(ode_function, initial_conditions, k, time_span)
            interpolated_data = [
                np.interp(data_exp["time"], solution.t, solution.y[i])
                for i in range(len(substances))
            ]
            residuals = np.concatenate([
                interpolated_data[i] - data_exp[substance]
                for i, substance in enumerate(substances) if substance in data_exp
            ])
            return residuals
        except Exception as e:
            print(f"Error in cost function: {e}")
            return np.inf * np.ones(len(k_values))  # Avoid breaking optimization

    try:
        start_time = time.time()
        result = least_squares(
            cost_function,
            k_guess,
            bounds=(0, np.inf),
            max_nfev=1000,  # Limit function evaluations
            verbose=2        # Add verbose output for debugging
        )
        optimization_time = time.time() - start_time

        if not result.success:
            print(f"Optimization failed for {model_name}: {result.message}")
            return None, None

        optimized_k = dict(zip(rate_constants, result.x))
        solution = solve_ode(ode_function, initial_conditions, optimized_k, time_span)

        residuals_dict = {}
        r2_values = {}
        for i, substance in enumerate(substances):
            if substance in data_exp:
                interpolated_solution = np.interp(data_exp["time"], solution.t, solution.y[i])
                residuals = interpolated_solution - data_exp[substance]
                residuals_dict[substance] = residuals

                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((data_exp[substance] - np.mean(data_exp[substance]))**2)
                r2_values[substance] = 1 if ss_tot == 0 else 1 - ss_res / ss_tot

        plot_residue_graphs(residuals_dict, data_exp["time"], model_name)
        export_results(solution, substances, model_name)

        return optimized_k, r2_values

    except Exception as e:
        print(f"Error solving model {model_name}: {e}")
        return None, None

if __name__ == "__main__":
    time_span = [0, 210]

    for model_name in models.keys():
        print(f"\n{'=' * 40}")
        print(f"Solving {model_name}...")
        optimized_k, r2_values = solve_model(model_name, time_span)

        if optimized_k:
            print(f"\nOptimized Rate Constants for {model_name}:")
            for k_name, k_value in optimized_k.items():
                print(f"  {k_name}: {k_value:.40f}")

            print(f"\nR² Values for {model_name}:")
            for substance, r2 in r2_values.items():
                print(f"  {substance}: {r2:.4f}")

            print(f"\nResults saved to: {model_name}_simulation_results.csv")
        else:
            print(f"\nFailed to solve {model_name}.")

