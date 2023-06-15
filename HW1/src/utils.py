import matplotlib.pyplot as plt
import matplotlib.ticker
from unconstrained_min import * 
import numpy as np


# def plot_line_search(func, func_name, line_search_results):
#     fig, ax = plt.subplots(figsize=(7, 7))
#     plt.title("Convergence of function: " + func_name)
#     colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(line_search_results)))
#     # resort such that shorter path will appear above longer path
#     line_search_results = sorted(line_search_results, key=lambda ls_res: len(ls_res), reverse=True)
#     for ind, ls in enumerate(line_search_results):
#         objective_values = ls.objective_values
#         num_vals = len(objective_values)
#         color = colors[ind]
#         label = get_plot_label(ls)
#         if num_vals <= 2:
#             plt.scatter(np.arange(1, num_vals + 1), objective_values, label=label, linewidth=2, color=color)
#         else:
#             plt.plot(np.arange(1, num_vals + 1), objective_values, label=label, color=color)
#     min_y = min([min(ls.objective_values) for ls in line_search_results])
#     max_y = max([max(ls.objective_values) for ls in line_search_results])
#     if min_y < 0:
#         ax.set_ylim(top=max_y + 0.2 * (max_y - min_y))
#     ax.set_ylabel("Objective function value")
#     ax.set_xlabel("Number of iterations")
#     max_iter = max([len(ls) for ls in line_search_results])
#     min_iter = min([len(ls) for ls in line_search_results])
#     ax.set_xlim(left=0.9, right=max_iter + 0.1)
#     if max_iter > 10 * min_iter:
#         ax.set_xscale('log')
#         formatter = matplotlib.ticker.FuncFormatter(lambda y, _: '{:.16g}'.format(y))
#         ax.xaxis.set_major_formatter(formatter)
#     plt.legend(loc="upper right", prop={'size': 6})
#     plt.show()

def plot_graphs(f, x, a, b, l, obj_tol=1e-12, param_tol=1e-8, max_iter=100):
    paths = {}
    flag = {}
    x1, y1, flag["gradient_descent"], paths["gradient_descent"] = gradient_descent(f, x, obj_tol, param_tol, max_iter)

    x2, y2, flag["newton"], paths["newton"] = newton(f, x, obj_tol, param_tol, max_iter)

    x3, y3, flag["BFGS"], paths["BFGS"] = BFGS(f, x, obj_tol, param_tol, max_iter)

    x4, y4, flag["SR1"], paths["SR1"] = SR1(f, x, obj_tol, param_tol, max_iter)

    # Define the contour grid
    x1_contour = np.linspace(-a, a, 100)
    x2_contour = np.linspace(-b, b, 100)
    X1_contour, X2_contour = np.meshgrid(x1_contour, x2_contour)

    Z_contour = np.zeros_like(X1_contour)
    for i in range(len(x1_contour)):
        for j in range(len(x2_contour)):
            Z_contour[i, j] = f([X1_contour[i, j], X2_contour[i, j]])

    fig, axes = plt.subplots(4, 2, figsize=(12, 20))
    for i, method in enumerate(paths):
        ax1 = axes[i, 0]
        ax2 = axes[i, 1]

        # Plot the contour lines
        ax1.contour(X1_contour, X2_contour, Z_contour, levels=l)

        # Plot the iteration paths
        path = paths[method]
        x_vals, y_vals = zip(*path)
        ax1.plot(x_vals, y_vals, marker='o', linestyle='-', color='red')
        ax1.plot(x_vals[-1], y_vals[-1], marker='o', linestyle='-', color='black')
        ax1.annotate(f'({x_vals[-1]:.2f}, {y_vals[-1]:.2f})', xy=(x_vals[-1], y_vals[-1]), xytext=(x_vals[-1] + 0.1, y_vals[-1] - 0.5))

        ax1.set_xlabel('x')
        ax1.set_ylabel('y')

        ax1.set_title(method + ', iteration times: ' + str(len(path) - 1) + ', Flag: ' + str(flag[method]))
        ax1.grid(True)

        # Scatter plot of objective function values with number of iterations
        iterations = range(1, len(path))
        objective_values = [f(point) for point in path[1:]]
        ax2.scatter(iterations, objective_values, marker='o', color='blue')
        ax2.set_xlabel('Number of Iterations')
        ax2.set_ylabel('Objective Function Value')
        ax2.set_title(method + ' - Objective Function Value vs. Iterations')
        ax2.grid(True)

    plt.tight_layout()
    plt.show()
