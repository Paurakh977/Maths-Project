import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def linear_regression_with_animation(X, y, m, c, epochs=1000, learning_rate=0.0001):
    N = float(len(y))
    errors = []  
    min_error = float('inf')  
    fig, ax = plt.subplots()
    ax.plot(X, y, 'b*')
    ax.set_xlabel('Ad investment (In 1000s of rupees)')
    ax.set_ylabel('Sales output (In 1000s of rupees)')
    ax.axis([0, 12, 0, 60])
    ax.grid(True)
    ax.set_aspect('auto', adjustable='box')  

    line, = ax.plot([], [], '-g', label='Current Fit Line')
    ax.legend(loc='best')

    def init():
        line.set_data([], [])
        return line,

    def update(i):
        nonlocal m, c, min_error
        y_guess = (m * X) + c
        error = np.sum((y_guess - y) ** 2) / N
        errors.append(error)         
        if error < min_error:
            min_error = error

        small_value_of_m = (2/N) * np.sum(X * (y_guess - y))
        small_value_of_c = (2/N) * np.sum(y_guess - y)
        m = m - (learning_rate * small_value_of_m)
        c = c - (learning_rate * small_value_of_c)

        x_values = np.linspace(0, 12, 10)
        y_values = m * x_values + c
        line.set_data(x_values, y_values)

        if error == min_error:
            best_fit_equation = f'y = {m:.2f}x + {c:.2f}'
            final_error = f'Final Error: {error:.2f}'
            ax.text(0.05, 0.9, f'Best Fit Line: {best_fit_equation}\n', transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

        return line,

    ani = animation.FuncAnimation(fig, update, frames=epochs, init_func=init, blit=True, interval=10)  # Decrease the interval
    plt.tight_layout()    
    plt.show()

    return m, c, errors

def run():
    m = 1
    c = 1
    points = np.genfromtxt("advertising-sales-data.txt", delimiter=",")
    best_fit_m, best_fit_c, errors = linear_regression_with_animation(points[:, 0], points[:, 1], m, c)
    print("Best fit slope:", best_fit_m)
    print("Best fit intercept:", best_fit_c)
    print("Final error:", errors[-1])

if __name__ == '__main__':
    run()
