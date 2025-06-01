import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from typing import Optional

class Visualizer:

    @staticmethod
    def plot_multiple_solutions(
        Q: np.ndarray,
        c: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
        solutions: dict[str, np.ndarray],
        bounds: Optional[tuple] = (0, 5),
        show_gradients: bool = True
    ):
        """
        Визуализация нескольких решений QP на одном графике.
        
        Параметры:
            Q, c, A, b - параметры задачи QP
            solutions - словарь {метод: решение}
            bounds - границы области визуализации
            show_gradients - отображать градиенты
        """
        if Q.shape[0] != 2:
            print("Визуализация доступна только для 2D задач.")
            return

        # Создаем сетку для расчетов
        x = np.linspace(bounds[0], bounds[1], 100)
        y = np.linspace(bounds[0], bounds[1], 100)
        X, Y = np.meshgrid(x, y)
        
        # Вычисляем значения целевой функции
        Z = 0.5 * (Q[0,0]*X**2 + (Q[0,1]+Q[1,0])*X*Y + Q[1,1]*Y**2) + c[0]*X + c[1]*Y

        # Создаем фигуру
        fig, ax = plt.subplots(figsize=(12, 8))

        # 1. Контурный график целевой функции
        contour = ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.5)
        fig.colorbar(contour, label='Значение целевой функции (Z)')
        ax.contour(X, Y, Z, levels=20, colors='k', alpha=0.3, linewidths=0.5)

        # 2. Ограничения
        colors = plt.cm.tab10.colors
        for i in range(len(b)):
            if A[i, 1] != 0:
                y_line = (b[i] - A[i, 0] * x) / A[i, 1]
                ax.plot(x, y_line, '--', color=colors[i], 
                        linewidth=2, label=f"{A[i, 0]}x1 + {A[i, 1]}x2 ≤ {b[i]}")
                ax.fill_between(x, y_line, bounds[0], color=colors[i], alpha=0.1)

        # 3. Градиенты (если включено)
        if show_gradients:
            grad_X = Q[0,0]*X + 0.5*(Q[0,1]+Q[1,0])*Y + c[0]
            grad_Y = Q[1,1]*Y + 0.5*(Q[0,1]+Q[1,0])*X + c[1]
            step = 15
            ax.quiver(X[::step, ::step], Y[::step, ::step], 
                     -grad_X[::step, ::step], -grad_Y[::step, ::step], 
                     color='blue', scale=30, width=0.003, alpha=0.7, label='Антиградиент')

        # 4. Решения разными методами
        marker_styles = ['o', 's', 'D', '^', 'v', 'p', '*', 'h']
        for i, (method, solution) in enumerate(solutions.items()):
            ax.scatter(solution[0], solution[1], 
                      color=colors[i+len(b)], 
                      marker=marker_styles[i % len(marker_styles)],
                      s=150, 
                      edgecolors='k',
                      linewidth=1.5,
                      label=f'{method}: ({solution[0]:.2f}, {solution[1]:.2f})',
                      zorder=10)

        # Настройки графика
        ax.set_xlabel('x1', fontsize=12)
        ax.set_ylabel('x2', fontsize=12)
        ax.set_title('Сравнение методов решения QP', fontsize=14)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(bounds)
        ax.set_ylim(bounds)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_convergence(history: dict[str, list[float]]):
        """
        Визуализация сходимости разных методов.
        
        Параметры:
            history - словарь {метод: список значений целевой функции по итерациям}
        """
        plt.figure(figsize=(10, 6))
        
        for method, values in history.items():
            plt.plot(values, '.-', label=method, linewidth=2, markersize=8)
        
        plt.xlabel('Итерация', fontsize=12)
        plt.ylabel('Значение целевой функции', fontsize=12)
        plt.title('Сравнение сходимости методов', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def plot_constraints(A: np.ndarray, b: np.ndarray, solution: np.ndarray = None):
        """Рисует ограничения и решение для 2D задач."""
        if A.shape[1] != 2:
            raise ValueError("Визуализация доступна только для 2D задач.")
        
        plt.figure(figsize=(10, 6))
        x = np.linspace(0, 10, 400)
        
        # Отрисовка каждого ограничения A_i1*x1 + A_i2*x2 <= b_i
        colors = plt.cm.tab10.colors
        for i in range(len(b)):
            a1, a2 = A[i, 0], A[i, 1]
            y = (b[i] - a1 * x) / a2
            plt.plot(x, y, label=f"{a1:.1f}x1 + {a2:.1f}x2 ≤ {b[i]:.1f}", color=colors[i])
            plt.fill_between(x, y, 0, alpha=0.1, color=colors[i])
        
        # Решение
        if solution is not None:
            plt.scatter(solution[0], solution[1], color='red', s=100, label='Оптимум', zorder=5)
        
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.grid(True)
        plt.legend()
        plt.title('Область допустимых решений')
        plt.show()

    def plot_quadratic_solution(Q: np.ndarray, c: np.ndarray, A: np.ndarray, b: np.ndarray, solution: np.ndarray):
        """Визуализация решения QP с цветовой легендой и градиентами."""
        n = len(c)
        if n != 2:
            print("Визуализация доступна только для 2D задач.")
            return

        # Область решений
        x = np.linspace(-1, 5, 100)
        y = np.linspace(-1, 5, 100)
        X, Y = np.meshgrid(x, y)
        Z = 0.5 * (Q[0,0]*X**2 + (Q[0,1]+Q[1,0])*X*Y + Q[1,1]*Y**2) + c[0]*X + c[1]*Y

        # Градиенты (для стрелок)
        grad_X = Q[0,0]*X + 0.5*(Q[0,1]+Q[1,0])*Y + c[0]  # Производная по x1
        grad_Y = Q[1,1]*Y + 0.5*(Q[0,1]+Q[1,0])*X + c[1]  # Производная по x2

        # Создаём фигуру
        fig, ax = plt.subplots(figsize=(12, 8))

        # 1. Контурный график с цветовой легендой
        contour = ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)
        fig.colorbar(contour, label='Значение целевой функции (Z)')

        # 2. Стрелки градиента (уменьшаем плотность для наглядности)
        step = 30  # Рисуем каждую 10-ю стрелку
        ax.quiver(X[::step, ::step], Y[::step, ::step], 
                  -grad_X[::step, ::step], -grad_Y[::step, ::step], 
                  color='blue', scale=30, width=0.002, label='Антиградиент')

        # 3. Ограничения
        for i in range(len(b)):
            if A[i, 1] != 0:
                y_line = (b[i] - A[i, 0] * x) / A[i, 1]
                ax.plot(x, y_line, 'r--', lw=2, label=f"{A[i, 0]}x1 + {A[i, 1]}x2 ≤ {b[i]}")

        # 4. Оптимальная точка
        ax.scatter(solution[0], solution[1], color='red', s=200, 
                  label=f'Оптимум: ({solution[0]:.2f}, {solution[1]:.2f})', zorder=5)

        # Настройки графика
        ax.set_xlabel('x1', fontsize=12)
        ax.set_ylabel('x2', fontsize=12)
        ax.set_title('Квадратичная оптимизация: целевая функция и ограничения', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True)
        plt.tight_layout()
        plt.show()