from .simplex_tableau import SimplexTableau
import numpy as np

class LinearSolver:
    def __init__(self):
        self.M = 1e6  # Штраф для M-метода

    def solve(self, c, A, b, task_type="max"):
        """
        Решает ЗЛП симплекс-методом.
        :param task_type: "max" или "min".
        """
        if task_type == "min":
            c = -c  # Преобразуем задачу на минимум к максимизации

        solver = SimplexTableau(c, A, b)
        solution, objective = solver.solve()

        if task_type == "min":
            objective = -objective
        return solution, objective

    def solve_M_method(self, c, A, b, task_type="max"):
        """Решает задачу с искусственным базисом."""
        num_constraints = len(b)
        artificial_vars = np.eye(num_constraints)
        extended_A = np.hstack([A, artificial_vars])
        extended_c = np.hstack([c, [-self.M] * num_constraints])

        # Решаем вспомогательную задачу
        solver = SimplexTableau(extended_c, extended_A, b)
        solution, _ = solver.solve()

        # Проверяем искусственные переменные
        if any(var.startswith("a") for var in solver.basis):
            raise ValueError("Нет допустимого решения!")

        # Решаем исходную задачу
        return self.solve(c, A, b, task_type)   
        
    def solve_interior_point(self, c, A, b, max_iter=1000, tol=1e-6, mu=10):
        """
        Решает ЗЛП методом внутренней точки.
        Minimize: c^T @ x
        Subject to: A @ x <= b, x >= 0
        """
        
        def _line_search_interior_point(x, s, delta_x, delta_s):
            """
            Выполняет поиск шага для обеспечения положительности x и s.
            """
            alpha = 1.0
            while np.any(x + alpha * delta_x <= 0) or np.any(s + alpha * delta_s <= 0):
                alpha *= 0.5
            return alpha
        n = len(c)
        m = len(b)
        
        # Инициализация переменных
        x = np.ones(n)  # Начальное приближение (должно быть > 0)
        s = np.ones(m)  # Слабые переменные для ограничений
        y = np.zeros(m)  # Двойственные переменные
        
        for iteration in range(max_iter):
            # Вычисление градиента и гессиана для барьерной функции
            grad_x = c - A.T @ y + mu / x
            grad_y = b - A @ x - s
            grad_s = -y + mu / s
            
            # Гессиан для барьерной функции
            H_x = np.diag(mu / (x ** 2))
            H_s = np.diag(mu / (s ** 2))
            
            # Сборка системы уравнений
            H = np.block([
                [H_x, np.zeros((n, m)), -A.T],
                [np.zeros((m, n)), H_s, np.eye(m)],
                [-A, np.eye(m), np.zeros((m, m))]
            ])
            grad = np.hstack([grad_x, grad_s, grad_y])
            
            # Решение системы уравнений
            delta = np.linalg.solve(H, -grad)
            delta_x, delta_s, delta_y = delta[:n], delta[n:n+m], delta[n+m:]
            
            # Обновление переменных
            alpha = _line_search_interior_point(x, s, delta_x, delta_s)
            x += alpha * delta_x
            s += alpha * delta_s
            y += alpha * delta_y
            
            # Проверка на сходимость
            if np.linalg.norm(grad) < tol:
                break
        
        objective = c @ x
        return x, objective


        