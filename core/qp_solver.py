import numpy as np
from typing import Tuple, Optional

class QuadraticSolver:
    def __init__(self):
        self.max_iter = 1000  # Максимальное число итераций
        self.tol = 1e-6      # Допустимая погрешность

    def solve(
        self,
        Q: np.ndarray,       # Квадратичная часть (n x n)
        c: np.ndarray,       # Линейная часть (n)
        A: Optional[np.ndarray] = None,  # Ограничения-неравенства (m x n)
        b: Optional[np.ndarray] = None,  # Правые части неравенств (m)
        A_eq: Optional[np.ndarray] = None,  # Ограничения-равенства (k x n)
        b_eq: Optional[np.ndarray] = None,  # Правые части равенств (k)
        bounds: Optional[list] = None       # Границы переменных [(low, high), ...]
    ) -> Tuple[np.ndarray, float]:
        """
        Решает задачу квадратичного программирования:
        Minimize:  (1/2) * x^T @ Q @ x + c^T @ x
        Subject to: A @ x <= b, A_eq @ x == b_eq, bounds
        """
        # Проверки входных данных
        n = len(c)
        if Q.shape != (n, n):
            raise ValueError("Матрица Q должна быть размером n x n.")
        
        # Инициализация
        x = np.zeros(n) if bounds is None else np.array([b[0] for b in bounds])
        
        # Метод: активные ограничения + градиентный спуск
        for _ in range(self.max_iter):
            grad = Q @ x + c  # Градиент целевой функции
            
            # Проверка на сходимость
            if np.linalg.norm(grad) < self.tol:
                break
            
            # Шаг градиентного спуска (упрощённый)
            step_size = 0.01
            x_new = x - step_size * grad
            
            # Проекция на ограничения (упрощённо)
            if A is not None and b is not None:
                for _ in range(10):  # Несколько итераций для соблюдения ограничений
                    violation = A @ x_new - b
                    if np.any(violation > 0):
                        x_new -= 0.1 * (A.T @ np.maximum(violation, 0))
            
            x = x_new
        
        objective = 0.5 * x.T @ Q @ x + c.T @ x
        return x, objective
    
    #(Q, c, A, b, bounds, 1000)    
    def solve_newton_qp(self, Q, c, A=None, b=None, A_eq=None, b_eq=None, bounds=None, max_iter=1000, tol=1e-6):
        def _line_search_newton_qp(Q, c, A, b, A_eq, b_eq, x, delta_x):
            alpha = 1.0
            for _ in range(100):  # Максимум 100 попыток
                x_new = x + alpha * delta_x
                
                # Проверка границ
                if bounds is not None:
                    x_new = np.clip(x_new, 
                                    a_min=[low for low, high in bounds], 
                                    a_max=[high if high is not None else np.inf for low, high in bounds])
                
                # Проверка неравенств
                if A is not None and np.any(A @ x_new > b):
                    alpha *= 0.5
                    continue
                    
                # Проверка равенств (если они есть)
                if A_eq is not None and np.linalg.norm(A_eq @ x_new - b_eq) > 1e-6:
                    alpha *= 0.5
                    continue
                    
                break
            return alpha
        n = len(c)
        if bounds is None:
            x = np.zeros(n)
        else:
            x = np.array([(low + (high if high is not None else low + 1.0)) / 2 for low, high in bounds])

        for _ in range(max_iter):
            grad = Q @ x + c
            hessian = Q

            # Учёт равенств через систему KKT
            if A_eq is not None:
                # Расширенная система: [Q, A_eq^T; A_eq, 0] @ [x; lambda] = [-c; b_eq]
                m_eq = A_eq.shape[0]
                KKT_matrix = np.block([[Q, A_eq.T], [A_eq, np.zeros((m_eq, m_eq))]])
                rhs = np.concatenate([-grad, b_eq - A_eq @ x])
                solution = np.linalg.solve(KKT_matrix, rhs)
                delta_x = solution[:n]
            else:
                delta_x = np.linalg.solve(hessian, -grad)

            step_size = _line_search_newton_qp(Q, c, A, b, A_eq, b_eq, x, delta_x)
            x_new = x + step_size * delta_x

            if np.linalg.norm(grad) < tol:
                break
            x = x_new

        return x, 0.5 * x.T @ Q @ x + c.T @ x
        
    def solve_interior_point(self, Q, c, A=None, b=None, A_eq=None, b_eq=None, bounds=None, max_iter=100, tol=1e-6):
        n = len(c)
        x = np.ones(n)  # Начальная точка внутри области
        mu = 1.0  # Параметр барьера

        for _ in range(max_iter):
            # Целевая функция с логарифмическим барьером
            def barrier_func(x):
                obj = 0.5 * x.T @ Q @ x + c.T @ x
                if A is not None:
                    obj -= mu * np.sum(np.log(b - A @ x))  # Барьер для неравенств
                if bounds is not None:
                    obj -= mu * np.sum(np.log(x - np.array([low for low, high in bounds])))  # Барьер для x >= low
                return obj

            # Решаем безусловную задачу (например, методом Ньютона)
            x, _ = self.solve_newton_qp(Q, c, A_eq=A_eq, b_eq=b_eq, bounds=None)
            
            mu *= 0.1  # Уменьшаем параметр барьера
            if mu < tol:
                break

        return x, 0.5 * x.T @ Q @ x + c.T @ x
        
    def solve_active_set(
        self,
        Q: np.ndarray,
        c: np.ndarray,
        A: Optional[np.ndarray] = None,
        b: Optional[np.ndarray] = None,
        A_eq: Optional[np.ndarray] = None,
        b_eq: Optional[np.ndarray] = None,
        bounds: Optional[list[Tuple]] = None,
        max_iter: int = 100,
        tol: float = 1e-6
    ) -> Tuple[np.ndarray, float]:
        """
        Метод активных множеств для квадратичного программирования.
        Возвращает решение и значение целевой функции.
        """
        n = len(c)
        
        # 1. Начальная допустимая точка
        if bounds is None:
            x = np.zeros(n)
        else:
            x = np.array([(low + (high if high is not None else low + 1.0)) / 2 
                          for low, high in bounds])
        
        # 2. Инициализация активного множества
        active_set = set()
        if A_eq is not None:
            for i in range(A_eq.shape[0]):
                active_set.add(('eq', i))
        
        for iter in range(max_iter):
            # 3. Решаем подзадачу с активными ограничениями
            grad = Q @ x + c
            
            # Проверка оптимальности
            if np.linalg.norm(grad) < tol:
                break
                
            # 4. Определяем направление поиска
            try:
                if active_set:
                    # Строим матрицу активных ограничений
                    A_active = []
                    b_active = []
                    for constr in active_set:
                        if constr[0] == 'ineq':
                            A_active.append(A[constr[1]])
                            b_active.append(b[constr[1]])
                        elif constr[0] == 'eq':
                            A_active.append(A_eq[constr[1]])
                            b_active.append(b_eq[constr[1]])
                    
                    A_active = np.array(A_active)
                    b_active = np.array(b_active)
                    
                    # Решаем систему KKT
                    KKT = np.block([[Q, A_active.T], 
                                   [A_active, np.zeros((len(active_set), len(active_set)))]])
                    rhs = np.concatenate([-grad, np.zeros(len(active_set))])
                    step = np.linalg.solve(KKT, rhs)[:n]
                else:
                    step = -np.linalg.solve(Q, grad)
            except np.linalg.LinAlgError:
                step = -grad  # Если система вырождена, используем градиент
                
            # 5. Линейный поиск
            alpha = 1.0
            if A is not None:
                for i in range(A.shape[0]):
                    if ('ineq', i) not in active_set and A[i] @ step > 0:
                        alpha_i = (b[i] - A[i] @ x) / (A[i] @ step)
                        alpha = min(alpha, alpha_i)
            
            x_new = x + alpha * step
            
            # 6. Обновление активного множества
            if alpha < 1.0:
                # Добавляем новое активное ограничение
                for i in range(A.shape[0]):
                    if ('ineq', i) not in active_set and abs(A[i] @ x_new - b[i]) < tol:
                        active_set.add(('ineq', i))
            else:
                # Проверяем множители Лагранжа
                if active_set:
                    KKT_sol = np.linalg.solve(KKT, np.concatenate([-grad, np.zeros(len(active_set))]))
                    lambdas = KKT_sol[n:]
                    
                    # Удаляем ограничения с отрицательными множителями
                    to_remove = []
                    for idx, constr in enumerate(active_set):
                        if constr[0] == 'ineq' and lambdas[idx] < -tol:
                            to_remove.append(constr)
                    
                    for constr in to_remove:
                        active_set.remove(constr)
            
            x = x_new
        
        return x, 0.5 * x.T @ Q @ x + c.T @ x