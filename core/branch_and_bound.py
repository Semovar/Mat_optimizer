from .linear_solver import LinearSolver
from .qp_solver import QuadraticSolver
import numpy as np

class BranchAndBound:
    def __init__(self):
        self.solver = LinearSolver()

    def solve(self, c, A, b, integer_vars, task_type="max"):
        best_solution = None
        best_obj = float('-inf') if task_type == "max" else float('inf')
        queue = [(A.copy(), b.copy())]

        while queue:
            current_A, current_b = queue.pop(0)
            try:
                solution, obj = self.solver.solve(c, current_A, current_b, task_type)
                if all(solution[i].is_integer() for i in integer_vars):
                    if (task_type == "max" and obj > best_obj) or (task_type == "min" and obj < best_obj):
                        best_solution, best_obj = solution, obj
                else:
                    # Ветвление по первой нецелой переменной
                    for i in integer_vars:
                        if not solution[i].is_integer():
                            new_A1 = np.vstack([current_A, [1 if j == i else 0 for j in range(len(c))]])
                            new_b1 = np.append(current_b, np.floor(solution[i]))
                            new_A2 = np.vstack([current_A, [-1 if j == i else 0 for j in range(len(c))]])
                            new_b2 = np.append(current_b, -np.ceil(solution[i]))
                            queue.extend([(new_A1, new_b1), (new_A2, new_b2)])
                            break
            except:
                continue

        return best_solution, best_obj

class QPBranchAndBound:
    def __init__(self):
        self.solver = QuadraticSolver()

    def solve(
        self,
        Q: np.ndarray,
        c: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
        integer_vars: list  # Индексы целочисленных переменных
    ) -> tuple[np.ndarray, float]:
        best_solution = None
        best_obj = float('inf')
        queue = [(A.copy(), b.copy())]  # Очередь задач

        while queue:
            current_A, current_b = queue.pop(0)
            try:
                solution, obj = self.solver.solve(Q, c, current_A, current_b)
                
                # Проверка целочисленности
                is_integer = all(solution[i].is_integer() for i in integer_vars)
                if is_integer and obj < best_obj:
                    best_solution, best_obj = solution, obj
                elif not is_integer:
                    # Ветвление по первой нецелой переменной
                    for i in integer_vars:
                        if not solution[i].is_integer():
                            # Добавляем ограничения x_i <= floor и x_i >= ceil
                            new_A1 = np.vstack([current_A, [1 if j == i else 0 for j in range(len(c))]])
                            new_b1 = np.append(current_b, np.floor(solution[i]))
                            new_A2 = np.vstack([current_A, [-1 if j == i else 0 for j in range(len(c))]])
                            new_b2 = np.append(current_b, -np.ceil(solution[i]))
                            queue.extend([(new_A1, new_b1), (new_A2, new_b2)])
                            break
            except:
                continue

        return best_solution, best_obj