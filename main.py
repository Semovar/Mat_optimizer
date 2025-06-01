import numpy as np
from core.linear_solver import LinearSolver
from core.branch_and_bound import BranchAndBound
from core.io import print_problem, read_problem
from core.qp_solver import QuadraticSolver
from core.visualizer import Visualizer
from time import time

'''
# Данные задачи
c = np.array([3, 2])
A = np.array([[2, 1], [1, 1]])
b = np.array([10, 8])
'''

# Чтение задачи
params, c, A, b = read_problem("input.csv")

# Вывод системы
print_problem(params, c, A, b)

# Решение
solver = LinearSolver()
solution, objective = solver.solve(c, A, b, task_type=params["type"])

# Вывод
print("Решение:", solution)
print("Целевая функция:", objective)

# Решение симплекс-методом
s1s = time()
solver = LinearSolver()
solution, obj = solver.solve(c, A, b, task_type=params["type"])
s1e = time()
print("Решение:", solution, "Z =", obj)


# Решение M-методом (если нет начального базиса)
s2s = time()
try:
    solution_M, obj_M = solver.solve_M_method(c, A, b, task_type=params["type"])
    print("Решение (M-метод):", solution_M, "Z =", obj_M)
except ValueError as e:
    print("Ошибка:", e)
s2e = time()


# Целочисленное решение
s3s = time()
bb = BranchAndBound()
bb_solution, bb_obj = bb.solve(c, A, b, integer_vars=[0, 1], task_type=params["type"])
s3e = time()
print("Целочисленное решение:", bb_solution, "Z =", bb_obj)

# Решение методом внутренней точки
s4s = time()
solution, objective = solver.solve_interior_point(c, A, b)
s4e = time()
print("Решение (внутренняя точка):", solution, "Z = ", objective)

print('quadratic check:')


# Данные для QP
Q = np.array([[2, 0], [0, 2]])  # Минимизировать x1² + x2²
c = np.array([-8, -6])           # Линейная часть: -8x1 -6x2
A = np.array([[1, 1], [-1, 2]])  # Ограничения
b = np.array([3, 2])
bounds = [(0, None), (0, None)]  # Границы переменных

# Создаем экземпляр решателя
qp_solver = QuadraticSolver()

# Тестируем все методы
methods = {
    "Градиентный метод": qp_solver.solve,
    "Метод Ньютона": qp_solver.solve_newton_qp,
    "Метод активных множеств": qp_solver.solve_active_set
}

results = {}
timings = {}

for name, method in methods.items():
    start_time = time()
    if name == "Метод активных множеств":
        solution, obj = method(Q, c, A, b, bounds=bounds)
    else:
        solution, obj = method(Q, c, A, b, bounds=bounds)
    end_time = time()
    
    results[name] = (solution, obj)
    timings[name] = (end_time - start_time) * 1000  # в миллисекундах

# Вывод результатов
print("\nРезультаты решения QP задачи:")
for name, (solution, obj) in results.items():
    print(f"{name}:")
    print(f"  Решение: {solution}")
    print(f"  Значение целевой функции: {obj:.4f}")
    print(f"  Время выполнения: {timings[name]:.2f} мс\n")

# Визуализация (если есть класс Visualizer)
try:
    Visualizer.plot_quadratic_solution(Q, c, A, b, results["Метод активных множеств"][0])
except NameError:
    print("Визуализация недоступна (не определен класс Visualizer)")