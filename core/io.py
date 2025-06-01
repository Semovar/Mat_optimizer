import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Union
from colorama import Fore, init

def read_problem(file_path: str) -> Tuple[Dict[str, Union[str, int]], np.ndarray, np.ndarray, np.ndarray]:
    """
    Читает задачу из файла CSV или Excel.
    
    Формат файла:
    - Первая строка: "type,num_vars" (например, "max,2").
    - Вторая строка: коэффициенты целевой функции (например, "3,2").
    - Остальные строки: ограничения в формате "a1,a2,...,an,rhs" (например, "2,1,10").
    
    :param file_path: Путь к файлу.
    :return: (params, c, A, b), где:
        - params: {"type": "max/min", "num_vars": int}.
        - c: коэффициенты целевой функции.
        - A: матрица ограничений.
        - b: правые части ограничений.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл {file_path} не найден.")

    try:
        # Чтение файла как текста (для CSV)
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        if len(lines) < 2:
            raise ValueError("Файл должен содержать минимум 2 строки: заголовок и целевую функцию.")

        # Парсинг заголовка (первая строка)
        header = lines[0].split(',')
        if len(header) != 2:
            raise ValueError("Заголовок должен быть в формате 'type,num_vars' (например, 'max,2').")

        task_type, num_vars = header[0].lower(), int(header[1])
        if task_type not in ["max", "min"]:
            raise ValueError("Тип задачи должен быть 'max' или 'min'.")

        # Парсинг целевой функции (вторая строка)
        c = np.array([float(x) for x in lines[1].split(',')])
        if len(c) != num_vars:
            raise ValueError(f"Ожидается {num_vars} коэффициентов в целевой функции. Получено: {len(c)}")

        # Парсинг ограничений (остальные строки)
        A = []
        b = []
        for line in lines[2:]:
            row = [float(x) for x in line.split(',')]
            if len(row) != num_vars + 1:
                raise ValueError(f"Ожидается {num_vars + 1} значений в строке ограничения. Получено: {len(row)}")
            A.append(row[:-1])
            b.append(row[-1])

        A = np.array(A, dtype=float)
        b = np.array(b, dtype=float)

        return {"type": task_type, "num_vars": num_vars}, c, A, b

    except Exception as e:
        raise ValueError(f"Ошибка при чтении файла: {str(e)}")


def save_solution(file_path: str, solution: np.ndarray, objective: float, params: Dict):
    """Сохраняет решение в CSV."""
    df = pd.DataFrame({
        "Variable": [f"x{i+1}" for i in range(len(solution))],
        "Value": solution,
    })
    df.loc[len(df)] = ["Objective", objective]
    df.to_csv(file_path, index=False)
    
def print_problem(params: Dict, c: np.ndarray, A: np.ndarray, b: np.ndarray):
    """
    Выводит задачу в человекочитаемом формате.
    
    Пример:
    >>> Задача: максимизировать
    >>> Целевая функция: 3.0*x1 + 2.0*x2
    >>> Ограничения:
    >>> 2.0*x1 + 1.0*x2 <= 10.0
    >>> 1.0*x1 + 1.0*x2 <= 8.0
    """
    init(autoreset=True)
    task_type = "максимизировать" if params["type"] == "max" else "минимизировать"
    num_vars = params["num_vars"]
    
    # Целевая функция
    objective_parts = []
    for i in range(num_vars):
        if c[i] != 0:
            sign = "+" if c[i] > 0 and i > 0 else ""
            objective_parts.append(f"{sign}{c[i]:.1f}*x{i+1}")
    objective_str = " ".join(objective_parts)

    print(Fore.GREEN + f"Задача: {task_type}")
    print(Fore.BLUE + f"Целевая функция: {objective_str}")
    
    print(f"Задача: {task_type}")
    print(f"Целевая функция: {objective_str}")
    print(Fore.RED + "Ограничения:")
    
    # Ограничения
    for row_idx in range(A.shape[0]):
        constraint_parts = []
        for col_idx in range(num_vars):
            coeff = A[row_idx, col_idx]
            if coeff != 0:
                sign = "+" if coeff > 0 and col_idx > 0 else ""
                constraint_parts.append(f"{sign}{coeff:.1f}*x{col_idx+1}")
        constraint_str = " ".join(constraint_parts) if constraint_parts else "0"
        print(Fore.RED + f"{constraint_str} <= {b[row_idx]:.1f}")