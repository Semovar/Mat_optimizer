import numpy as np

class SimplexTableau:
    def __init__(self, c, A, b):
        """
        Инициализация симплекс-таблицы.
        :param c: Вектор коэффициентов целевой функции (1D numpy array).
        :param A: Матрица ограничений (2D numpy array).
        :param b: Вектор правых частей (1D numpy array).
        """
        self.c = c
        self.A = A
        self.b = b
        self.num_vars = len(c)
        self.num_constraints = len(b)
        self.tableau = self._initialize_tableau()
        self.basis = [f"s{i+1}" for i in range(self.num_constraints)]

    def _initialize_tableau(self):
        """Создаёт начальную симплекс-таблицу."""
        slack_vars = np.eye(self.num_constraints)
        tableau = np.hstack([self.A, slack_vars, self.b.reshape(-1, 1)])
        z_row = np.hstack([-self.c, np.zeros(self.num_constraints), 0])
        return np.vstack([tableau, z_row])
        
    def print_tableau(self, tableau=None):
        if tableau is None:
            tableau = self._initialize_tableau()
            
        rows, cols = tableau.shape
        #print("Симплекс-таблица:")
        header = ["Базис"] + list(range(cols))
        print(" ".join(["{:>8}".format(h) for h in header]))
        
        for i in range(rows):
            row_data = []
            if i < self.num_constraints:
                basis_var = f"S{i+1}"  # Названия базисных переменных (S1, S2...)
            else:
                basis_var = "Z"
                
            for j in range(cols):
                val = "{:>8.2f}".format(tableau[i, j])
                row_data.append(val)
            
            print("{:>8}: {}".format(basis_var, " ".join(row_data)))

    def solve(self):
        """Запускает симплекс-метод и возвращает решение."""
        while True:
            
            z_row = self.tableau[-1, :-1]
            pivot_col = np.argmin(z_row)
            if z_row[pivot_col] >= 0:
                print('Оптимум достигнут: ', ''.join(list(map(str, z_row))))
                break  # Оптимальное решение найдено

            rhs = self.tableau[:-1, -1]
            pivot_col_values = self.tableau[:-1, pivot_col]
            valid_rows = pivot_col_values > 0
            ratios = np.where(valid_rows, rhs / pivot_col_values, np.inf)
            pivot_row = np.argmin(ratios)

            # Обновляем базис
            self.basis[pivot_row] = f"x{pivot_col + 1}" if pivot_col < self.num_vars else f"s{pivot_col - self.num_vars + 1}"
            self._pivot(pivot_row, pivot_col)
            print('Новая таблица: ')
            self.print_tableau(self.tableau)

        return self._extract_solution()

    def _pivot(self, pivot_row, pivot_col):
        """Выполняет поворот вокруг элемента."""
        pivot_val = self.tableau[pivot_row, pivot_col]
        self.tableau[pivot_row, :] /= pivot_val
        for i in range(self.tableau.shape[0]):
            if i != pivot_row:
                self.tableau[i, :] -= self.tableau[i, pivot_col] * self.tableau[pivot_row, :]

    def _extract_solution(self):
        """Извлекает решение из таблицы."""
        solution = np.zeros(self.num_vars)
        for i, var in enumerate(self.basis):
            if var.startswith("x"):
                index = int(var[1:]) - 1
                solution[index] = self.tableau[i, -1]
        return solution, self.tableau[-1, -1]