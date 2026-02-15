import numpy as np

# -----------------------------
# Jednorodne stany stacjonarne (część reakcyjna)
# -----------------------------
def v_stac(a: float, m: float, mode: int = 1, add_delta = True) -> float:
    """
    Oblicza dodatni jednorodny punkt stacjonarny v* układu reakcyjnego
    wynikający z rozwiązania równania kwadratowego.

    Parametry
    a : float
        Parametr sterujący (np. dopływ/zasób).
    m : float
        Parametr liniowej utraty.
    mode : float
        1 podmienia delte na 0, 2 pozwala na ujemną delte.
    add_delta : bool
        Czy dodajemy delte? False zwraca odejmowanie

    Zwraca
    float
        Rozwiązanie v*, jeśli istnieje.
    """
    delta = a * a - 4 * m * m
    if delta < 0 and mode == 1:
        return 0
    if delta < 0 and mode == 2:
        raise ValueError("Ujemna delta.")

    if add_delta == True:
        return (a + np.sqrt(delta)) / (2 * m)
    else:
        return (a - np.sqrt(delta)) / (2 * m)


def u_stac(v: float, m: float) -> float:
    """
    Oblicza odpowiadający punkt stacjonarny u* dla zadanego v*,
    korzystając z relacji między zmiennymi.

    Parametry
    v : float
        Wartość stacjonarna v*.
    m : float
        Parametr modelu.

    Zwraca
    float
        Wartość u*. Jeśli v <= 0, zwraca 0.
    """
    if v <= 0.0:
        return 0.0
    return m / v

# -----------------------------
# Macierz drugich pochodnych
# -----------------------------
def D2(N: int) -> np.ndarray:
    """
    Konstruuje macierz dyskretnej drugiej pochodnej w 1D
    przy użyciu standardowego schematu trójdiagonalnego.

    Parametry
    N : int
        Wymiar macierzy.

    Zwraca
    np.ndarray
        Macierz NxN aproksymująca operator d²/dx²
        bez narzuconych warunków brzegowych.
    """
    return -2.0 * np.eye(N) + np.eye(N, k=1) + np.eye(N, k=-1)

# -----------------------------
# Laplasjan
# -----------------------------
def laplacian2D(Nx: int, Ny: int, h: float) -> np.ndarray:
    """
    Konstruuje macierz dyskretnego operatora Laplace’a w 2D
    na prostokątnej siatce o kroku h, wykorzystując iloczyn
    Kroneckera macierzy jednowymiarowych.

    Parametry
    Nx : int
        Liczba kolumn macierzy dla wartości x.
    Ny : int
        Liczba wierszy macierzy dla wartości y.
    h : float
        Krok siatki (zakładamy hx = hy).

    Zwraca
    np.ndarray
        Macierz (Nx*Ny) x (Nx*Ny) będąca
        dyskretnym operatorem Laplace’a.
    """
    D2x = D2(Nx)
    D2y = D2(Ny)
    Ix = np.eye(Nx)
    Iy = np.eye(Ny)
    return (np.kron(Iy, D2x) + np.kron(D2y, Ix)) / (h * h)

# -----------------------------
# Dyskretyzacja przestrzeni
# -----------------------------
def make_grid(Lx: float, Ly: float, Nx: int, Ny: int):
    """
    Generuje jednorodną siatkę prostokątną na obszarze
    [0, Lx] x [0, Ly] wraz z krokiem przestrzennym.

    Parametry
    Lx : float
        Długość domeny w kierunku x.
    Ly : float
        Długość domeny w kierunku y.
    Nx : int
        Liczba wartości dla x.
    Ny : int
        Liczba wartości dla y.

    Zwraca
        (x, y, X, Y, h), gdzie:
        x, y  – wektory współrzędnych,
        X, Y  – macierze siatki (meshgrid),
        h     – krok przestrzenny (hx = hy).

    Wyjątki
    ValueError
        Gdy kroki w kierunku x i y nie są równe.
    """
    x = np.linspace(0.0, Lx, Nx)
    y = np.linspace(0.0, Ly, Ny)
    X, Y = np.meshgrid(x, y)
    h = x[1] - x[0]
    if x[1] - x[0] != y[1] - y[0]:
        raise ValueError("")
    return x, y, X, Y, h

# -----------------------------
# Warunki brzegowe Dirichleta
# -----------------------------
def dirichlet_boundary_mask(X: np.ndarray, Y: np.ndarray, Lx: float, Ly: float) -> np.ndarray:
    """
    Wyznacza maskę logiczną dla warunków brzegowych
    Dirichleta na prostokątnej domenie.

    Parametry
    X, Y : np.ndarray
        Macierze współrzędnych siatki.
    Lx, Ly : float
        Wymiary domeny.

    Zwraca
    np.ndarray
        Spłaszczona maska typu bool wskazująca punkty
        należące do brzegu obszaru.
    """
    Xf = X.flatten()
    Yf = Y.flatten()
    boundary = (Xf == 0.0) | (Xf == Lx) | (Yf == 0.0) | (Yf == Ly)
    return boundary

# -----------------------------
# Jacobian
# -----------------------------
def jacobian(u_stac: float, v_stac: float, m: float) -> np.ndarray:
    return np.array([[-1 - v_stac**2, -2 * u_stac * v_stac],
                     [v_stac**2, 2 * u_stac * v_stac - m]])

# -----------------------------
# Czy punkt jest stabilny?
# -----------------------------
def is_stable(J: np.ndarray) -> bool:
    trJ = np.trace(J)
    detJ = np.linalg.det(J)
    return (trJ < 0 and detJ > 0)

