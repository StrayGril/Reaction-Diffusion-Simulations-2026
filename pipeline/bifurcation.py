import numpy as np
from scipy.linalg import lu_factor, lu_solve
from pipeline.model_core import (
    v_stac, u_stac,
    laplacian2D,
    make_grid,
    dirichlet_boundary_mask)

# ---------------------------------------
# Niejawna dyfuzja
# ---------------------------------------
def precompute_diffusion(Nx: int, Ny: int, h: float, ht: float, d1: float, d2: float):
    """
    Przygotowuje rozkład formy LU macierzy odpowiadających
    niejawnemu schematowi czasowemu dla części dyfuzyjnej.

    W każdym kroku czasowym rozwiązywany jest układ:
        (I - ht * d1 * L) u^{n+1} = u^n + ht * f_u(u^n, v^n)
        (I - ht * d2 * L) v^{n+1} = v^n + ht * f_v(u^n, v^n),
        gdzie:
            L  – dyskretny operator Laplace’a,
            ht – krok czasowy,
            d1, d2 – współczynniki dyfuzji,
            f_u, f_v – część reakcyjna.

    Parametry
    Nx, Ny : int
        Liczba wartości dla x i y.
    h : float
        Krok przestrzenny (zakładamy hx = hy).
    ht : float
        Krok czasowy.
    d1 : float
        Współczynnik dyfuzji pierwszej zmiennej.
    d2 : float
        Współczynnik dyfuzji drugiej zmiennej.

    Zwraca
        (lu_Au, lu_Av), gdzie każdy element reprezentuje rozkład macierzy:
            Au = I - ht * d1 * L,
            Av = I - ht * d2 * L.
    """
    L = laplacian2D(Nx, Ny, h)
    I = np.eye(Nx * Ny)

    Au = I - ht * d1 * L
    Av = I - ht * d2 * L

    return lu_factor(Au), lu_factor(Av)

# -----------------------------
# Pochodne obliczone jawnie
# -----------------------------
def reaction(u: np.ndarray, v: np.ndarray, a: float, m: float):
    """
    Oblicza część reakcyjną układu (bez dyfuzji).

    Równania reakcyjne mają postać:
        du/dt = a - u - u v²
        dv/dt = u v² - m v

    Parametry
    u, v : np.ndarray
        Aktualne wartości zmiennych w chwili t_n.
    a : float
        Parametr wilgotności.
    m : float
        Parametr śmiertelności.

    Zwraca
        (du, dv) — wartości pochodnych czasowych części reakcyjnej.
    """
    du = a - u - u * (v * v)
    dv = u * (v * v) - m * v
    return du, dv

# -----------------------------
# Krok czasowy
# -----------------------------
def step_reaction_diffusion(
    u: np.ndarray,
    v: np.ndarray,
    a: float,
    m: float,
    ht: float,
    lu_Au,
    lu_Av,
    brzeg: np.ndarray,
):
    """
    Wykonuje jeden krok czasowy metodą:
        reakcja jawnie (Euler),
        dyfuzja niejawnie (rozwiązanie układu liniowego).

    Schemat numeryczny ma postać:
        (I - ht d1 L) u^{n+1} = u^n + ht f_u(u^n, v^n)
        (I - ht d2 L) v^{n+1} = v^n + ht f_v(u^n, v^n)
    gdzie:
        L  – dyskretny operator Laplace’a,
        ht – krok czasowy,
        d1, d2 – współczynniki dyfuzji,
        f_u, f_v – część reakcyjna.

    Parametry
    u, v : np.ndarray
        Wartości w chwili t^n.
    a, m : float
        Parametry modelu.
    ht : float
        Krok czasowy.
    lu_Au, lu_Av :
        Rozkłady LU macierzy (I - ht d L).
    brzeg : np.ndarray
        Maska warunków brzegowych Dirichleta.
    clip_nonnegative : bool
        Czy wymuszać nieujemność rozwiązania.

    Zwraca
        (u_new, v_new) — wartości w chwili t_{n+1}.
    """
    # część reakcyjna
    du, dv = reaction(u, v, a, m)

    # prawa strona układu liniowego
    ru = u + ht * du
    rv = v + ht * dv

    # wymuszenie Dirichleta przed rozwiązaniem
    ru[brzeg] = 0
    rv[brzeg] = 0

    # rozwiązanie układów liniowych
    u_new = lu_solve(lu_Au, ru)
    v_new = lu_solve(lu_Av, rv)

    # wymuszenie nieujemności
    u_new = np.maximum(u_new, 0)
    v_new = np.maximum(v_new, 0)

    u_new[brzeg] = 0
    v_new[brzeg] = 0

    return u_new, v_new

# -----------------------------
# Symulacja do punktu stacjonarnego
# -----------------------------
def simulate_to_steady(
    u0: np.ndarray,
    v0: np.ndarray,
    a: float,
    m: float,
    ht: float,
    lu_Au,
    lu_Av,
    brzeg: np.ndarray,
    krok_max: int = 500,
    eps: float = 1e-8,
):
    """
    Iteruje schemat czasowy aż do osiągnięcia stanu stacjonarnego, gdzie stosowane kryterium to:
        || v^{n+1} - v^n || < eps

    Parametry
    u0, v0 : np.ndarray
        Warunki początkowe.
    a, m : float
        Parametry modelu.
    ht : float
        Krok czasowy.
    lu_Au, lu_Av :
        Rozkłady LU macierzy dyfuzyjnych.
    brzeg : np.ndarray
        Maska warunków Dirichleta.
    krok_max : int
        Maksymalna liczba iteracji.
    eps : float
        Tolerancja zbieżności.

    Zwraca
        (u, v, i) — rozwiązanie stacjonarne oraz liczba wykonanych iteracji.
    """
    u = u0.copy()
    v = v0.copy()

    for i in range(krok_max):
        v_prev = v.copy()
        u, v = step_reaction_diffusion(u, v, a, m, ht, lu_Au, lu_Av, brzeg=brzeg)

        # test zbieżności
        if np.linalg.norm((v - v_prev)[~brzeg]) < eps:
            return u, v, i + 1

    return u, v, krok_max


# ---------------------------------------
# Kontynuacja symulacji
# ---------------------------------------
def continuation_sweep(
    a_values: np.ndarray,
    u_init: np.ndarray,
    v_init: np.ndarray,
    m: float,
    ht: float,
    lu_Au,
    lu_Av,
    brzeg: np.ndarray,
    krok_max: int = 500,
    eps: float = 1e-8,
    store_states: bool = True,
):
    """
    Wykonuje kontynuację numeryczną względem parametru a.

    Dla każdej wartości a z a_values:
        1. Rozwiązuje układ do stanu stacjonarnego.
        2. Używa otrzymanego rozwiązania jako warunku początkowego
           dla kolejnej wartości parametru.
        3. Zapisuje otrzymane wyniki.

    Rejestrowane miary:
        μ_v  = średnia biomasa,
        max_v = maksimum biomasy.

    Parametry
    ----------
    a_values : np.ndarray
        Wartości parametru a (w ustalonej kolejności).
    u_init, v_init : np.ndarray
        Warunki początkowe dla pierwszej wartości parametru.
    m : float
        Parametr modelu.
    ht : float
        Krok czasowy.
    lu_Au, lu_Av :
        Rozkłady LU macierzy dyfuzyjnych.
    brzeg : np.ndarray
        Maska warunków Dirichleta.
    krok_max : int
        Maksymalna liczba iteracji w solverze steady-state.
    eps : float
        Tolerancja zbieżności.
    store_states : bool
        Czy zapisywać pełne stany (u, v) dla każdej wartości a.

    Zwraca
        (avg, max, states)
    """
    u = u_init.copy()
    v = v_init.copy()

    avgs = []
    maxs = []
    states = [] if store_states else None

    for a in a_values:
        u, v, _ = simulate_to_steady(
            u, v, a, m, ht, lu_Au, lu_Av,
            brzeg=brzeg, krok_max=krok_max, eps=eps
        )

        avgs.append(float(np.mean(v)))
        maxs.append(float(np.max(v)))

        if store_states:
            states.append((u.copy(), v.copy()))

    out = {"avg": np.array(avgs), "max": np.array(maxs)}
    if store_states:
        out["states"] = states
    return out

# ---------------------------------------
# Estymacja tipping point
# ---------------------------------------
def estimate_tipping_point(a_values_desc: np.ndarray, max_series: np.ndarray) -> tuple[float, int]:
    """
    Szacuje punkt krytyczny (tipping point) na podstawie
    największego skoku maksimum biomasy.

    Algorytm:
        1. Oblicza różnice kolejnych wartości max(v).
        2. Wybiera indeks największej bezwzględnej zmiany.
        3. Przyjmuje punkt krytyczny jako środek przedziału
           pomiędzy dwiema sąsiednimi wartościami parametru.

    Parametry
    a_values_desc : np.ndarray
        Malejące wartości parametru a.
    max_series : np.ndarray
        Maksimum biomasy dla kolejnych wartości a.

    Zwraca
        (tp, idx), gdzie:
            tp  – przybliżony punkt krytyczny,
            idx – indeks odpowiadający największemu skokowi.
    """
    dv = np.abs(np.diff(max_series))
    idx = int(np.argmax(dv))
    tp = 0.5 * (a_values_desc[idx] + a_values_desc[idx + 1])
    return float(tp), idx

# ---------------------------------------
# Pełna symulacja bifurkacyjna
# ---------------------------------------
def run_bifurcation(
    m: float,
    d1: float,
    d2: float,
    Lx: float = 10,
    Ly: float = 10,
    Nx: int = 30,
    Ny: int = 30,
    ht: float = 0.025,
    krok_max: int = 500,
    eps: float = 1e-8,
    ha: float = 5e-4,
    amax_factor: float = 4.0,
):
    """
    Wykonuje pełną analizę bifurkacyjną względem parametru a.

    Algorytm:
        1. Generacja siatki przestrzennej oraz maski Dirichleta.
        2. Faktoryzacja macierzy dyfuzji (schemat niejawny).
        3. Kontynuacja dla malejących wartości a (przejście „w dół”).
        4. Identyfikacja punktu krytycznego (tipping point)
           na podstawie największego skoku max(v).
        5. Kontynuacja dla rosnących wartości a
           (przejście „w górę” z okolic punktu krytycznego).

    Parametry
    m : float
        Parametr śmiertelności biomasy.
    d1, d2 : float
        Współczynniki dyfuzji.
    Lx, Ly : float
        Wymiary domeny przestrzennej.
    Nx, Ny : int
        Liczba wartości dla x i y.
    ht : float
        Krok czasowy.
    krok_max : int
        Maksymalna liczba iteracji w solverze stanu stacjonarnego.
    eps : float
        Tolerancja zbieżności.
    ha : float
        Krok parametru a w procedurze kontynuacji.
    amax_factor : float
        Współczynnik wyznaczający maksymalną wartość parametru:
            a_max = amax_factor * m.

    Zwraca
        - serie parametrów (a_down, a_up),
        - odpowiadające średnie i maksima biomasy,
        - przybliżony punkt krytyczny tp,
        - indeks skoku tp_idx,
        - wartość referencyjną a = 2m,
        - informacje o siatce i parametrach numerycznych.
    """
    amax = amax_factor * m

    # siatka i brzeg
    _, _, X, Y, h = make_grid(Lx, Ly, Nx, Ny)
    brzeg = dirichlet_boundary_mask(X, Y, Lx, Ly)

    # LU dla dyfuzji
    lu_Au, lu_Av = precompute_diffusion(Nx, Ny, h, ht, d1, d2)

    # a malejące
    a_down = np.arange(amax, 0, -ha)

    # start na równowadze ODE dla amax
    v0 = v_stac(a_down[0], m)
    u0 = u_stac(v0, m)
    u_init = u0 * np.ones(Nx * Ny)
    v_init = v0 * np.ones(Nx * Ny)
    u_init[brzeg] = 0
    v_init[brzeg] = 0

    down = continuation_sweep(
        a_values=a_down,
        u_init=u_init,
        v_init=v_init,
        m=m,
        ht=ht,
        lu_Au=lu_Au,
        lu_Av=lu_Av,
        brzeg=brzeg,
        krok_max=krok_max,
        eps=eps,
        store_states=True,
    )

    tp, idx = estimate_tipping_point(a_down, down["max"])

    # stan startowy "tuż nad" tp (tak jak u Ciebie)
    idx_c = np.where(a_down > tp)[0][-1]
    u_start, v_start = down["states"][idx_c]

    a_up = np.arange(tp, amax, ha)

    up = continuation_sweep(
        a_values=a_up,
        u_init=u_start,
        v_init=v_start,
        m=m,
        ht=ht,
        lu_Au=lu_Au,
        lu_Av=lu_Av,
        brzeg=brzeg,
        krok_max=krok_max,
        eps=eps,
        store_states=False,
    )

    return {
        "a_down": a_down,
        "down_avg": down["avg"],
        "down_max": down["max"],
        "a_up": a_up,
        "up_avg": up["avg"],
        "up_max": up["max"],
        "tp": tp,
        "tp_idx": idx,
        "a_2m": 2.0 * m,
        "brzeg": brzeg,
        "grid": {"Lx": Lx, "Ly": Ly, "Nx": Nx, "Ny": Ny, "h": h},
        "params": {"m": m, "d1": d1, "d2": d2, "ht": ht, "krok_max": krok_max, "eps": eps, "ha": ha},
    }

# ---------------------------------------
# Wykres
# ---------------------------------------
def plot_bifurcation(result: dict, title: str | None = None, show: bool = True, ax=None):
    """
    Wizualizuje diagram bifurkacyjny względem parametru a.

    Na wykresie dla przejścia w dół (a malejące) oraz w górę (a rosnące) przedstawiane są:
        - średnia przestrzenna biomasy μ_v,
        - maksimum przestrzenne biomasy max(v),
        - wartość referencyjna a = 2m,
        - przybliżony punkt krytyczny (tipping point).

    Parametry
    result : dict
        parametry zwrócone przez funkcję run_bifurcation.
    title : str | None
        Tytuł wykresu.
    show : bool
        Czy wywołać plt.show().
    ax : matplotlib.axes.Axes | None
        Oś, na której ma zostać narysowany wykres.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    a_down = result["a_down"]
    a_up = result["a_up"]

    ax.scatter(a_down, result["down_avg"], s=8, color="cyan", label=r"$v_{avg}$ dla a$\downarrow$")
    ax.scatter(a_down, result["down_max"], s=8, color="blue", label=r"$v_{\max}$ dla a$\downarrow$")

    ax.scatter(a_up, result["up_avg"], s=8, color="red", label=r"$v_{avg}$ dla a$\uparrow$")
    ax.scatter(a_up, result["up_max"], s=8, color="orange", label=r"$v_{\max}$ dla a$\uparrow$")

    ax.axvline(x=result["a_2m"], color="black", linestyle=":", linewidth=2, label="a = 2m")
    ax.axvline(x=result["tp"], color="purple", linestyle=":", linewidth=2, label=rf"$tp \approx {result['tp']:.4f}$")

    ax.set_xlabel(r"$a$")
    ax.set_ylabel("Biomasa w stanie stacjonarnym")

    if title is None:
        title = rf"Diagram bifurkacyjny dla $d_1 = {result['params']['d1']:.2f}$"
    ax.set_title(title)

    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    if show:
        plt.show()

    return ax