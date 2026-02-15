import numpy as np
import matplotlib.pyplot as plt
from pipeline.model_core import v_stac, u_stac, jacobian, is_stable

# --------------------------------------------------
# Punkt jednorodny
# --------------------------------------------------

def homogeneous_state(a: float, m: float):
    """
    Zwraca jednorodny punkt stacjonarny (u*, v*)

    Parametry
    a : float
        Parametr zasobu wody.
    m : float
        Parametr śmiertelności.

    Zwraca
        Jednorodny stan równowagi modelu reakcji (u*, v*).
    """
    vs = v_stac(a, m, mode  = 1, add_delta = True)
    us = u_stac(vs, m)
    return us, vs


# --------------------------------------------------
# Stabilność bez dyfuzji (ODE)
# --------------------------------------------------

def check_ode_stability(a: float, m: float):
    """
    Sprawdza stabilność układu reakcyjnego (bez dyfuzji).

    Parametry
    a : float
        Parametr zasobu wody.
    m : float
        Parametr śmiertelności.

    Zwraca
        stable : bool
            Czy punkt jednorodny jest stabilny.
        J : ndarray (2x2)
            Jacobian w punkcie (u*, v*).
    """
    us, vs = homogeneous_state(a, m)
    J = jacobian(us, vs, m)

    return is_stable(J), J


# --------------------------------------------------
# Relacja dyspersji
# --------------------------------------------------

def dispersion(J: np.ndarray, d1: float, d2: float,
            k_min: float = 0, k_max: float = 5, n_k: int = 1000):
    """
    Oblicza relację dyspersji λ_max(k).

    Algorytm:
        Dla każdego k liczone są wartości własne macierzy J - k^2 D
        i wybierana jest największa część rzeczywista.

    Parametry
    J : ndarray (2x2)
        Jacobian reakcji.
    d1, d2 : float
        Współczynniki dyfuzji.
    k_min, k_max : float
        Zakres analizowanych k.
    n_k : int
        Liczba równomiernie rozłożonych punktów k.

    Zwraca
        k_vals : ndarray
            Wartości k.
        lambda_max : ndarray
            Największa część rzeczywista λ(k).
    """
    k_vals = np.linspace(k_min, k_max, n_k)
    lambda_max = np.zeros_like(k_vals)

    # macierz dyfuzji
    D = np.diag([d1, d2])

    for i, k in enumerate(k_vals):
        M = J - (k**2) * D
        # wartosci własne
        eig_vals = np.linalg.eigvals(M)
        # maksymalna część rzeczywista
        lambda_max[i] = np.max(np.real(eig_vals))

    return k_vals, lambda_max


# --------------------------------------------------
# Pasmo Turinga
# --------------------------------------------------

def turing_band(k_vals: np.ndarray, lambda_max: np.ndarray):
    """
    Wyznacza punkty niestabilne k (pasmo Turinga).

    Parametry
    k_vals : ndarray
    lambda_max : ndarray
        Największa część rzeczywista λ(k).

    Zwraca
        Jeśli istnieje niestabilność:
            {
                "k_min" : początek pasma,
                "k_max" : koniec pasma,
                "k_dom" : globalne maksimum relacji dyspersji
            }
        W przeciwnym razie None.
    """
    unstable = k_vals[lambda_max > 0]

    if unstable.size == 0:
        return None

    return {
        "k_min": unstable.min(),
        "k_max": unstable.max(),
        "k_dom": k_vals[np.argmax(lambda_max)]
    }


# --------------------------------------------------
# Analiza Turinga
# --------------------------------------------------

def turing_analysis(a: float, m: float, d1: float, d2: float,
                    k_min: float = 0, k_max: float = 5, n_k: int = 1000):
    """
     Analiza niestabilności Turinga.

    Algorytm:
        1. Wyznaczenie punktu jednorodnego.
        2. Sprawdzenie stabilności ODE.
        3. Obliczenie relacji dyspersji λ(k) (stabilny -> niestabilny po dodaniu dyfuzji).
        4. Wyznaczenie pasma niestabilnych k.

    Parametry
    a : float
        Parametr zasobu wody.
    m : float
        Parametr śmiertelności.
    d1, d2 : float
        Współczynniki dyfuzji.
    k_min, k_max : float
        Zakres analizowanych k.
    n_k : int
        Liczba punktów k.

    Zwraca
        - J : Jacobian
        - k : wartości k
        - lambda : λ_max(k)
        - band : zakres pasma niestabilności (lub None)
    """

    # stabilność ODE
    stable, J = check_ode_stability(a, m)

    if not stable:
        raise ValueError("Układ reakcyjny niestabilny — brak Turinga")

    # relacja dyspersji
    k_vals, lambda_max = dispersion(J, d1, d2, k_min, k_max, n_k)

    # pasmo Turinga
    band = turing_band(k_vals, lambda_max)

    return {
        "J": J,
        "k": k_vals,
        "lambda": lambda_max,
        "band": band
    }


# --------------------------------------------------
# Wykres
# --------------------------------------------------

def plot_dispersion(k_vals: np.ndarray, lambda_max: np.ndarray, band: dict):
    """
    Rysuje relację dyspersji λ(k) z zaznaczonym pasmem Turinga.

    Parametry
    k_vals : ndarray
    lambda_max : ndarray
        Największa część rzeczywista λ(k).
    """
    plt.figure(figsize=(8,5))
    # krzywa dyspersji
    plt.plot(k_vals, lambda_max, color = "navy")
    plt.axhline(0, color = "black")

    # Zaznaczanie punktów kluczowych dla pasma Turinga w postaci:
    # punkty szczególne k na osi x
    # wysokość wykresu dokładnie w n_k miejscu na osi y (interpolacja liniowa)
    k_min = band["k_min"]
    l_min = np.interp(k_min, k_vals, lambda_max)

    k_max = band["k_max"]
    l_max = np.interp(k_max, k_vals, lambda_max)

    k_dom = band["k_dom"]
    l_dom = np.interp(k_dom, k_vals, lambda_max)

    plt.axvline(k_min, linestyle = ":", color = "gray", label = "k_min")
    plt.axvline(k_max, linestyle = ":", color = "gray", label = "k_max")
    plt.axvline(k_dom, linestyle = "--", color = "gray", label = "k_dom")

    plt.scatter([k_min, k_max, k_dom], [l_min, l_max, l_dom], zorder=5)

    plt.xlabel("k")
    plt.ylabel("max Re(λ(k))")
    plt.title("Relacja dyspersji – analiza Turinga")
    plt.legend()
    plt.tight_layout()
    plt.show()

