from graphe import graphe
import numpy as np
import pandas as pd

# TODO: implémenter calcul probas pour mdp et mc (avec s0, s1 et s?), utiliser scipy.linprog pour mdp
# TODO: implémenter calcul récompenses pour mdp et mc

def _create_Matrix(g : graphe):
    pass # TODO

def _create_vector(g : graphe):
    pass # TODO

def pctl_finally(g : graphe, state):
    assert g.transact == [], "Il ne s'agit pas d'une MC"
    A = g._create_Matrix()
    b = g._create_vector()
    result = np.linalg.inv(np.eye(len(A)) - A)@b
    return result

def pctl_finally_max_bound(g : graphe, state, max_bound: int):
    assert g.transact == [], "Il ne s'agit pas d'une MC"
    A = g._create_Matrix()
    b = g._create_vector()
    x0 = np.zeros((len(A), 1))
    for _ in range(max_bound):
        x0 = A@x0 + b
    return x0

def pctl_mdp(g : graphe):
    assert g.transnoact == [], "Il ne s'agit pas d'une MDP"
    # TODO

def statistiques(g : graphe, N_pas=50, N_parcours=50):
    """
    Étude statistique de graphe par parcours aléatoires
    N_pas : Nombre de pas à effectuer dans le graphe
    N_parcours : Nombre de parcours 
    """

    header1 = g.states + ["Total"]
    header2 = ["->" + state for state in g.states] + ["Total"]
    ll = len(header1) + 1
    freq = [[0 for _ in range(ll)] for _ in range(ll)]
    freq = np.array(freq)

    for _ in range(N_parcours):
        chemin = g.parcours(regle="alea", N_pas=N_pas, ret_chemin=True)
        old_state = 0
        for state in chemin[1:]:
            freq[old_state + 1, state + 1] += 1
            old_state = state

    freq = freq / (N_pas * N_parcours)
    freq[-1] = np.sum(freq, axis=0)
    freq[:, -1] = np.sum(freq, axis=1)

    # Fréquece de visite d'un état à partir d'un autre état
    freq = pd.DataFrame(freq[1:, 1:], columns=header2, index=header1)
    freq.columns.name = 'Freq'

    print(freq)

def montecarlo_SMC(g : graphe, eps=0.01, delta=0.05, max_depth = 5):
    N = int((np.log(2) - np.log(delta))/(2*eps)**2)
    freq = np.zeros(len(g.states))
    for state in g.states:
        for _ in range(N):
            s = g._parcoursRapideAlea(N_pas=max_depth)
            if s == g.dictStates[state]:
                freq[s] += 1
    return freq/N 

def sprt_SMC(g : graphe, propriété,alpha=0.01, beta=0.01, theta=0.5, eps=0.01, max_depth=5):
    gamma1 = theta - eps
    gamma0 = theta + eps
    A = np.log(1-beta) - np.log(alpha)
    B = np.log(beta) - np.log(1-alpha)
    Rm = 0
    Rtrue = np.log(gamma1) - np.log(gamma0)
    Rfalse = np.log(1-gamma1) - np.log(1-gamma0)
    i = 0
    while Rm < A and Rm > B :
        i += 1
        s = g._parcoursRapideAlea(N_pas = max_depth)
        if propriété(g.states[s]) :
            Rm += Rtrue
        else :
            Rm += Rfalse 

    print(f"i = {i} itérations")
    print(f"A : {A}, B : {B} et Rm : {Rm}")
    if Rm >= A :
        return f"H1 : gamma < {theta} accepté"
    elif Rm <= B :
        return f"H0 : gamma >= {theta} accepté"
    
def iter_valeurs(g : graphe, gamma, eps = 0.1):
    pass

def iter_politique(g : graphe, gamma):
    pass
