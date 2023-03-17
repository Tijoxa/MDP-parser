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

def bellman(g: graphe, gamma: float, V0: list, V1: list, Sigma: list):
    for i in range(len(g.states)): # s
        rw = g.reward[i] # r(s)
        best_action = 0
        best_adding = 0
        for j in range(len(g.actions)): # a
            if np.any(g.mat[j, i, :]):
                # k is s'
            if adding > best_adding:
                best_adding = adding
                best_action = j
        V1[i] = rw + gamma * best_adding
        Sigma[i] = best_action

def iter_valeurs(g: graphe, gamma: float, eps: float = 0.1):
    V0 = [0 for _ in range(len(g.states))]
    V1 = [0 for _ in range(len(g.states))]
    Sigma = [0 for _ in range(len(g.states))]
    bellman(g, gamma, V0, V1, Sigma)
    while np.linalg.norm(np.array(V1) - np.array(V0)) > eps:
        V0 = V1.copy()
        bellman(g, gamma, V0, V1, Sigma)
    
    return V1, Sigma


def montecarlo_rl(g : graphe):
    pass

def td_rl(g : graphe):
    pass

def sarsa_rl(g: graphe, T_tot=1000, gamma=0.5):
    mat = g.grapheToMat()
    N_etats = len(g.states)
    N_actions = len(g.actions)
    q0 = np.zeros((N_etats,N_actions))
    visited = np.zeros((N_etats,N_actions))
    s = 0
    # Sélection de l'action a et de s1
    actions_possibles = g.actions_possibles[g.states[s]]
    if len(actions_possibles) == 0:
        s1 = 0
        a = np.random.choice(np.arange(0, N_actions))
        r = g.reward[0]
    else : 
        a = actions_possibles[np.random.randint(len(actions_possibles))]
        proba = mat[a, s] / np.sum(mat[a, s])
        s1 = np.random.choice(np.arange(0, N_etats), p=proba)
        r = g.reward[s]

    for _ in range(T_tot):
        # Sélection de l'action a1 et de s2
        actions_possibles = g.actions_possibles[g.states[s1]]
        if len(actions_possibles) == 0:
            s2 = 0
            a1 = np.random.choice(np.arange(0, N_actions))
            r = g.reward[0]
        else : 
            a1 = actions_possibles[np.random.randint(len(actions_possibles))]
            proba = mat[a, s1] / np.sum(mat[a, s1])
            s2 = np.random.choice(np.arange(0, N_etats), p=proba)
            r = g.reward[s1]

        q1 = q0
        delta = r + gamma*q0[s1][a1] - q0[s][a]

        visited[s][a] += 1
        q1[s][a] = q0[s][a] + (1/visited[s][a])*delta

        q0=q1
        s=s1
        s1=s2
        a=a1

    print(q1)
    policy = np.argmax(q1, axis=1)
    return policy

def qlearning_rl(g: graphe, T_tot=1000, gamma=0.5):
    mat = g.grapheToMat()
    N_etats = len(g.states)
    N_actions = len(g.actions)
    q0 = np.zeros((N_etats,N_actions))
    visited = np.zeros((N_etats,N_actions))
    s = 0
    a = 0
    for _ in range(T_tot):
        actions_possibles = g.actions_possibles[g.states[s]]
        if len(actions_possibles) == 0:
            s1 = 0
            a = np.random.choice(np.arange(0, N_actions))
            r = g.reward[0]
        else : 
            a = actions_possibles[np.random.randint(len(actions_possibles))]
            proba = mat[a, s] / np.sum(mat[a, s])
            s1 = np.random.choice(np.arange(0, N_etats), p=proba)
            r = g.reward[s]
            
        q1 = q0
        delta = r + gamma*max(q0[s1]) - q0[s][a]

        visited[s][a] += 1
        q1[s][a] = q0[s][a] + (1/visited[s][a])*delta

        q0=q1
        s=s1

    print(q1)
    policy = np.argmax(q1, axis=1)
    return policy