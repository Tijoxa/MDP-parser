from graphe import graphe
import numpy as np
import pandas as pd
from typing import List
from scipy.optimize import linprog

# TODO : BONUS : Compatibilité fonctions pctl / MC accessibilité pour les récompenses
# TODO : implémenter algo de RL pour les MDP

def _accesibility(successeurs: list, target_states: list, N: int):
    accessible = []
    new_accessible = target_states.copy()
    while new_accessible != accessible:
        accessible = new_accessible.copy()
        for pred in range(N):
            if pred not in accessible and len([x for x in accessible if x in successeurs[pred]]) > 0:
                new_accessible.append(pred)
    S1 = accessible[:len(target_states)]
    S2 = accessible[len(target_states):] 
    return S1,S2

def _identify(g: graphe, target_states: list):
    """
    Identification des ensembles d'états pouvant atteindre un état donné en argument :
    - target_states : liste d'états à atteindre
    Sorties : 
    - S1 : états qui correspondent à "state"
    - S2 : états qui peuvent atteindre "state"
    """
    # Recherche des successeurs 
    N = len(g.states)
    target_states = [g.dictStates[state] for state in target_states]
    successeurs = [[j for j in range(N) if g.mat[0][i][j] > 0] for i in range(N)]  
    S1,S2 = _accesibility(successeurs, target_states, N)
    # Ajout dans S1 des prédecesseurs de S2 qui vont avec une probabilité 1 dans S1.
    to_switch = []
    change = True
    while change:
        change = False
        for i in S2:
            for j in S1:
                if g.mat[0][i][j] == 1:
                    to_switch.append(i)
                    change = True
        if change :
            S1 += to_switch
            S2 = [x for x in S2 if x not in to_switch]
            to_switch = []
    S1.sort()
    S2.sort()
    return S1,S2
                     

def _create_Matrix_mc(g: graphe, S2: list) -> list:
    A = [[g.mat[0][i][j] for j in S2] for i in S2]
    return np.array(A)

def _create_vector_mc(g: graphe, S1: list, S2: list):
    b = np.zeros(len(S2))
    for k in range(len(S2)):
        state = S2[k]
        b[k] += sum([g.mat[0][state][i] for i in S1])
    return b

def pctl_finally(g : graphe, states):
    assert g.transact == [], "Il ne s'agit pas d'une MC"
    S1,S2 = _identify(g, states)
    A = _create_Matrix_mc(g, S2)
    b = _create_vector_mc(g, S1,S2)
    result = np.linalg.inv(np.eye(len(A)) - A)@b
    states = [g.states[x] for x in S2]
    result = pd.DataFrame(result, columns = ["P"], index=states).transpose()
    return result

def pctl_finally_max_bound(g : graphe, states, max_bound: int):
    assert g.transact == [], "Il ne s'agit pas d'une MC"
    S1,S2 = _identify(g, states)
    A = _create_Matrix_mc(g, S2)
    b = _create_vector_mc(g, S1,S2)
    x0 = np.zeros(len(b))
    for _ in range(max_bound):
        x0 = A@x0 + b
    states = [g.states[x] for x in S2]
    x0 = pd.DataFrame(x0, columns = ["P"], index=states).transpose()
    return x0

def _identifyMDP(g: graphe, target_states:list):
    """
    Identification des ensembles d'états pouvant atteindre un état donné en argument :
    - target_states : liste d'états à atteindre
    Sorties : 
    - S1 : états qui correspondent à "state"
    - S2 : états qui peuvent atteindre "state"
    """
    # Recherche des successeurs 
    N = len(g.states)
    target_states = [g.dictStates[state] for state in target_states]
    successeurs_all_actions = []
    [successeurs_all_actions.append([[j for j in range(N) if g.mat[action][i][j] > 0] for action in range(len(g.actions))]) for i in range(N)]
    successeurs_all_actions = [sum(x,[]) for x in successeurs_all_actions]
    successeurs = []
    for succ in successeurs_all_actions:
        L = []
        [L.append(x) for x in succ if x not in L]
        successeurs.append(L)  
    return _accesibility(successeurs, target_states, N)

def _create_Matrix_mdp(g: graphe, S2: list):
    N_actions = len(g.actions)
    N_states = len(g.states)
    N_S2 = len(S2)
    A_projected = []
    for action in range(N_actions):
        scheduler = [action]*N_states
        projection = projection_mdp_to_mc(g.mat, scheduler)
        projection_A = np.array([[projection[i][j] for j in S2] for i in S2])
        projection_A = np.eye(N_S2,N_S2) - projection_A 
        A_projected.append(projection_A)
    A = np.zeros(shape=(N_S2*N_actions,N_S2))
    for i in range(N_S2):
        for j in range(N_actions):
            A[i*(N_actions) + j] = A_projected[j][i]
    return A

def _create_vector_mdp(g: graphe, S1: list, S2: list):
    N_actions = len(g.actions)
    N_S2 = len(S2)
    b_actions = np.zeros((N_S2, N_actions))
    for action in range(len(g.actions)):
        for k in range(len(S2)):
            state = S2[k]
            b_actions[k][action] += sum([g.mat[action][state][i] for i in S1])
    b = np.zeros(shape=(N_S2*N_actions))
    for state in range(N_S2):
        b[state*N_actions:state*N_actions+N_actions] = b_actions[state]
    return b

def pctl_mdp(g : graphe, states):
    assert g.transact != [], "Il ne s'agit pas d'une MDP"
    S1,S2 = _identifyMDP(g, states)
    A = _create_Matrix_mdp(g, S2)
    b = _create_vector_mdp(g, S1,S2)
    c = np.ones(len(S2)) # Fonction objectif [1,1,1..] car on veut minimiser c.x la somme des probabilités, cf diapo
    res = linprog(c, A_ub=-A, b_ub=-b, bounds=(0,1)) # On a A et b tels que A.x >= b, donc on passe -A et -b en argument
    states = [g.states[x] for x in S2]
    x = pd.DataFrame(res.x, columns = ["P"], index=states).transpose()
    return x


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
        print(f"H1 : gamma < {theta} accepté")
        return False
    elif Rm <= B :
        print(f"H0 : gamma >= {theta} accepté")
        return True

def mean_reward_mc(g: graphe, state: str):
    state = g.dictStates[state]
    rs = g.reward[state] + np.sum(g.mat[0][state]*g.reward)
    return rs

def mean_reward_mdp(g: graphe, state: str):
    state = g.dictStates[state]
    rs = g.reward[state] + np.max([np.sum(g.mat[action][state]*g.reward) for action in range(len(g.actions))])
    return rs

def bellman(g: graphe, gamma: float, V0: list, V1: list, Sigma: list):
    for i in range(len(g.states)): # s
        rw = g.reward[i] # r(s)
        best_action = 0
        best_adding = 0
        for j in range(len(g.actions)): # a
            if np.any(g.mat[j, i, :]):
                # k is s'
                adding = np.sum([g.mat[j, i, k] * V0[k] for k in range(len(g.states))])
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

def projection_mdp_to_mc(matrix: np.ndarray, scheduler):
    result = np.zeros(matrix.shape[1:3])
    for i, adv in enumerate(scheduler):
        result[i, :] = matrix[adv, i, :]

    return result

def bellman_2(g: graphe, V: np.ndarray):
    Sigma = [0 for _ in range(len(g.states))]
    for i in range(len(g.states)): # s
        best_action = 0
        best_adding = 0
        for j in range(len(g.actions)): # a
            if np.any(g.mat[j, i, :]):
                # k is s'
                adding = np.sum([g.mat[j, i, k] * V[k] for k in range(len(g.states))])
            if adding > best_adding:
                best_adding = adding
                best_action = j
        Sigma[i] = best_action
    return Sigma

def iter_politique(g: graphe, gamma: float):
    def auxiliaire(g: graphe, Sigma_0: list, gamma: float):
        rw = np.array(g.reward)
        projection_mat = projection_mdp_to_mc(g.mat, Sigma_0)
        A = np.eye(projection_mat.shape[0]) - gamma * projection_mat
        V = np.linalg.solve(A, rw)
        if not np.allclose(np.dot(A, V), rw):
            print("La résolution a échouée")
        Sigma_1 = bellman_2(g, V)
        return Sigma_1, V
    
    Sigma_0 = [0 for _ in range(len(g.states))]
    Sigma_1, V = auxiliaire(g, Sigma_0, gamma)
    while Sigma_0 != Sigma_1:
        Sigma_0 = Sigma_1.copy()
        Sigma_1, V = auxiliaire(g, Sigma_0, gamma)

    return V, Sigma_1

def montecarlo_rl(g: graphe, Sigma: List[int], alpha: float=0.8, gamma: float=0.9, k: int=10, N: int=50):
    mat_projected = projection_mdp_to_mc(g.mat, Sigma)
    V_glob = []
    for _ in range(k): # nombre de simulations
        V = [0 for _ in range(len(g.states))]
        visited = [False for _ in range(len(g.states))]
        s = np.random.randint(len(g.states))
        rw = [g.reward[s]]
        visited[s] = True
        V[s] = (1 - alpha) * V[s] + alpha * np.polynomial.polynomial.polyval(gamma, np.flip(np.array(rw)))
        for _ in range(N):
            s = np.random.choice(len(g.states), p=mat_projected[s, :])
            if visited[s] == False:
                rw.append(g.reward[s])
                visited[s] = True
                V[s] = (1 - alpha) * V[s] + alpha * np.polynomial.polynomial.polyval(gamma, np.flip(np.array(rw)))
        V_glob.append(V)
    V_mean = np.mean(V_glob, axis=0)
    return V_mean

def td_rl(g: graphe, Sigma: List[int], alpha: float=0.8, gamma: float=0.9, k: int=10, N: int=50):
    mat_projected = projection_mdp_to_mc(g.mat, Sigma)
    V_glob = []
    for _ in range(k):
        V = [0 for _ in range(len(g.states))]
        s = np.random.randint(len(g.states))
        next_s = np.random.choice(len(g.states), p=mat_projected[s, :])
        rw = g.reward[s]
        difference_temporelle = rw + gamma * V[next_s] - V[s]
        V[s] += alpha * difference_temporelle
        for _ in range(N):
            s = next_s
            next_s = np.random.choice(len(g.states), p=mat_projected[s, :])
            rw = g.reward[s]
            difference_temporelle = rw + gamma * V[next_s] - V[s]
            V[s] += alpha * difference_temporelle
        V_glob.append(V)
    V_mean = np.mean(V_glob, axis=0)
    return V_mean

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
            proba = mat[a1, s1] / np.sum(mat[a1, s1])
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

def _scheduler_evaluate(g: graphe, Sigma: List[float], N: int, phi, k: int=50):
    """
    Hyper-paramètres:
    - N : Nombre maximum d'échantillons
    - k : Profondeur de parcours
    - phi : Propriété à vérifier
    """
    argmax_Sigma = np.argmax(Sigma, axis=1)
    mat_projected = projection_mdp_to_mc(g.mat, argmax_Sigma)
    R_plus = np.zeros_like(Sigma)
    R_moins = np.zeros_like(Sigma)
    Q_hat = Sigma.copy()
    for _ in range(N):
        s = 0
        a = 0
        for _ in range(k):
            s = np.random.choice(len(g.states), p=mat_projected[s, :])
            a = argmax_Sigma[s]
            if phi(g.states[s]): # if s satisfies phi
                R_plus[s][a] += 1
            else:
                R_moins[s][a] += 1
    for s in range(len(g.states)):
        for a in range(len(g.actions)):
            if R_plus[s][a] != 0 or R_moins[s][a] != 0:
                Q_hat[s][a] = R_plus[s][a] / (R_plus[s][a] + R_moins[s][a])
    return Q_hat

def _scheduler_improvement(g :graphe, Sigma: List[float], h: float, eps: float, Q_hat: List[int]):
    """
    Hyper-paramètres:
    - h : paramètre d'histoire, 0 < h < 1
    - eps : Paramètre Glouton, 0 < eps < 1
    """
    Sigma2 = Sigma.copy()
    for s in range(len(g.states)):
        a_etoile = np.argmax([Q_hat[s]])
        p = np.zeros((1,len(g.actions)))[0]
        for a in range(len(g.actions)):
            if a == a_etoile:
                p[a] += (1-eps)
            p[a] += eps*(Q_hat[s][a])/(np.sum(Q_hat[s]))
        Sigma2[s] = h*Sigma[s] + (1-h)*p
    return Sigma2

def _scheduler_optimisation(g: graphe, Sigma: List[float], h: float, eps: float, N: int, L: int, phi):
    """
    Hyper-paramètres:
    - L : Nombre d'optimisations
    """
    for _ in range(L):
        Q_hat = _scheduler_evaluate(g, Sigma, N, phi)
        Sigma = _scheduler_improvement(g, Sigma, h, eps, Q_hat)
    return Sigma

def _scheduler_determinise(Sigma : List[float]):
    return [[1 if i == np.argmax(x) else 0 for i in range(len(x))] for x in Sigma]

def smc_mdp(g: graphe, h: float, eps: float, N: int, L: int, p: float, tau: float, phi, theta: float):
    """
    Paramètres: 
    - h : paramètre d'histoire, 0 < h < 1
    - eps : Paramètre Glouton, 0 < eps < 1
    - N : Nombre maximum d'échantillons
    - L : Nombre d'optimisations
    - p : Paramètre de convergence (0 < p < 1)
    - tau : Confiance 
    - phi : Hypothèse à vérifier
    - theta : Confiance en l'hypothèse
    """
    T_tau_p = round(np.log2(tau)/np.log2(1-p))
    for _ in range(T_tau_p):
        Sigma = np.full((len(g.states), len(g.actions)), 1/(len(g.actions)))
        Sigma = _scheduler_optimisation(g, Sigma, h, eps, N, L, phi)
        Sigma = _scheduler_determinise(Sigma)
        mat_projected = projection_mdp_to_mc(g.mat, np.argmax(Sigma, axis=1))
        old_mat = g.mat
        g.mat = np.array([mat_projected]*len(g.actions))
        if not sprt_SMC(g, phi, theta = theta):
            g.mat = old_mat
            return False
        g.mat = old_mat
    return True