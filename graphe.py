from mdp import *
import numpy as np
import graphviz

class graphe(gramPrintListener) :

    def __init__(self, path) :
        super().__init__()
        lexer = gramLexer(FileStream(path))
        stream = CommonTokenStream(lexer)
        parser = gramParser(stream)
        tree = parser.program()
        walker = ParseTreeWalker()
        walker.walk(self, tree)
        self.dictStates = {}
        for i, state in enumerate(self.states):
            self.dictStates[state] = i
        self.dictActions = {}
        for i, action in enumerate(self.actions):
            self.dictActions[action] = i
        erreur = self._verifGraphe()
        assert (erreur == ""), erreur
        pass

    def __repr__(self) : 
        ret = ""
        ret += "States: %s" % str([str(x) for x in self.states]) + "\n"
        ret +="Actions: %s" % str([str(x) for x in self.actions]) + "\n"
        for x in self.transact : 
            ret +="Transition from " + x[0] + " with action "+ x[1] + " and targets " + str(x[2]) + " with weights " + str(x[3]) + "\n"
        for x in self.transnoact : 
            ret +="Transition from " + x[0] + " with no action and targets " + str(x[1]) + " with weights " + str(x[2]) + "\n"
        return ret

    def _grapheTransNotActToMatAdj(self) -> np.ndarray: 
        matAdjNoAct = np.zeros((len(self.states), len(self.states)))
        for transnoact in self.transnoact:
            for arrivee, poids in zip(transnoact[1], transnoact[2]):
                ligne = self.dictStates[transnoact[0]]
                colonne = self.dictStates[arrivee]
                matAdjNoAct[ligne, colonne] = poids
        return matAdjNoAct
    
    def _grapheTransActToMatAdj(self) -> np.ndarray:
        matAdjAct = np.zeros((len(self.actions), len(self.states), len(self.states)))
        for transact in self.transact:
            for arrivee, poids in zip(transact[2], transact[3]):
                coor_x = self.dictActions[transact[1]]
                coor_y = self.dictStates[transact[0]]
                coor_z = self.dictStates[arrivee]
                matAdjAct[coor_x, coor_y, coor_z] = poids
        return matAdjAct
    
    def grapheToMat(self) -> np.ndarray:
        """
        Fonction appelée qui permet de définir la matrice du graphe.
        Elle prend en compte la présence ou non de transition avec ou sans actions.
        return:
            np.ndarray de taille nb_actions, nb_etats, nb_etats
        """
        pattern = (self.transact, self.transnoact)
        match pattern:
            case ([], []):
                return np.zeros(0)
            case (_, []):
                return self._grapheTransActToMatAdj()
            case ([], _):
                return np.expand_dims(self._grapheTransNotActToMatAdj(), 1)
            case _:
                mat = self._grapheTransActToMatAdj()
                for i in range(mat.shape[0]):
                    mat[i] += self._grapheTransNotActToMatAdj()
                return mat

    def _verifGraphe(self) -> str: 
        """
        Fonction appelée lors de l'initialisation. Vérifie que le graphe est correct : 
            - Vérifie que les états et actions utilisés dans les transitions sont bien définis.
            - Vérifie que s'il existe une transition avec action de x vers y, alors il n'existe pas de transition sans action de x vers y.
            - Vérifie que les poids se somment à un 
        """
        erreurs = []
        check_states = [[trans[0]] + trans[2] for trans in self.transact] + [[trans[0]] + trans[1] for trans in self.transnoact]
        check_actions = [trans[1] for trans in self.transact]
        # Vérification de transact 
        for state in sum(check_states,[]) : 
            if not (state in self.states) : 
                erreurs.append(f"{state} état non défini")
        # Vérification de transnoact
        for action in check_actions :
            if not (action in self.actions) : 
                erreurs.append(f"{action} action non définie")
        # Vérification de l'unicité des transactions avec/sans action 
        for transact in self.transact : 
            for transnoact in self.transnoact : 
                if transact[0] == transnoact[0] : 
                    for state1 in transact[2] :
                        for state2 in transnoact[1] : 
                            if state1 == state2 :
                                erreurs.append(f"La transition de {transact[0]} vers {state1} existe avec ET sans action.")
        # Vérification des sommes des poids 
        for transact in self.transact : 
            if np.sum(transact[3]) != 10 :
                erreurs.append(f"Les poids des transitions à partir de {transact[0]} avec l'action {transact[1]} ne se somment pas à 1")
        for transnoact in self.transnoact : 
            if np.sum(transnoact[2]) != 10 :
                erreurs.append(f"Les poids des transitions sans action à partir de {transact[0]} ne se somment pas à 1")
        # Suppression des doublons
        erreur = []
        [erreur.append(x) for x in erreurs if x not in erreur]
        return "\n".join(erreur)
    
    def visualizeGraphe(self):
        viz = graphviz.Digraph("Graphe", comment="vive Markov")
        for state in self.states:
            viz.node(state)

        for transNoAct in self.transnoact:
            origin_state = transNoAct[0]
            for destination_state, destination_weight in zip(transNoAct[1], transNoAct[2]):
                viz.edge(origin_state, destination_state, label=str(destination_weight))
        for i, transAct in enumerate(self.transact):
            origin_state = transAct[0]
            tmp_action = transAct[1] + str(i)
            viz.node(tmp_action, shape="point")
            viz.edge(origin_state, tmp_action, label=tmp_action[:-1])
            for destination_state, destination_weight in zip(transAct[2], transAct[3]):
                viz.edge(tmp_action, destination_state, label=str(destination_weight))
        return viz

    def _visualizeGrapheState(self, current_state) : 
        """
        Modification de visualizeGraphe mettant en évidence l'état où se trouve le parcours du graphe.
        """
        viz = graphviz.Digraph("Graphe", comment="vive Markov")
        for state in self.states:
            if state == current_state :
                viz.node(state, color='red')

        for transNoAct in self.transnoact:
            origin_state = transNoAct[0]
            for destination_state, destination_weight in zip(transNoAct[1], transNoAct[2]):
                viz.edge(origin_state, destination_state, label=str(destination_weight))
        for i, transAct in enumerate(self.transact):
            origin_state = transAct[0]
            tmp_action = transAct[1] + str(i)
            viz.node(tmp_action, shape="point")
            viz.edge(origin_state, tmp_action, label=tmp_action[:-1])
            for destination_state, destination_weight in zip(transAct[2], transAct[3]):
                viz.edge(tmp_action, destination_state, label=str(destination_weight))
        return viz

    def parcours(self, N_pas = 50, regle = "", ret_chemin = False) -> list : 
        if regle == "alea" : 
            chemin = self._parcoursAlea(N_pas)
        else :
            chemin = self._parcoursUser(N_pas)
        if ret_chemin : 
            return chemin
    
    def _parcoursAlea(self, N_pas = 50) -> list : 
        """
        Parcours du graphe en choisissant aléatoirement les actions et les états. 
        """
        N_etat = len(self.states)
        etat = 0
        mat = self.grapheToMat()
        chemin = [self.states[0]]
        for i in range(N_pas) :
            possible = [x[etat] for x in mat  if np.any(x[etat])]
            action = np.random.randint(len(possible))
            proba = possible[action]/10
            etat = np.random.choice(np.arange(0, N_etat), p=proba)
            print(f"L'action {action} est choisie, l'état {self.states[etat]} est atteint avec une probabilité p = {proba[etat]}")
            chemin.append(self.states[etat])
        return chemin

    def _parcoursUser(self, N_pas = 50) -> list :
        """
        Parcours du graphe en demandant à l'utilisateur les actions, et en choisissant aléatoirement les états.
        """
        N_etat = len(self.states)
        etat = 0
        mat = self.grapheToMat()
        chemin = [self.states[0]]
        for i in range(N_pas) :
            print(f"L'état actuel est l'état {self.states[etat]}")
            actions_possibles = [self.actions[i] for i in range(len(self.actions)) if np.any(mat[i][etat])]
            print(f"Les actions possibles sont {actions_possibles}")
            action = input("Choisissez une action : ")
            while action not in actions_possibles :
                print(f"Action incorrecte. {action} n'est pas dans {actions_possibles}")
                action = input("Choisissez une action : ")
            proba = mat[self.dictActions[action]][etat]/10
            etat = np.random.choice(np.arange(0, N_etat), p=proba)
            print(f"L'action {action} est choisie, l'état {self.states[etat]} est atteint avec une probabilité p = {proba[etat]}")
            chemin.append(self.states[etat])
        return chemin
