from mdp import *
import numpy as np

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


    def grapheToMatInc(self) : 
        pass
