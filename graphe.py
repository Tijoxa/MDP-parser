from mdp import *

class graphe(gramPrintListener) :

    def __init__(self, path) :
        super().__init__()
        lexer = gramLexer(FileStream(path))
        stream = CommonTokenStream(lexer)
        parser = gramParser(stream)
        tree = parser.program()
        walker = ParseTreeWalker()
        walker.walk(self, tree) 
        erreur = self.verifGraphe()
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

    def grapheToMat(self) : 
        pass

    def verifGraphe(self) -> str: 
        """
        Fonction appelée lors de l'initialisation. Vérifie que le graphe est correct : 
            - Parmi les transitions, vérifie que les états et actions utilisés sont bien définis.
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
        # Suppression des doublons
        erreur = []
        [erreur.append(x) for x in erreurs if x not in erreur]
        return "\n".join(erreur)