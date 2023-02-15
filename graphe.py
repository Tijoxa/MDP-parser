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


