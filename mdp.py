from antlr4 import *
from gramLexer import gramLexer
from gramListener import gramListener
from gramParser import gramParser


class gramPrintListener(gramListener):

    def __init__(self):
        self.states = []
        self.actions = []
        self.transact = []
        self.transnoact = []
        pass

    def enterStatenoreward(self, ctx):
        self.states = [str(x) for x in ctx.ID()]

    def enterStatereward(self, ctx):
        self.states = [str(x) for x in ctx.ID()]
        self.reward = [int(str(x)) for x in ctx.INT()]

    def enterDefactions(self, ctx):
        self.actions = [str(x) for x in ctx.ID()]

    def enterTransact(self, ctx):
        ids = [str(x) for x in ctx.ID()]
        dep = ids.pop(0)
        act = ids.pop(0)
        weights = [int(str(x)) for x in ctx.INT()]
        self.transact.append([dep, act, ids, weights])

    def enterTransnoact(self, ctx):
        ids = [str(x) for x in ctx.ID()]
        dep = ids.pop(0)
        weights = [int(str(x)) for x in ctx.INT()]
        self.transnoact.append([dep, ids, weights])


def main():
    lexer = gramLexer(StdinStream())
    stream = CommonTokenStream(lexer)
    parser = gramParser(stream)
    tree = parser.program()
    printer = gramPrintListener()
    walker = ParseTreeWalker()
    walker.walk(printer, tree)


if __name__ == '__main__':
    main()
