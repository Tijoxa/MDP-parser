import numpy as np
import sys
sys.path.append(".")
import graphe


MDP_PATH = "exemples/ex.mdp"
"""Content of ex.mdp:
States S0, S1, S2;
Actions a,b,c,d;
S0 -> 2:S1 + 8:S2;
S1[b] -> 2: S1 + 8:S0;
S1[a] -> 1:S2+3:S0+6:S1;
S2[c] -> 5:S0 + 5:S1;
"""
graph = graphe.graphe(MDP_PATH)

MDP_RW_PATH = "exemples/exreward.mdp"
graph_rw = graphe.graphe(MDP_RW_PATH)


class TestGraphAttributes:
    def test_actions(self):
        assert isinstance(graph.actions, list)

    def test_states(self):
        assert isinstance(graph.states, list)

    def test_verif_transact(self):
        verif_transact = True
        for transact in graph.transact:
            if len(transact) != 4:
                verif_transact = False
        assert verif_transact

    def test_verif_transnoact(self):
        verif_transnoact = True
        for transnoact in graph.transnoact:
            if len(transnoact) != 3:
                verif_transnoact = False
        assert verif_transnoact

    def test_verif_dictStates(self):
        verif_dictStates = True
        for k, v in graph.dictStates.items():
            if not (isinstance(k, str) and isinstance(v, int)):
                verif_dictStates = False
        assert verif_dictStates

    def test_verif_dictActions(self):
        verif_dictActions = True
        for k, v in graph.dictActions.items():
            if not (isinstance(k, str) and isinstance(v, int)):
                verif_dictActions = False
        assert verif_dictActions

    def test_reward(self):
        assert np.all(graph_rw.reward == [1, 3, 0])


class TestGraphMethods:
    def test_grapheToMat(self):
        mat = graph.grapheToMat()

        expected_result = np.array([
            [[0., 2., 8.],
             [3., 6., 1.],
             [0., 0., 0.]],

            [[0., 2., 8.],
             [8., 2., 0.],
             [0., 0., 0.]],

            [[0., 2., 8.],
             [0., 0., 0.],
             [5., 5., 0.]],

            [[0., 2., 8.],
             [0., 0., 0.],
             [0., 0., 0.]]
        ])

        assert np.all(mat == expected_result)

    def test_verifGraphe(self):
        verification = graph._verifGraphe()
        assert verification == ""

    def test_parcours(self):
        parcours_list = graph.parcours(N_pas=10, regle="alea", ret_chemin=True)
        assert len(parcours_list) == 11
