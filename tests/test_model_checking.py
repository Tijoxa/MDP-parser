import sys
sys.path.append(".")
import graphe
import model_checking


MDP_PATH = "exemples/manchot2bras.mdp"
"""Content of ex.mdp:
States S0:0,S1:100,S2:5,S3:500,S4:3;
Actions a,b;
S0[a]->1:S1 + 1:S2;
S0[b]->1:S3 + 9:S4;
S1 -> 1:S0;
S2 -> 1:S0;
S3 -> 1:S0;
S4 -> 1:S0;
"""
graph = graphe.graphe(MDP_PATH)

MDP_RW_PATH = "exemples/exreward.mdp"
graph_rw = graphe.graphe(MDP_RW_PATH)


def test_pctl_finally():
    assert True


def test_pctl_finally_max_bound():
    assert True


def test_pctl_mdp():
    assert True


def test_statistiques():
    assert True


def test_montecarlo_SMC():
    assert True


def test_sprt_SMC():
    assert True


def test_mean_reward_mc():
    assert True


def test_mean_reward_mdp():
    assert True


def test_iter_valeurs():
    assert True


def test_iter_politique():
    assert True


def test_montecarlo_rl():
    assert True


def test_td_rl():
    assert True


def test_sarsa_rl():
    assert True


def test_qlearning_rl():
    assert True


def test_smc_mdp():
    assert True
