import random
from math import exp, floor, log

from hypergraphx import Hypergraph
import numpy as np


def find_intersection(x, func):
    num = floor((x / func[0]) ** (1 / func[1]))

    return num


def non_linear_distribution(infected_nodes, param, len_edge):
    # TODO: define the non-linear distribution, in order to give some sort of non linearity to the contagion process
    # B(n, i) = (gamma * i)**param
    # if we use non linear probability
    # if i node in the same group try to infect the node j, the probability of success (gamma * i^param)/n
    # where n is the number of nodes in the edge, 0 < param < 1, gamma > 0
    gamma = 0.2

    # using function proposed, little modified, by the paper https://arxiv.org/pdf/2105.07092.pdf
    return (gamma * infected_nodes) ** param
    # return 0.005


def regular_probability():
    # if we use regular probability
    # if i node try to infect the node j, the probability of success is 1 - (1 - gamma) ^ i
    # where 0 < gamma < 1, i > 0 is the number of infected nodes trying to infect the node j

    gamma = 0.01

    return gamma


def m_contagion(hypergraph, I_0, T):
    # make a documentation for the function
    """
    non_linear_Contagion

    Simulates the contagion process on a simplicial hypergraph, in a non-linear function of probability.
    The process is run for T time steps.
    The initial condition is given by I_0, which is a vector of length equal to the number of nodes in the hypergraph.
    The infection rate is beta, the three-body infection rate is beta_D, and the recovery rate is mu.
    The output is a vector of length T, where the i-th entry is the fraction of infected nodes at time i.

    ....

    Parameters
    ----------
    hypergraph : hypergraphx.Hypergraph
        The hypergraph on which the contagion process is run.

    I_0 : numpy.ndarray
        The initial condition of the contagion process.

    T : int
        The number of time steps.

    Returns
    -------
    numpy.ndarray
        The fraction of infected nodes at each time step.

    """

    file = open("./results/contagion_param.txt", "w")

    file.write(
        "we calculate probability of infection (gamma * infected_nodes)**param with the following parameters (gamma, param): \n")

    file.write("param: " + str(1) + "\n")

    file.write("gamma: " + str(0.05) + "\n")

    numberInf = np.linspace(0, 0, T)
    Infected = np.sum(I_0)
    new_Infected = Infected
    numberInf[0] = Infected
    N = len(I_0)

    nodes = hypergraph.get_nodes()
    mapping = hypergraph.get_mapping()
    I_old = np.copy(I_0)
    I_prec = I_old
    t = 1
    print("start campaign")

    while t < T:

        new_Infected -= new_Infected

        # print (new_Infected)
        # create a new vector I_new that is same length as I_old but only contains zeros
        I_new = np.zeros(len(I_old))
        count = 0
        mean = 0
        for edge in hypergraph.get_edges():

            # find the percentage of the infected nodes in the edge
            infected_nodes = 0
            for node in edge:
                if I_old[node] == 1:
                    infected_nodes += 1
            infected_nodes_perc = infected_nodes / len(edge)

            # run the infection process through the edge
            for node in edge:
                if I_old[node] == 0 and I_new[node] == 0:
                    # we can chose between a regular probability or a non-linear probability
                    prob = regular_probability()
                    if random.random() * 10000 < prob * 10000:
                        I_new[node] = 1

            if t == 1:
                s = ""
                for node in edge:
                    if I_old[node] == 0 and I_new[node] == 0:
                        s += "0"
                    else:
                        s += "1"
                if s.__contains__("1"):
                    count += 1
                    mean += s.__len__()
        if t == 1:
            print(count)
            print(mean / count)

        I_prec = I_new
        # update the infected nodes
        for i in range(0, len(I_old)):
            if I_new[i] == 1:
                I_old[i] = 1

        new_Infected += np.sum(I_new)
        Infected += new_Infected
        numberInf[t] = Infected
        # print("Time step: " + str(t) + " - Infected nodes: " + str(Infected))
        t += 1
        if new_Infected == 0:
            pass
            # print("No more infected nodes")
            # print("at time step: " + str(t) + " - Infected nodes: " + str(Infected))

    return numberInf[:t - 1] / N


def threshold_calc(length_edge, theta_0):

    return length_edge * theta_0


def function_len(param):

    return np.log2(param)


def SIS_contagion(hypergraph, I_condition, T, lam_in, delta_in, theta_in):
    # make a documentation
    """
    SIS contagion
    model of contagion proposed by
    https://arxiv.org/pdf/2103.03709.pdf

    -----
    parameters
        hypergraph:
            the hypergraph on which the contagion process is run
        I_condition:
            the initial condition of the contagion process
        T:
            the number of time steps

    ------
    returns
        the fraction of infected nodes at each time step

    """

    t = 0

    N = len(I_condition)

    I_prec = I_condition

    sum = 0
    for i in hypergraph.get_nodes():
        sum += I_condition[i]
    total_infected = [sum]

    lam_0 = lam_in
    delta_0 = delta_in
    theta_0 = theta_in

    # define a vector delta_exp that contains the delta exponent for each node
    delta_exp = np.zeros(N)
    edges = hypergraph.get_edges()



    while t < T:

        # shuffle the edges
        random.shuffle(edges)
        for edge in edges:
            # find the percentage of the infected nodes in the edge
            if len(edge) == 2:

                node_1 = edge[0]
                node_2 = edge[1]

                if (I_prec[node_1] + I_prec[node_2]) >= 1:
                    # poisson process with parameter lambda = lambda_0 * function(size(edge))
                    # function is log base 2 of the size of the edge
                    lam = lam_0 * function_len(len(edge))
                    rep = 1

                    poisson_rand = np.random.poisson(lam, rep)

                    p_rand = np.sum(poisson_rand) / rep

                    # TAG 3
                    if random.random() < p_rand:
                        I_prec[node_1] = 1
                        I_prec[node_2] = 1

            else:
                infected_nodes = 0
                for node in edge:
                    if I_prec[node] == 1:
                        infected_nodes += 1

                if infected_nodes > threshold_calc(len(edge), theta_0):
                    # print("contagion process in group:", edge)
                    for node in edge:
                        if I_prec[node] == 0:
                            # poisson process with parameter lambda = lambda_0 * function(size(edge))
                            # function is log base 2 of the size of the edge
                            lam = lam_0 * function_len(len(edge))
                            rep = 1

                            poisson_rand = np.random.poisson(lam, rep)

                            p_rand = np.sum(poisson_rand) / rep

                            # TAG 2
                            if random.random() < p_rand:
                                I_prec[node] = 1

        for node in hypergraph.get_nodes():
            if I_prec[node] == 1:
                rep = 1

                # d_exp = delta_exp[node]
                poisson_rand = np.random.poisson(delta_0, rep)
                p_rand = np.sum(poisson_rand) / rep

                # TAG 1
                if random.random() < p_rand:
                    I_prec[node] = 0
                    pass
                pass

        sum_infected = 0
        for node in hypergraph.get_nodes():
            sum_infected += I_prec[node]
        total_infected.append(sum_infected)
        t += 1

    total_infected = np.array(total_infected)
    total_infected = total_infected[:t]

    return total_infected / N
