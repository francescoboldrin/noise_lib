import random
import sys
from math import floor

import numpy
from scipy.stats import powerlaw

from hypergraphx import Hypergraph

sys.path.append('..')


def error_generator(size, coin_flip_prob, incr_prob):
    error = 0

    rand_factor = random.random() * 0.5

    max_increment = size - 1 + (size - 1) * rand_factor
    max_increment = int(max_increment)

    if random.random() < 0.99:
        max_increment = size - 1

    for i in range(0, max_increment):
        if random.random() < incr_prob:
            error += 1
        else:
            break

    if random.random() < coin_flip_prob:
        error = -error

    return error

# def error_distribution_gaussian(size_of_edge, mean_param=None):
#     # TODO: we have to define the distribution
#     # we have to set same variables to determine the distribution of the error (mean and std)
#     # cause the research topic is how the contagion changes if we have noise in the hypergraph, and how much noise we need to have a significant change
#     mean_param = 0
#     std_param = 3
#     return floor(random.gauss(mean_param, size_of_edge / std_param))
#
#
# def error_distribution_powerlaw(size_of_edge):
#     coin_flip = (random.random() < 0.5)
#     # print(coin_flip)
#
#     alpha = 0.5
#
#     pw_rand = numpy.random.power(alpha, 3)
#
#     error = 0
#
#     for p in pw_rand:
#         error += p
#
#     error = error / len(pw_rand)
#
#     if coin_flip:
#         error = -error
#     return int(error)
#
#
# def error_distribution_poisson(size_of_edge):
#     """
#     ERROR DISTRIBUTION POISSON
#
#     return a random number following a poisson distribution with parameter lambda = size_of_edge / fraction_factor
#
#
#     :param size_of_edge: the size of the edge
#     :return: the error
#     """
#
#     coin_flip = random.random() < coin_flip_prob
#     # print(coin_flip)
#     repeat_factor = 1
#
#     lam = size_of_edge / fraction_factor
#
#     res = numpy.random.poisson(lam, repeat_factor)
#
#     error = 0
#
#     for r in res:
#         error += r
#
#     error = error / len(res)
#
#     if coin_flip:
#         error = -error
#
#     return int(error)


def take_random_nodes_from_edge(edge, hypergraph, param):
    # TODO: rank the nodes

    # we have to take param nodes from the edge
    list_of_nodes = []

    for i in edge:
        list_of_nodes.append(i)

    # shuffle the list of nodes
    random.shuffle(list_of_nodes)

    # take the first param nodes
    list_of_nodes = list_of_nodes[:param]

    # shuffle noise
    # ------------------------------------
    # shuffle_prob = 0.
    # shuffle_param = 0.3
    #
    # if random.random() < shuffle_prob:
    #     random.shuffle(list_of_nodes)
    #
    #     for i in range(0, int(shuffle_param * len(list_of_nodes))):
    #         # chose another node in the hypergraph
    #         node = random.choice(hypergraph.get_nodes())
    #         if node not in list_of_nodes:
    #             # swap the node with a random node in the list
    #             list_of_nodes[random.randint(0, len(list_of_nodes) - 1)] = node
    # --------------------------------------------------

    return list_of_nodes


def take_random_nodes_from_hypegraph(edge, hypergraph, param):
    # TODO: rank the nodes

    # we have to take param nodes from the hypergraph who are not in the edge
    nodes = hypergraph.get_nodes()
    list_of_nodes = []

    for i in range(0, param):
        # TODO: set a probability distribution to chose the nodes
        index = random.choice([x for x in nodes if x not in edge and x not in list_of_nodes])
        list_of_nodes.append(index)

    return list_of_nodes


# def global_definition():
#     global std_param
#     global mean_param
#
#     std_param = 0.5
#     mean_param = 0
#
#     # file = open("./results/noise_param.txt", "w")
#     #
#     # file.write(
#     #     "we add noise to the hypergraph, by modifying the size of the edges in terms of a gaussian dist with these parameters:\n")
#     # file.write("std_param: " + str(std_param) + "\n")
#     # file.write("mean_param: " + str(mean_param) + "\n")


def check_if_edge_is_in_hypergraph(new_edge, edges):
    for edge in edges:
        if set(new_edge) == set(edge):
            return False
    return True


def noisy_hypergraph_generator(hypergraph, incr_prob_in, coin_flip_prob_in, perc_to_change):
    # make a documentation for this function, make bold title and add a description
    """
    NOISY HYPERGRAPH GENERATOR

    change the size of the edges in the hypergraph

    size new edge = size old edge + error_generator(size old edge)

    if the size of the new edge is < 2, we remove the edge from the hypergraph


    Parameters
    ----------
    hypergraph: Hypergraph
        The hypergraph to add noise to

    Returns
    -------
    Hypergraph:
        The hypergraph with noise
    """

    coin_flip_prob = coin_flip_prob_in
    incr_prob = incr_prob_in

    # we create a brand new hypergraph, with no edges and nodes from the original one
    h = Hypergraph()

    # we add the nodes from the original hypergraph
    h.add_nodes(hypergraph.get_nodes())

    edges = hypergraph.get_edges()

    # edges_changes = {
    #     edge: bool that says if the edge will be change or not
    # }
    edge_changes = {
    }

    # we chose the edges to change at random but the total number of edges to change is a percentage of the total number of edges, from parameter perc_to_change
    edges_to_change = random.sample(edges, int(len(edges) * perc_to_change))
    for edge in edges:
        if edge in edges_to_change:
            edge_changes[edge] = True
        else:
            # print(edge)
            edge_changes[edge] = False

    c = 0

    for edge in edges:

        # print(edge)
        new_edge = []
        size_of_edge = len(edge)
        new_size_of_edge = 0
        if edge_changes[edge]:
            error = error_generator(size_of_edge, coin_flip_prob, incr_prob)
            # print("error: " + str(error))

            new_size_of_edge = size_of_edge + error

            if new_size_of_edge < 2:
                new_size_of_edge = 0
            elif (new_size_of_edge < size_of_edge):
                # print("caso in cui la nuova dimensione dell'edge è minore della dimensione originale")
                # TODO: we think about shuffling some nodes in the edge and some nodes from the hypergraph
                for i in take_random_nodes_from_edge(edge, hypergraph, new_size_of_edge):
                    new_edge.append(i)
            else:
                for i in edge:
                    new_edge.append(i)
                for i in take_random_nodes_from_hypegraph(edge, hypergraph, abs(size_of_edge - new_size_of_edge)):
                    new_edge.append(i)
            # print(new_edge)

            n_edge = tuple(new_edge)
            # discard the edge if it is empty or if it is already in the hypergraph
        else:
            for i in edge:
                new_edge.append(i)
            n_edge = tuple(new_edge)
            # print("edge not changed: " + str(n_edge))

        if len(new_edge) > 1:
            try:
                h.add_edge(n_edge)
                c += 1
            except:
                pass

    # print("number of edges added: " + str(c))
    return h
