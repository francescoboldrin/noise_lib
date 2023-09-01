from hypergraphx import Hypergraph
import random
import numpy as np

from hypergraphx.generation.random import random_hypergraph
from hypergraphx.generation.scale_free import scale_free_hypergraph


def global_definition():
    global std_parameter
    global mean_parameter
    global n_edges_max_percentage
    global epsilon
    global std_mean

    std_mean = 3
    std_parameter = 0.005
    mean_parameter = 0.05
    epsilon = 2
    n_edges_max_percentage = 0.5

    file = open("./results/m_generator_graph_parameters.txt", "w")

    s = "the algorithm generates a hyper graph, using normal distribution in cardinality of edges sets for each edge size\nmax size of edges is number of nodes * n_edges_max perc\n"

    file.write(s)
    file.write("std_mean: " + str(std_mean) + "\n")
    file.write("std_parameter: " + str(std_parameter) + "\n")
    file.write("mean_parameter: " + str(mean_parameter) + "\n")
    # file.write("epsilon: " + str(epsilon) + "\n")
    file.write("n_edges_max_percentage: " + str(n_edges_max_percentage) + "\n")


def is_connected(h):
    # TODO: make a function that checks if the hypergraph is connected

    return True


def generate_scale_free_graph():
    num_nodes = 20
    edges_by_size = {2: 10, 3: 8, 4: 5, 5: 2}
    scale_by_size = {2: 0.5, 3: 0.5, 4: 0.5, 5: 0.5}
    correlated = True
    corr_target = 0.5
    num_shuffles = 0

    h = scale_free_hypergraph(num_nodes, edges_by_size, scale_by_size, correlated, corr_target, num_shuffles)

    return h



def generate_from_file(path):
    file_reader = open(path, "r")

    hg = Hypergraph()

    size_init = 2

    hg.add_nodes(list(range(size_init)))

    # read each line separately
    for line in file_reader:
        # print(line)
        # split the line in words
        words = line.split()
        # print(words)
        # convert the words in int
        nodes = []
        for word in words:
            nodes.append(int(word))
            if int(word) > size_init:

                hg.add_nodes(list(range(size_init, int(word))))
                size_init = int(word)

        hg.add_edge(tuple(nodes))


    return hg

def generate_random_graph(number_of_nodes, range_of_edge_size, number_of_edges):
    percentage = [0] * (range_of_edge_size - 2)

    sum = 0
    while sum < 1:
        sum = 0
        for i in range(0, range_of_edge_size - 2):
            sum += percentage[i]

        incr = random.random()/10
        percentage[random.randint(0, range_of_edge_size - 3)] += incr


    print(percentage)

    # transform the list of percentage in a dict

    edge_by_size = {}

    for i in range(0, range_of_edge_size - 2):
        edge_by_size[i + 2] = int(percentage[i] * number_of_edges)

    h = random_hypergraph(number_of_nodes, edge_by_size)

    return h

