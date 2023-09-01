import random
import sys
import matplotlib.pyplot as plt
from matplotlib import ticker

from hypergraphx.generation.scale_free import scale_free_hypergraph

sys.path.append('..')

import hypergraphx
from hypergraphx.generation.random import *
from m_noise import noisy_hypergraph_generator
from hypergraphx.dynamics.contagion import *
from hypergraphx.measures.eigen_centralities import *
from m_contagion import *
from m_generator_graph import *
from hypergraphx.readwrite.loaders import load_high_school
from hypergraphx.measures.s_centralities import *
from hypergraphx.measures.degree import *
from collections import OrderedDict
from itertools import islice
import os

# def create_temporal_hypergraph():
#     h = random_uniform_hypergraph(num_nodes, 15, n_edges)
#     # print(h)
#
#     return h
#
#
# def make_contagion_campaign(hypergraph, v):
#     # make a contagion campaign
#     res = simplicial_contagion(hypergraph, v, time_steps, infection_rate, infection_rate_2, recovery_rate)
#
#     return res


def print_hgraph(hypergraph, file_name):
    # print the length of the hypergraph, (number of nodes and edges)
    file = open(file_name, "w")
    # print in the file the number of nodes and edges and then the nodes and edges
    # print("number of nodes: ", hypergraph.number_of_nodes(), file=file)
    # print("number of edges: ", hypergraph.number_of_edges(), file=file)
    print("nodes: ", hypergraph.get_nodes(), file=file)
    for i in hypergraph.get_edges():
        print(i, file=file)
    # for i in hypergraph.get_edges():
    #     print(i, file=file_name)
    list_node = list(hypergraph.get_nodes())


    # print(sorted(list_node))


def define_global_variables():
    global start_seed_percentage
    global time_steps
    global coin_flip_probability
    global delta_SIS_contagion
    global lambda_SIS_contagion
    global theta_SIS_contagion
    global incr_prob_start
    global incr_prob_end
    global incr_prob_step
    global greedy_algorithm_choice
    global perc_to_change_start
    global perc_to_change_end
    global perc_to_change_step

    try :
        file_in = open("./input/parameters", "r")

        # read the parameters from the file
        for line in file_in.buffer:
            line = line.decode("utf-8")
            line = line.strip()
            line = line.split(" ")

            if line[0] == "start_seed_percentage:":
                start_seed_percentage = float(line[1])
            elif line[0] == "time_steps:":
                time_steps = int(line[1])
            elif line[0] == "coin_flip_probability:":
                coin_flip_probability = float(line[1])
            elif line[0] == "delta_SIS_contagion:":
                delta_SIS_contagion = float(line[1])
            elif line[0] == "lambda_SIS_contagion:":
                lambda_SIS_contagion = float(line[1])
            elif line[0] == "theta_SIS_contagion:":
                theta_SIS_contagion = float(line[1])
            elif line[0] == "incr_prob_start:":
                incr_prob_start = float(line[1])
            elif line[0] == "incr_prob_end:":
                incr_prob_end = float(line[1])
            elif line[0] == "incr_prob_step:":
                incr_prob_step = float(line[1])
            elif line[0] == "greedy_algorithm_choice:":
                greedy_algorithm_choice = int(line[1])
            elif line[0] == "perc_to_change_start:":
                perc_to_change_start = float(line[1])
            elif line[0] == "perc_to_change_end:":
                perc_to_change_end = float(line[1])
            elif line[0] == "perc_to_change_step:":
                perc_to_change_step = float(line[1])

        file_in.close()
    except FileNotFoundError:
        print("File parameters not found")
        exit(1)


# def init_condition_generator():
#     # create a vector v of length num_nodes with 0 and 1 with a percentage of 1 equal to start_seed_percentage
#     v = []
#     for i in range(0, num_nodes):
#         if random.random() < start_seed_percentage:
#             v.append(1)
#         else:
#             v.append(0)
#     return v


def s_init_condition_generator(h_graph):
    # create a vector v of length num_nodes with 0 and 1 with a percentage of 1 equal to start_seed_percentage
    # find the most n% important nodes and set them to 1
    v = {}

    for node in h_graph.get_nodes():
        v[int(node)] = 0

    dict = degree_sequence(h_graph)

    sorted_dict = sorted(dict.items(), key=lambda x: x[1], reverse=True)

    s_perc = start_seed_percentage

    len_seed = int(len(h_graph.get_nodes()) * s_perc)

    if len_seed < 3:
        len_seed = 3

    for i in range(0, len_seed):
        v[sorted_dict[i][0]] = 1

    return v


# def calculating_func(h, i_0):
#     sourceFile = open("./results/res_1.txt", "w")
#     count = 0
#     mean = 0
#     for edge in h.get_edges():
#         # count += 1 if there is a seed in the edge
#         flag = False
#         for node in edge:
#             if i_0[node] == 1:
#                 count += 1
#                 flag = True
#                 break
#         if flag:
#             mean += len(edge)
#
#     s = "number of edge in the original graph with at least a seed: " + str(
#         count) + "\naverage size of edges with at least a seed:  " + str(mean / count) + "\n"
#
#     print(s, file=sourceFile)
#
#     overall_mean = 0
#     overall_count = 0
#     max = 100
#     for i in range(0, max):
#         noise = add_noise_to_hypergraph(h, fraction_factor, coin_flip_probability)
#
#         count = 0
#         mean = 0
#         for edge in noise.get_edges():
#             # count += 1 if there is a seed in the edge
#             flag = False
#             for node in edge:
#                 if i_0[node] == 1:
#                     count += 1
#                     flag = True
#                     break
#             if flag:
#                 mean += len(edge)
#         overall_mean += mean / count
#         overall_count += count
#
#     s = "number of edge in the noise graph (avg after 100 simulations) with at least a seed: " + str(
#         overall_count / max) + "\naverage size of edges with at least a seed:  " + str(overall_mean / max) + "\n"
#     print(s, file=sourceFile)

def plot_results(r1, r2,r3, r4, fraction, coin_flip_prob):
    # plot the results with matplotlib
    # results is an array of floats
    x = np.arange(len(r1+r2/2))

    plt.plot(x, r1, label='original graph with original seed', color='blue')
    plt.plot(x, r2, label='noise graph with original seed', color='red')
    plt.plot(x, r3, label='original graph with noise seed', color='green')
    plt.plot(x, r4, label='noise graph with noise seed', color='yellow')

    plt.xlabel('time')
    plt.ylabel('number of infected nodes in percentage')

    plt.title('SIS contagion model')
    plt.legend()

    try:
        # make a description below the plot
        description = "simulation with high school data using node degree rank"
        description += "\n" + "noise following poisson distribution with parameter lambda for each edge = (size of edge/fraction_factor) with fraction factor = " + fraction

        x_position = 0.5  # X-coordinate of the text
        y_position = -0.1  # Y-coordinate of the text (negative value places it below the graph)
        plt.text(x_position, y_position, description, ha='center', va='center', transform=plt.gca().transAxes)
    except:
        pass

    # add a grid and more details for the axis
    plt.grid(True)
    # plt.xticks(np.arange(len(x)), x)
    plt.yticks(np.arange(0, 1, 0.05))

    plt.show()
    pass

def find_intersection(edge, I0):
    # return true if there is at least one node in the edge that is in I0
    for node in edge:
        if I0[node] == 1:
            return True
    return False


def edge_init_condition_generator(h1):
    # return a dictionary with the initial condition for the CEC model
    # the dictionary is a map from node to state
    # the state is 1 if the node is in the initial condition, 0 otherwise
    print("edge init condition generator")
    rank_edges = s_betweenness(h1)
    # print(rank_edges)
    # print(rank)
    I_0 = {}
    for node in h1.get_nodes():
        I_0[node] = 0

    len_seed = int(len(h1.get_nodes()) * start_seed_percentage)

    if len_seed < 2:
        len_seed = 2

    degrees_rank = degree_sequence(h1)

    first_edges = list(islice(rank_edges.keys(), len_seed))

    print("first keys: ", first_edges)

    for edge in first_edges:
        # promote to 1 the node with max degree in the edge
        max_degree = 0
        max_node = 0
        for node in edge:
            if degrees_rank[node] > max_degree and I_0[node] == 0:
                max_degree = degrees_rank[node]
                max_node = node

        I_0[max_node] = 1


    return I_0

def plot_outbreaks(result, outbreak_original, perc_to_change_tmp):
    """
    plot_outbreaks
    plot the results of the simulation in a pyplot graph

    :param result: dictionary with the results of the simulation
    :param outbreak_original: number of infected nodes in the original graph
    :param perc_to_change_tmp: percentage of nodes affected by noise

    :return: nothing
    """
    file = open("./results/IM_highschool/counter_results.txt", "r")
    counter = file.read()
    if counter == "":
        counter = 0
    counter = int(counter)
    file.close()
    try:
        os.mkdir("./results/IM_highschool/results_" + str(round(perc_to_change_tmp,3)))
        pass
    except Exception as e:
        print(e)
        pass
    try:
        # plot the results with matplotlib
        # results is a dictionary: key: fraction factor, value: number of infected nodes at the end of the simulation
        x = list(result.keys())
        y = list(result.values())

        plt.clf()
        plt.plot(x, y, label='outbreaks graph, with percentage of nodes affected by noise: ', color='blue', linewidth=1.7)
        plt.title('outbreaks graph, with percentage of nodes affected by noise: ' + str(round(perc_to_change_tmp,3)))

        plt.xlabel('probability of incrementing noise')
        plt.ylabel('percentage of infected nodes at the end of simulation')
        plt.axhline(y=outbreak_original, color='green', linestyle='-', linewidth=1.5, label='percentage of infection in original graph')

        a = plt.gca()
        a.xaxis.set_major_locator(ticker.MultipleLocator(max(incr_prob_step, 0.1)))
        a.xaxis.set_minor_locator(ticker.MultipleLocator(0.01))
        a.yaxis.set_minor_locator(ticker.MultipleLocator(0.01))

        try:
            plt.grid(True, which='both', axis='both', linestyle='-', linewidth=0.5)
        except Exception as e:
            print(e)
            plt.grid(True)

        plt.savefig("./results/IM_highschool/results_" + str(round(perc_to_change_tmp, 3)) + "/outbreaks.png")
    except Exception as e:
        print(e)
        pass

    try:
        # print the results in outbreaks.txt
        f = open("./results/IM_highschool/results_" + str(round(perc_to_change_tmp, 3)) + "/outbreaks.txt", "w")
        f.write("parameters: \n")
        f.write("\ttime steps: " + str(time_steps) + "\n")
        f.write("\tlambda: " + str(lambda_SIS_contagion) + "\n")
        f.write("\tdelta: " + str(delta_SIS_contagion) + "\n")
        f.write("\ttheta: " + str(theta_SIS_contagion) + "\n")
        f.write("\tstart seed percentage: " + str(start_seed_percentage) + "\n")
        f.write("\tcoin flip probability: " + str(coin_flip_probability) + "\n")
        f.write("\tpercentage of nodes affected by noise: " + str(perc_to_change_tmp) + "\n")

        f.write("\n\npercentage of infected nodes at the end of simulation in the original graph: " + str(outbreak_original) + "\n")
        f.write("probability of increment noise, percentage of infected nodes at the end of simulation\n")
        for i in range(0, len(x)):
            f.write("\tprobability: " + str(round(x[i], 3)) + ", average outbreak:" + str(y[i]) + "\n")
        f.close()
        pass
    except Exception as e:
        print("error in writing the results in the file")
        print(e)
        pass




def simulation(hypergraph, I_condition):
    """
    simulation

    :param hypergraph: the hypergraph on which the simulation is performed
    :param I_condition: the initial condition of the simulation

    perform the simulation of contagion on the hypergraph with the initial condition I_condition
    the simulation is performed with parameters defined in the global variables
    then increase the noise on the hypergraph and perform the simulation again
    at the end plot the results

    """
    I_tmp = I_condition.copy()

    res_0 = SIS_contagion(hypergraph, I_tmp, time_steps,  lambda_SIS_contagion, delta_SIS_contagion, theta_SIS_contagion)

    outbreak_original = (res_0[len(res_0)-1] + res_0[len(res_0)-2] + res_0[len(res_0)-3])/3
    print("outbreak original: ", outbreak_original)

    for i in range(0, 19):
        I_tmp = I_condition.copy()
        res_tmp = SIS_contagion(hypergraph, I_tmp, time_steps, lambda_SIS_contagion, delta_SIS_contagion,
                                theta_SIS_contagion)
        outbreak_original += (res_tmp[len(res_0)-1] + res_tmp[len(res_0)-2] + res_tmp[len(res_0)-3])/3
    outbreak_original = outbreak_original / 20
    print_hgraph(hypergraph, "./results/IM_highschool/o_hypergraph.txt")
    print("outbreak original: ", outbreak_original)
    exit(2)
    perc_to_change_tmp = perc_to_change_start
    while perc_to_change_tmp <= perc_to_change_end:
        print("percentage of nodes affected by noise: ", perc_to_change_tmp)

        perc = perc_to_change_tmp

        perc = round(perc, 3)

        incr_prob = incr_prob_start

        result = {}

        while incr_prob < incr_prob_end:
            p = incr_prob
            p = round(p, 3)
            print("probability: ", p)

            result[p] = 0
            max_iter = 5

            for i in range(0, max_iter):
                noise_h = noisy_hypergraph_generator(hypergraph, p, coin_flip_probability, perc)
                # print_hgraph(noise_h, "./results/n"+str(i)+"_hypergraph.txt")
                outbreak = 0
                for i in range(0, max_iter):
                    I_tmp = I_condition.copy()
                    res_tmp = SIS_contagion(noise_h, I_tmp, time_steps, lambda_SIS_contagion, delta_SIS_contagion,theta_SIS_contagion)
                    outbreak += (res_tmp[len(res_tmp)-1] + res_tmp[len(res_tmp)-2])/2
                outbreak = outbreak/(max_iter)
                result[p] += outbreak

            result[p] = result[p]/(max_iter)

            incr_prob += incr_prob_step
            pass

        plot_outbreaks(result, outbreak_original, perc)
        perc_to_change_tmp += perc_to_change_step
    pass

def rand_init_condition_generator(hypergraph):
    """
    rand_init_condition_generator

    :param hypergraph: the hypergraph on which the simulation is performed
    :return: the initial condition of the simulation

    generate a random initial condition for the simulation
    """
    len_seed = int(len(hypergraph.get_nodes()) * start_seed_percentage)

    if len_seed < 2:
        len_seed = 2
    nodes_seed = random.sample(hypergraph.get_nodes(), len_seed)
    I_condition = {}

    for node in hypergraph.get_nodes():
        if node in nodes_seed:
            I_condition[node] = 1
        else:
            I_condition[node] = 0
    return I_condition
    pass

def calculate_size_of_edges_distribution(hypergraph):
    # calculate the size of the edges distribution
    # return a dictionary with key: size of the edge, value: number of edges with that size
    dict = {}
    for edge in hypergraph.get_edges():
        if len(edge) in dict:
            dict[len(edge)] += 1
        else:
            dict[len(edge)] = 1
    # transform the number of edges in percentage
    for key in dict.keys():
        if dict[key] > 0:
            dict[key] = dict[key]/hypergraph.get_edges().__len__()
        else:
            dict[key] = 0

    return dict

def average_size_of_edges_distribution(list_of_distribution):
    # calculate the average size of edges distribution
    # return a dictionary with key: size of the edge, value: average number of edges with that size
    dict = {}
    for distribution in list_of_distribution:
        for key in distribution.keys():
            if key in dict:
                dict[key] += distribution[key]
            else:
                dict[key] = distribution[key]
    for key in dict.keys():
        dict[key] = dict[key]/len(list_of_distribution)
    return dict

def plot_size_of_edges_distribution(dict, name_of_dist, incr_prob):
    # plot the size of edges distribution
    # dict is a dictionary with key: size of the edge, value: number of edges with that size
    x = list(dict.keys())
    print (x)
    y = list(dict.values())

    plt.clf()
    # plot as histogram
    # plt.hist(x, weights=y, bins=10, label='size of edges distribution', color='blue', linewidth=1.7)
    plt.plot(x, y, label='size of edges distribution', color='green', linewidth=1.7)
    # make a grid with lines every 0.5 in x-axis and 0.05 in y-axis
    a = plt.gca()
    # a.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    a.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    a.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    plt.grid(True, which='both', axis='both', linestyle='-', linewidth=0.5)

    # plt.figure(figsize=(30, 10))

    incr_prob = round(incr_prob, 3)

    if name_of_dist == "original_distribution":
        plt.title(str('size of edges distribution on: '+ name_of_dist))
    elif name_of_dist == "random_distribution":
        plt.title(str('size of edges distribution on: '+ name_of_dist))
    else:
        plt.title(str('size of edges distribution on: '+ name_of_dist + " \nwith probability of incrementing noise: " + str(incr_prob)))

    plt.xlabel('size of edges')
    plt.ylabel('number of edges')

    if name_of_dist == "original_distribution":
        plt.savefig("./results/" + name_of_dist + ".png")
    elif name_of_dist == "random_distribution":
        plt.savefig("./results/" + name_of_dist + ".png")
    else:
        plt.savefig("./results/" + name_of_dist + "_"+str(incr_prob)+".png")

def group_distribution_analysis(hypergraph, dist, name_dist):
    coin_flip_probability = 0.5
    incr_prob = 0.3
    perc_to_change = 1.

    distances = {}
    while incr_prob < 0.96:
        list_of_distribution = []
        for i in range(0, 5):
            noisy_hypergraph = noisy_hypergraph_generator(hypergraph, incr_prob, coin_flip_probability, perc_to_change)
            # print("check")
            list_of_distribution.append(calculate_size_of_edges_distribution(noisy_hypergraph))
        average_distribution = average_size_of_edges_distribution(list_of_distribution)
        average_distribution = dict(sorted(average_distribution.items()))
        print(average_distribution)
        distance = calculate_distance_between_distributions(average_distribution, dist)
        incr_prob = round(incr_prob, 3)
        distances[incr_prob] = distance
        # plot_size_of_edges_distribution(original_distribution, "original_distribution")
        # plot_size_of_edges_distribution(average_distribution, "average_distribution", incr_prob)
        incr_prob += 0.05

    print(distances)
    plot_distances(distances, name_dist)


def calculate_distance_between_distributions(average_distribution, original_distribution):
    # calculate the distance between the average distribution and the original distribution
    # return the distance
    avg_distance = 0
    for key in average_distribution.keys():
        if key in original_distribution.keys():
            avg_distance += abs(average_distribution[key] - original_distribution[key])
        else:
            avg_distance += average_distribution[key]

    avg_distance = avg_distance/len(average_distribution.keys())

    return avg_distance


def plot_degree_distribution(original_distribution,name_of_dist, param):

    # plot the degree distribution
    # original_distribution is a dictionary with key: degree, value: percentage of nodes with that degree
    x = list(original_distribution.keys())
    y = list(original_distribution.values())

    plt.clf()

    # plot as histogram
    # plt.hist(x, weights=y, bins=10, label='size of edges distribution', color='blue', linewidth=1.7)
    plt.plot(x, y, label='degree distribution', color='violet', linewidth=1.7)

    # plt.figure(figsize=(30, 10))
    if name_of_dist == "average_distribution":
        param = round(param, 3)

    if name_of_dist == "original_distribution":
        plt.title(str('degree distribution on: ' + str(name_of_dist)))
    else:
        plt.title(str('degree distribution on: ' + str(name_of_dist) + " \nwith probability of incrementing noise: " + str(param)))

    plt.xlabel('degree')
    plt.ylabel('percentage of nodes')

    if name_of_dist == "original_distribution":
        plt.savefig("./results/res_degree_dist/" + name_of_dist + "2.png")
    else:
        plt.savefig("./results/res_degree_dist/" + name_of_dist + "_" + str(param) + "2.png")



def average_degree_distribution(list_of_distribution):
    # calculate the average degree distribution
    # return a dictionary with key: degree, value: average percentage of nodes with that degree
    dict = {}
    for distribution in list_of_distribution:
        for key in distribution.keys():
            if key in dict:
                dict[key] += distribution[key]
            else:
                dict[key] = distribution[key]
    for key in dict.keys():
        if dict[key] > 0:
            dict[key] = dict[key]/len(list_of_distribution)
        else:
            dict[key] = 0
    return dict


def degree_analysis(hypergraph):
    original_distribution = degree_distribution(hypergraph)
    original_distribution = dict(sorted(original_distribution.items()))

    # transform the number of edges in percentage
    for key in original_distribution.keys():
        if original_distribution[key] > 0:
            original_distribution[key] = original_distribution[key]/hypergraph.get_nodes().__len__()
        else:
            original_distribution[key] = 0

    plot_degree_distribution(original_distribution, "original_distribution", 0)

    coin_flip_probability = 0.5
    incr_prob = 0.999
    perc_to_change = 1.

    distances = {}
    while incr_prob < 1.:
        print("probability: ", incr_prob)
        list_of_distribution = []
        for i in range(0, 10):
            noisy_hypergraph = noisy_hypergraph_generator(hypergraph, incr_prob, coin_flip_probability, perc_to_change)
            # print("check")
            dist = degree_distribution(noisy_hypergraph)
            for key in dist.keys():
                if dist[key] > 0:
                    dist[key] = dist[key]/hypergraph.get_nodes().__len__()
                else:
                    dist[key] = 0
            list_of_distribution.append(dist)
        average_distribution = average_degree_distribution(list_of_distribution)
        average_distribution = dict(sorted(average_distribution.items()))
        print(average_distribution)
        plot_degree_distribution(average_distribution, "average_distribution", incr_prob)
        incr_prob = round(incr_prob, 3)
        return
        distance = calculate_distance_between_distributions(average_distribution, original_distribution)
        distances[incr_prob] = distance
        incr_prob += 0.05

    #plot_distances(distances, "degree_distribution")

def print_results(res, file_name):
    # print the results in a file
    file = open(file_name, "w")
    for i in range(0, len(res)):
        print(i, " ", str(res[i]), file=file)
    file.close()


def plot_distances(distances, name_dist):
    # plot the distances between the average distribution and the original distribution
    # distances is a dictionary with key: probability of incrementing noise, value: distance
    x = list(distances.keys())
    y = list(distances.values())

    plt.clf()
    plt.plot(x, y, label='distance between average distribution and original distribution', color='blue', linewidth=1.7)
    plt.title('distance between average distribution and '+ str(name_dist))

    plt.xlabel('probability of incrementing noise')
    plt.ylabel('distance')

    # set locator every 0.1 in x-axis and 0.5 in y-axis
    a = plt.gca()
    a.xaxis.set_major_locator(ticker.MultipleLocator(0.05))
    a.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    a.yaxis.set_minor_locator(ticker.MultipleLocator(0.005))


    try:
        plt.grid(True, which='both', axis='both', linestyle='-', linewidth=0.5)
    except Exception as e:
        print(e)
        plt.grid(True)

    plt.savefig("./results/res_degree_dist/distance_"+str(name_dist)+".png")


def plot_results_small_hg(list_of_difference, res_edges):
    # plot the results with matplotlib
    # results is a dictionary, key: probability of incrementing noise, value: average number of infected nodes at the end of the simulation
    x = list(list_of_difference.keys())
    y = list(list_of_difference.values())

    plt.clf()
    plt.plot(x, y, label='outbreaks graph', color='blue', linewidth=1.7)
    plt.title('outbreaks graph')

    plt.xlabel('probability of incrementing noise')
    plt.ylabel('percentage of infected nodes at the end of simulation')
    plt.axhline(y=res_edges, color='green', linestyle='-', linewidth=1.5, label='percentage of infection in original graph')

    a = plt.gca()
    a.xaxis.set_major_locator(ticker.MultipleLocator(max(incr_prob_step, 0.1)))
    a.xaxis.set_minor_locator(ticker.MultipleLocator(0.01))
    a.yaxis.set_minor_locator(ticker.MultipleLocator(0.01))

    try:
        plt.grid(True, which='both', axis='both', linestyle='-', linewidth=0.5)
    except Exception as e:
        print(e)
        plt.grid(True)

    plt.savefig("./results/small_hg/outbreaks.png")
    pass




def small_hg_simulation():
    define_global_variables()
    r1 = 0
    r2 = {}
    for i in range(0, 10):

        hypergraph = generate_scale_free_graph()
        print_hgraph(hypergraph, "./results/small_hg/hypergraph.txt")

        nodes_degree = degree_sequence(hypergraph)
        nodes_degree = dict(sorted(nodes_degree.items(), key=lambda x: x[1], reverse=True))
        print(nodes_degree)


        I_condition_2 = s_init_condition_generator(hypergraph)

        print(I_condition_2)

        res_degree = 0

        for i in range(0, 5):
            res_degree += SIS_contagion(hypergraph, I_condition_2, time_steps, lambda_SIS_contagion, delta_SIS_contagion, theta_SIS_contagion)


        res_degree = res_degree/5

        print(res_degree)
        r1 += res_degree

        incr_prob = 0.3
        perc_to_change = 1.
        coin_flip_probability = 0.6
        incr_step = 0.05

        list_of_difference = {}
        while incr_prob < 0.96:
            incr_prob = round(incr_prob, 3)
            list_of_difference[incr_prob] = 0
            for i in range(0,10):
                noisy_hypergraph = noisy_hypergraph_generator(hypergraph, incr_prob, coin_flip_probability, perc_to_change)
                res = 0
                for i in range(0,5):
                    res += SIS_contagion(noisy_hypergraph, I_condition_2, time_steps, lambda_SIS_contagion, delta_SIS_contagion, theta_SIS_contagion)
                res = res/5
                list_of_difference[incr_prob] += res
            list_of_difference[incr_prob] = list_of_difference[incr_prob]/10
            incr_prob += incr_step
        for key in list_of_difference.keys():
            if key in r2.keys():
                r2[key] += list_of_difference[key]
            else:
                r2[key] = list_of_difference[key]

    r1 = r1/10
    for key in r2.keys():
        r2[key] = r2[key]/10
    plot_results_small_hg(r2, r1)

def main():
    hypergraph = load_high_school("../test_data/hs/hs.json")
    define_global_variables()

    I_condition = s_init_condition_generator(hypergraph)

    simulation(hypergraph, I_condition)

if __name__ == '__main__':
    main()
