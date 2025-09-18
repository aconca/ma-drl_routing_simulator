# Standard library imports
import time, math, random, os, sys, io, atexit, pickle, glob, json, re, builtins, unicodedata
from datetime import datetime
from collections import defaultdict, Counter
from typing import List, Tuple

# Other imports
import pandas as pd
import numpy as np
import simpy
import networkx as nx
import geopy.distance
from geopy.distance import geodesic
import numba
from scipy.optimize import linear_sum_assignment
from PIL import Image
import seaborn as sns
import gc
import topohub
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Matplotlib imports
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.path import Path
from matplotlib.patches import FancyArrowPatch
import matplotlib.image as mpimg
from matplotlib import cm

# Deep Learning imports
import tensorflow as tf
from tensorflow import keras
from keras import Model, Sequential, losses
from keras.optimizers import Adam
from keras.layers import Dense, Embedding, Reshape, Input, Conv2D, Flatten
from collections import deque

###############################################################################
################################    Log file    ###############################
###############################################################################

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)
os.environ["PYTHONIOENCODING"] = "utf-8"


class Logger(object):
    def __init__(self, filename='logfile.log', mode="a", encoding="utf-8"):
        self.terminal = sys.__stdout__
        self.log = open(filename, mode, encoding=encoding)
        atexit.register(self.close)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        if not self.log.closed:
            self.log.flush()

    def close(self):
        if not self.log.closed:
            self.log.close()


###############################################################################
########################    Deep Learning Framework    ########################
###############################################################################

import tensorflow as tf
from tensorflow import keras
from keras import Model, Sequential, losses
from keras.optimizers import Adam
from keras.layers import Dense, Embedding, Reshape, Input, Conv2D, Flatten
from collections import deque

# Forcing TensorFlow to use GPU - No worth using GPU for reinforcement learning in this case
#                                 since the training is done every step with small buffers
# physical_devices = tf.config.list_physical_devices('GPU')
# if len(physical_devices) > 0:
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)
#     print('GPU(s) available:')
#     print(physical_devices)
# else:
#     print('No GPU available')

###############################################################################
###############################    Constants    ###############################
###############################################################################

# HOT PARAMS - This parameters should be revised before every simulation
pathings = ['hop', 'dataRate', 'dataRateOG', 'slant_range', 'Q-Learning', 'Deep Q-Learning']
pathing = pathings[
    3]  # dataRateOG is the original datarate. If we want to maximize the datarate we have to use dataRate, which is the inverse of the datarate

FL_Test = False  # If True, it plots the model divergence the model divergence between agents
plotSatID = True  # If True, plots the ID of each satellite
plotAllThro = True  # If True, it plots throughput plots for each single path between gateways. If False, it plots a single figure for overall Throughput
plotAllCon = True  # If True, it plots congestion maps for each single path between gateways. If False, it plots a single figure for overall congestion

movementTime = 5  # Every movementTime seconds, the satellites positions are updated and the graph is built again
# If do not want the constellation to move, set this parameter to a bigger number than the simulation time
ndeltas = 5805.44 / 20  # 1 Movement speedup factor. Every movementTime sats will move movementTime*ndeltas space. If bigger, will make the rotation distance bigger

Train = True  # Global for all scenarios with different number of GTs. if set to false, the model will not train any of them
explore = True  # If True, makes random actions eventually, if false only exploitation
importQVals = False  # imports either QTables or NN from a certain path
onlinePhase = False  # when set to true, each satellite becomes a different agent. Recommended using this with importQVals=True and explore=False
if onlinePhase:  # Just in case
    explore = False
    importQVals = True
else:
    FL_Test = False

w1 = 20  # rewards the getting to empty queues
w2 = 20  # rewards getting closes phisycally
w4 = 5  # Normalization for the distance reward, for the traveled distance factor

gamma = 0.99  # greedy factor. Smaller -> Greedy. Optimized params: 0.6 for Q-Learning, 0.99 for Deep Q-Learning

GTs = [31]  # number of gateways to be tested
# Gateways are taken from https://www.ksat.no/ground-network-services/the-ksat-global-ground-station-network/ (Except for Malaga and Aalborg)
# GTs = [i for i in range(2,9)] # This is to make a sweep where scenarios with all the gateways in the range are considered

# Physical constants
rKM = 500  # radio in km of the coverage of each gateway
Re = 6378e3  # Radius of the earth [m]
G = 6.67259e-11  # Universal gravitational constant [m^3/kg s^2]
Me = 5.9736e24  # Mass of the earth
Te = 86164.28450576939  # Time required by Earth for 1 rotation
Vc = 299792458  # Speed of light [m/s]
k = 1.38e-23  # Boltzmann's constant
eff = 0.55  # Efficiency of the parabolic antenna
C_TER = 2e8  # m/s fibra (terra)
C_SPACE = 3e8  # m/s spazio

# Downlink parameters
f = 20e9  # Carrier frequency GEO to ground (Hz)
B = 500e6  # Maximum bandwidth
maxPtx = 10  # Maximum transmission power in W
Adtx = 0.26  # Transmitter antenna diameter in m
Adrx = 0.26  # 0.33 Receiver antenna diameter in m
pL = 0.3  # Pointing loss in dB
Nf = 2  # 1.5 Noise figure in dB
Tn = 290  # 50 Noise temperature in K
min_rate = 10e3  # Minimum rate in kbps

# Uplink Parameters
balancedFlow = False  # if set to true all the generated traffic at each GT is equal
totalFlow = 2 * 1000000000  # Total average flow per GT when the balanced traffc option is enabled. Malaga has 3*, LA has 3*, Nuuk/500
avUserLoad = 5e6  # average traffic usage per second in bits (5 Mbps per user)

# User to node connection parameters
f_user = 3.5e9  # es: 3.5 GHz for 5G
B_user = 20e6  # Bandwidth
maxPtx_user = 10  # Max tx power
Adtx_user = 0.05  # antenna tx diameter(m) -> smaller than satellite antennas
Adrx_user = 0.2  # antenna rx diameter(m)
pL_user = 2.0  # Pointing loss(dB) -> higher for smaller and
Nf_user = 7  # Noise figure
Tn_user = 300  # Noise Temperature (K)
min_rate_user = 20e6  # min rate (1 Mbps)

# Block
BLOCK_SIZE = 64800

# Movement and structure
# movementTime= 0.05      # Every movementTime seconds, the satellites positions are updated and the graph is built again
#                         # If do not want the constellation to move, set this parameter to a bigger number than the simulation time
# ndeltas     = 5805.44/20#1 Movement speedup factor. This number will multiply deltaT. If bigger, will make the rotation distance bigger
saveISLs = True  # save ISLs map
const_moved = False  # Movement flag. If up, it means it has moved
matching = 'Greedy'  # ['Markovian', 'Greedy']
minElAngle = 30  # For satellites. Value is taken from NGSO constellation design chapter.
mixLocs = False  # If true, every time we make a new simulation the locations are going to change their order of selection
rotateFirst = False  # If True, the constellation starts rotated by 1 movement defined by ndeltas

# State pre-processing
coordGran = 20  # Granularity of the coordinates that will be the input of the DNN: (Lat/coordGran, Lon/coordGran)
diff = True  # If up, the state space gives no coordinates about the neighbor and destination positions but the difference with respect to the current positions
diff_lastHop = True  # If up, this state is the same as diff, but it includes the last hop where the block was in order to avoid loops
reducedState = False  # if set to true the DNN will receive as input only the positional information, but not the queueing information
notAvail = 0  # this value is set in the state space when the satellite neighbour is not available

# Learning Hyperparameters
ddqn = True  # Activates DDQN, where now there are two DNNs, a target-network and a q-network
# importQVals = False     # imports either QTables or NN from a certain path
plotPath = False  # plots the map with the path after every decision
alpha = 0.25  # learning rate for Q-Tables
alpha_dnn = 0.01  # learning rate for the deep neural networks
# gamma       = 0.99       # greedy factor. Smaller -> Greedy. Optimized params: 0.6 for Q-Learning, 0.99 for Deep Q-Learning
epsilon = 0.1  # exploration factor for Q-Learning ONLY
tau = 0.1  # rate of copying the weights from the Q-Network to the target network
learningRate = 0.001  # Default learning rate for Adam optimizer
plotDeliver = False  # create pictures of the path every 1/10 times a data block gets its destination
# plotSatID   = False     # If True, plots the ID of each satellite
GridSize = 8  # Earth divided in GridSize rows for the grid. Used to be 15
winSize = 20  # window size for the representation in the plots
markerSize = 50  # Size of the markers in the plots
nTrain = 2  # The DNN will train every nTrain steps
noPingPong = True  # when a neighbour is the destination satellite, send there directly without going through the dnn (Change policy)

# Queues & State
infQueue = 5000  # Upper boundary from where a queue is considered as infinite when obserbing the state
queueVals = 10  # Values that the observed Queue can have, being 0 the best (Queue of 0) and max the worst (Huge queue or inexistent link).
latBias = 90  # This value is added to the latitude of each position in the state space. This can be done to avoid negative numbers
lonBias = 180  # Same but with longitude

# rewards
ArriveReward = 50  # Reward given to the system in case it sends the data block to the satellite linked to the destination gateway
# w1          = 20        # rewards the getting to empty queues
# w2          = 20        # rewards getting closes phisycally
# w4          = 5         # Normalization for the distance reward, for the traveled distance factor
againPenalty = -10  # Penalty if the satellite sends the block to a hop where it has already been
unavPenalty = -10  # Penalty if the satellite tries to send the block to a direction where there is no linked satellite
biggestDist = -1  # Normalization factor for the distance reward. This is updated in the creation of the graph.
firstMove = True  # The biggest slant range is only computed the first time in order to avoid this value to be variable
distanceRew = 4  # 1: Distance reward normalized to total distance.
# 2: Distance reward normalized to average moving possibilities
# 3: Distance reward normalized to maximum close up
# 4: Distance reward normalized by max isl distance ~3.700 km for Kepler constellation. This is the one used in the papers.
# 5: Only negative rewards proportional to traveled distance normalized by 1.000 km

# Deep Learning
MAX_EPSILON = 0.99  # Maximum value that the exploration parameter can have
MIN_EPSILON = 0.001  # Minimum value that the exploration parameter can have
LAMBDA = 0.0005  # This value is used to decay the epsilon in the deep learning implementation
decayRate = 4  # sets the epsilon decay in the deep learning implementatio. If higher, the decay rate is slower. If lower, the decay is faster
Clipnorm = 1  # Maximum value to the nom of the gradients. Prevents the gradients of the model parameters with respect to the loss function becoming too large
hardUpdate = 1  # if up, the Q-network weights are copied inside the target network every updateF iterations. if down, this is done gradually
updateF = 1000  # every updateF updates, the Q-Network will be copied inside the target Network. This is done if hardUpdate is up
batchSize = 16  # batchSize samples are taken from bufferSize samples to train the network
bufferSize = 50  # bufferSize samples are used to train the network

# Stop Loss
# Train       = True      # Global for all scenarios with different number of GTs. if set to false, the model will not train any of them
stopLoss = False  # activates the stop loss function
nLosses = 50  # Nº of loss samples used for the stop loss
lThreshold = 0.5  # If the mean of the last nLosses are lower than lossThreshold, the mdoel stops training
TrainThis = Train  # Local for a single scenario with a certain number of GTs. If the stop loss is activated, this will be set to False and the scenario will not train anymore.
# When another scenario is about to run, TrainThis will be set to Train again

# Other
CurrentGTnumber = -1  # Number of active gateways. This number will be updated every time a gateway is added. In the simulation it will iterate the GTs list

###############################################################################
###############################      Paths      ###############################
###############################################################################

# nnpath      = './pre_trained_NNs/qNetwork_8GTs_6secs_nocon.h5'
# nnpathTarget= './pre_trained_NNs/qTarget_8GTs_6secs_nocon.h5'
# nnpath      = './pre_trained_NNs/qNetwork_3GTs.h5'
# nnpathTarget= './pre_trained_NNs/qTarget_3GTs.h5'
nnpath = './pre_trained_NNs/qNetwork_2GTs.h5'
nnpathTarget = './pre_trained_NNs/qTarget_2GTs.h5'
# nnpath      = './pre_trained_NNs/qNetwork_2GTs_lastHop.h5'
# nnpathTarget= './pre_trained_NNs/qTarget_2GTs_lastHop.h5'
tablesPath = './pre_trained_NNs/qTablesExport_8GTs/'

if __name__ == '__main__':
    # nnpath          = f'./pre_trained_NNs/qNetwork_8GTs.h5'
    outputPath = './Results/{}_{}s_[{}]_Del_[{}]_w1_[{}]_w2_{}_GTs/'.format(pathing, float(
        pd.read_csv("inputRL.csv")['Test length'][0]), ArriveReward, w1, w2, GTs)
    populationMap = 'Population Map/gpw_v4_population_count_rev11_2020_15_min.tif'

###############################################################################
#################################    Simpy    #################################
###############################################################################

receivedDataBlocks = []
createdBlocks = []
seed = np.random.seed(1)
upGSLRates = []
downGSLRates = []
interRates = []
intraRate = []


def getBlockTransmissionStats(sim_time_s, locations, earth, *args, **kwargs):
    """
    Returns (results, allLatencies, pathBlocks, blocks) like the original,
    but also works when there are no gateways (terrestrial mode).
    """
    # ---- If gateways exist, use original behavior (compat) ----
    try:
        if getattr(earth, "gateways", None) and len(earth.gateways) > 0:
            first = earth.gateways[0]  # <=== line that previously crashed
            # TODO: Add original gateway logic here if needed
    except Exception:
        pass  # if something goes wrong, fall back to terrestrial branch anyway

    # ---- TERRESTRIAL-ONLY Branch ----
    # Choose the pair to observe: first earth.observed_pair, then first two active
    src_name = dst_name = None
    if hasattr(earth, "observed_pair") and earth.observed_pair:
        src_name, dst_name = earth.observed_pair
    else:
        # fallback: first two active Cities
        active = getattr(earth, "active_terrestrial_nodes", []) or []
        if len(active) >= 2:
            src_name, dst_name = str(active[0].name), str(active[1].name)

    # If I don't have a pair, return "empty" but consistent results
    if not src_name or not dst_name:
        empty = {
            "pair": None,
            "delivered_blocks": 0,
            "avg_latency_s": 0.0,
            "avg_tx_s": 0.0,
            "avg_prop_s": 0.0,
            "throughput_bps": 0.0,
        }
        return empty, [], [], []

    # Filter received blocks related to the pair (both directions)
    try:
        blocks_all = receivedDataBlocks  # global like in the original
    except NameError:
        blocks_all = []

    def _name_of(x):
        if hasattr(x, "name"):
            return str(x.name)
        return str(x)

    pair_blocks = []
    for b in blocks_all:
        s = _name_of(getattr(b, "source", ""))
        d = _name_of(getattr(b, "destination", ""))
        if (s == src_name and d == dst_name) or (s == dst_name and d == src_name):
            pair_blocks.append(b)

    # Extract latencies
    l_tot, l_tx, l_prop = [], [], []
    for b in pair_blocks:
        tx = getattr(b, "txLatency", 0.0) or 0.0
        prop = getattr(b, "propLatency", 0.0) or 0.0
        # in some setups queueLatency exists, otherwise 0
        queue = getattr(b, "queueLatency", 0.0) or 0.0
        l_tx.append(tx)
        l_prop.append(prop)
        l_tot.append(tx + prop + queue)

    # Average throughput (simple): sum delivered bits / simulation time
    try:
        block_bits = BLOCK_SIZE  # same global used in tx
    except NameError:
        block_bits = 1e6  # fallback 1 Mbit
    delivered_bits = block_bits * len(pair_blocks)
    thr_bps = delivered_bits / max(sim_time_s, 1e-9)

    results = {
        "pair": (src_name, dst_name),
        "delivered_blocks": len(pair_blocks),
        "avg_latency_s": float(np.mean(l_tot)) if l_tot else 0.0,
        "avg_tx_s": float(np.mean(l_tx)) if l_tx else 0.0,
        "avg_prop_s": float(np.mean(l_prop)) if l_prop else 0.0,
        "throughput_bps": float(thr_bps),
    }

    # allLatencies → keep the series of total latencies (like in the original)
    allLatencies = l_tot

    # pathBlocks → if the rest of the code uses it for the map, pass the pair blocks
    pathBlocks = pair_blocks

    # blocks → you can return all received or only pair ones; I choose all (for compat)
    blocks = blocks_all

    return results, allLatencies, pathBlocks, blocks


def simProgress(simTimelimit, env):
    timeSteps = 100
    timeStepSize = simTimelimit / timeSteps
    progress = 1
    startTime = time.time()
    yield env.timeout(timeStepSize)
    while True:
        elapsedTime = time.time() - startTime
        estimatedTimeRemaining = elapsedTime * (timeSteps / progress) - elapsedTime
        print(
            "Simulation progress: {}% Estimated time remaining: {} seconds Current simulation time: {}".format(progress,
                                                                                                               int(estimatedTimeRemaining),
                                                                                                               env.now),
            end='\r')
        yield env.timeout(timeStepSize)
        progress += 1


###############################################################################
############################# Federated Learning ##############################
###############################################################################

FL_techs = ['nothing', 'modelAnticipation', 'plane', 'full', 'combination']
FL_tech = FL_techs[
    4]  # dataRateOG is the original datarate. If we want to maximize the datarate we have to use dataRate, which is the inverse of the datarate
if FL_tech == 'combination':
    global FL_counter
    FL_counter = 1

if pathing != 'Deep Q-Learning':
    FL_Test = False

if FL_Test:
    CKA_Values = []  # CKA matrix
    num_samples = 10  # number of random samples to test the divergence between models
    print(f'Federated Learning ongoing: {FL_tech}. Number of random samples to test divergence: {num_samples}')


def generate_test_data(num_samples, include_not_avail=False):
    data = []
    queue_values = np.arange(0, 11)  # Possible queue values from 0 to 10
    # Set probabilities: 0 at 35%, 10 at 20%, and 5% each for values 1-9
    queue_probs = [0.35] + [0.05] * 9 + [0.20]

    for _ in range(num_samples):
        sample = []
        if diff_lastHop:
            sample.append(random.randint(0, 4))
        # Queue Scores for each direction: Up, Down, Right, Left (4 scores each)
        for _ in range(4):
            # Queue scores biased towards 0 and 10
            sample.extend(np.random.choice(queue_values, 4, p=queue_probs))

            # Relative positions for each direction: latitude and longitude
            sample.append(np.random.uniform(-2, 2))  # Latitude relative position
            sample.append(np.random.uniform(-2, 2))  # Longitude relative position

        # Absolute positions
        sample.append(np.random.uniform(0, 9))  # Absolute latitude normalized
        sample.append(np.random.uniform(0, 18))  # Absolute longitude normalized

        # Destination differential coordinates
        sample.append(np.random.uniform(-2, 2))  # Destination differential latitude
        sample.append(np.random.uniform(-2, 2))  # Destination differential longitude

        # Optionally include not available values
        if include_not_avail and np.random.rand() < 0.1:  # 10% chance to introduce a -1 value
            idx_to_replace = np.random.choice(len(sample), int(0.1 * len(sample)), replace=False)
            sample[idx_to_replace] = -1

        data.append(sample)

    return np.array(data)


def get_models(earth):
    models = []
    model_names = []
    for plane in earth.LEO:
        for sat in plane.sats:
            models.append(sat.DDQNA.qNetwork)
            model_names.append(sat.ID)
    return models, model_names


def average_model_weights(models):
    """Average weights of multiple trained models."""
    weights = [model.get_weights() for model in models]
    new_weights = [np.mean(np.array(w), axis=0) for w in zip(*weights)]
    return new_weights


def full_federated_learning(models):
    averaged_weights = average_model_weights(models)
    for model in models:
        model.set_weights(averaged_weights)


def federate_by_plane(models, model_names):
    """Perform Federated Averaging within each orbital plane."""
    plane_dict = {}
    for model, name in zip(models, model_names):
        plane = name.split('_')[0]
        if plane in plane_dict:
            plane_dict[plane].append(model)
        else:
            plane_dict[plane] = [model]
    for plane_models in plane_dict.values():
        averaged_weights = average_model_weights(plane_models)
        for model in plane_models:
            model.set_weights(averaged_weights)


def model_anticipation_federate(models, model_names):
    """Perform Model Anticipation Federated Learning."""
    plane_dict = {}
    # Group models by orbital plane
    for model, name in zip(models, model_names):
        plane = name.split('_')[0]
        if plane not in plane_dict:
            plane_dict[plane] = []
        plane_dict[plane].append((model, name))

    # Process each plane for model anticipation
    for plane_models in plane_dict.values():
        # Sort models by their identifiers within the plane
        plane_models.sort(key=lambda x: int(x[1].split('_')[1]))
        for i in range(1, len(plane_models)):
            prev_model_weights = plane_models[i - 1][0].get_weights()
            current_model = plane_models[i][0]
            current_weights = current_model.get_weights()
            # Average weights from the previous model
            new_weights = [(w1 + w2) / 2 for w1, w2 in zip(current_weights, prev_model_weights)]
            current_model.set_weights(new_weights)


def update_sats_models(earth, models, model_names):
    '''Update each satellite model for the updated one'''
    print('Updating satellites models...')
    for model, satID in zip(models, model_names):
        sat = findByID(earth, satID)
        sat.DDQNA.qNetwork = model
        if ddqn:
            sat.DDQNA.qTarget = model


def compute_full_cka_matrix(models, data):
    """Compute the full CKA matrix for a list of models."""

    def gram_matrix(X):
        """Calculate the Gram matrix from layer activations."""
        n = X.shape[0]
        X = X - X.mean(axis=0)
        return X @ X.T / n

    def cka(G, H):
        """Compute the CKA metric."""
        return np.trace(G @ H) / np.sqrt(np.trace(G @ G) * np.trace(H @ H))

    def compute_cka(model1, model2, data):
        """Compute the CKA between layers of two models using data."""
        intermediate_model1 = tf.keras.Model(inputs=model1.input, outputs=[layer.output for layer in model1.layers])
        intermediate_model2 = tf.keras.Model(inputs=model2.input, outputs=[layer.output for layer in model2.layers])
        activations1 = intermediate_model1(data)
        activations2 = intermediate_model2(data)
        return np.mean([cka(gram_matrix(np.array(act1)), gram_matrix(np.array(act2))) for act1, act2 in
                        zip(activations1, activations2)])

    n = len(models)
    cka_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            if i == j:
                cka_matrix[i, j] = 1.0
            else:
                cka_matrix[i, j] = cka_matrix[j, i] = compute_cka(models[i], models[j], data)
    return cka_matrix


def compute_average_cka(cka_matrix):
    """Compute the average CKA value from a CKA matrix."""
    triu_indices = np.triu_indices_from(cka_matrix, k=1)
    return np.mean(cka_matrix[triu_indices])


def perform_FL(earth):  # , outputPath):

    # path = outputPath + 'FL' + str(len(earth.gateways)) + 'GTs/'
    # os.makedirs(path, exist_ok=True)
    print('----------------------------------')
    print(f'Federated Learning. Performing: {FL_tech}')

    data = generate_test_data(num_samples, include_not_avail=False)
    models, model_names = get_models(earth)

    CKA_Values_before = compute_full_cka_matrix(models, data)

    if FL_tech == 'nothing':
        return CKA_Values_before, CKA_Values_before

    if FL_tech == 'modelAnticipation':
        model_anticipation_federate(models, model_names)
    elif FL_tech == 'plane':
        federate_by_plane(models, model_names)
    elif FL_tech == 'full':
        full_federated_learning(models)
    elif FL_tech == 'combination':
        global FL_counter
        if FL_counter == 1:
            print(f'Model Anticipation, counter = {FL_counter}')
            FL_counter += 1
            model_anticipation_federate(models, model_names)

        elif FL_counter == 2:
            print(f'Plane FL, counter = {FL_counter}')
            FL_counter += 1
            federate_by_plane(models, model_names)

        elif FL_counter > 2:
            print(f'Global FL, counter = {FL_counter}')
            FL_counter = 1
            full_federated_learning(models)

    CKA_Values_after = compute_full_cka_matrix(models, data)
    update_sats_models(earth, models, model_names)

    print('----------------------------------')
    return CKA_Values_before, CKA_Values_after


def plot_cka_over_time(cka_data, outputPath, nGTs):
    """
    Plots each CKA value over time in milliseconds, connecting 'before' and 'after' points with a dashed line
    and using different colors for each type of dot, with quartile ranges represented by error bars.

    Parameters:
    - cka_data: List of [CKA_before, CKA_after, timestamp] entries.
    """
    path = outputPath + 'FL/'
    os.makedirs(path, exist_ok=True)  # create output path

    # Extract times and calculate CKA values for before and after
    times = [entry[2] * 1000 for entry in cka_data]  # Convert time to milliseconds
    cka_before_values = [np.mean(entry[0]) for entry in cka_data]
    cka_after_values = [np.mean(entry[1]) for entry in cka_data]

    # Calculate quartile ranges for before and after values
    cka_before_quartiles = [np.percentile(entry[0], [25, 75]) for entry in cka_data]
    cka_after_quartiles = [np.percentile(entry[1], [25, 75]) for entry in cka_data]
    cka_before_25th, cka_before_75th = zip(*cka_before_quartiles)
    cka_after_25th, cka_after_75th = zip(*cka_after_quartiles)

    # Construct the sequence for line plot: interleave before and after mean values
    line_times = [time for time in times for _ in (0, 1)]
    line_values = [val for pair in zip(cka_before_values, cka_after_values) for val in pair]

    # Set y-axis limits with margin to avoid cutting T-caps and ensure the max is exactly 1
    y_min = min(min(cka_before_25th), min(cka_after_25th)) * 0.95
    y_max = 1

    # Plotting
    plt.figure(figsize=(10, 6))

    # Line connecting mean CKA values
    plt.plot(line_times, line_values, label='CKA Value Sequence', color='gray', linestyle='-.', alpha=0.7)

    # Error bars for 'CKA Before FL' and 'CKA After FL' with T-caps
    cka_before_yerr = [np.abs(np.subtract(cka_before_values, cka_before_25th)),
                       np.abs(np.subtract(cka_before_75th, cka_before_values))]
    cka_after_yerr = [np.abs(np.subtract(cka_after_values, cka_after_25th)),
                      np.abs(np.subtract(cka_after_75th, cka_after_values))]

    plt.errorbar(times, cka_before_values, yerr=cka_before_yerr, fmt='s', color='blue',
                 ecolor='blue', capsize=8, capthick=2, label='CKA Before FL Quartiles')
    plt.errorbar(times, cka_after_values, yerr=cka_after_yerr, fmt='s', color='green',
                 ecolor='green', capsize=8, capthick=2, label='CKA After FL Quartiles')

    # Set x-axis and y-axis limits with a dynamic y-axis minimum
    plt.xlim(min(times) - 20, max(times) + 20)
    # plt.ylim(y_min, y_max)
    plt.ticklabel_format(style='plain', axis='y')  # Disable scientific notation for y-axis

    # Labels and title
    plt.xlabel('Time (ms)')
    plt.ylabel('CKA Value')
    plt.title('CKA Values Over Time (ms)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(path, f'CKA_over_time_{str(nGTs)}_GTs.png'), dpi=300, bbox_inches='tight')

    # Save mean CKA values over time
    mean_cka_values = np.column_stack((times, cka_before_values, cka_after_values))
    np.savetxt(os.path.join(path, 'mean_cka_values.csv'), mean_cka_values, delimiter=',',
               header="Time_ms,CKA_Before,CKA_After", comments='')

    # Save individual CKA matrices before and after FL for each timestamp
    for i, entry in enumerate(cka_data):
        np.savetxt(os.path.join(path, f'cka_matrix_before_{i}.csv'), entry[0], delimiter=',')
        np.savetxt(os.path.join(path, f'cka_matrix_after_{i}.csv'), entry[1], delimiter=',')


###############################################################################
################################# Helpers #####################################
###############################################################################

def haversine(lon1, lat1, lon2, lat2, radius=6371):
    """
    Calculate the distance in kilometers between two geographic coordinates (lat, lon).
    """

    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return radius * c


def _active_nodes_xy(earth, active_names):
    """
    Ritorna [(name, lon, lat)] per i nomi in active_names che esistono in earth.terr_graph o earth.space_graph.
    """
    out = []

    # Prova prima nel grafo terrestre
    G_terr = getattr(earth, "terr_graph", None)
    if G_terr is not None:
        set_names_terr = set(str(n) for n in G_terr.nodes)
        for nm in active_names:
            key = str(nm)
            if key in set_names_terr:
                nd = G_terr.nodes[key]
                lon = nd.get("lon")
                lat = nd.get("lat")
                if lon is not None and lat is not None:
                    out.append((key, float(lon), float(lat)))

    # Se non trovato nel terrestre, prova nel grafo spaziale
    G_space = getattr(earth, "space_graph", None)
    if G_space is not None:
        set_names_space = set(str(n) for n in G_space.nodes)
        for nm in active_names:
            key = str(nm)
            if key in set_names_space and not any(item[0] == key for item in out):
                nd = G_space.nodes[key]
                lon = nd.get("lon")
                lat = nd.get("lat")
                if lon is not None and lat is not None:
                    out.append((key, float(lon), float(lat)))

    return out


def _step_key(step):
    return step[0] if isinstance(step, (tuple, list)) else step


def _get_edge(g, a, b):
    """
    Get edge data from graph, handling both directions and MultiGraph.
    Returns edge data dict or None if not found.
    """
    if g is None:
        return None
    try:
        # Try a->b first, then b->a
        ed = g.get_edge_data(a, b)
        if ed is None:
            ed = g.get_edge_data(b, a)

        if ed is None:
            return None

        if isinstance(ed, dict) and ed and all(isinstance(v, dict) for v in ed.values()):
            ed = next(iter(ed.values()))

        return ed
    except Exception:
        return None


def coerce_key(k):
    """Normalizza una voce di path in una 'chiave' confrontabile:
    - tuple/list -> prende il primo elemento
    - gateway/terrestrial: nome (stringa)
    - satellite: ID (int) lasciato int
    """
    if isinstance(k, (list, tuple)) and k:
        k = k[0]
    # lascia int (ID satellite)
    return k


def coerce_path_only(path):
    """Converts path to a flat list of normalized keys, removes None/''."""
    if not path:
        return []
    out = []
    for k in path:
        kk = coerce_key(k)
        if kk is None:
            continue
        # convert to string for terrestrial/gateway nodes; keep int for satellites
        if not isinstance(kk, (int, str)):
            kk = str(kk)
        out.append(kk)
    return out


def edge_cost_from_graph(earth, u_key, v_key, block_size_bits=None, verbose=False):
    if block_size_bits is None:
        block_size_bits = BLOCK_SIZE

    u = coerce_key(u_key)
    v = coerce_key(v_key)

    # -------------------- TERRESTRE --------------------
    ed = _get_edge(getattr(earth, "terr_graph", None), u, v)
    if ed:
        data_rate = ed.get("dataRate", None)
        prop = ed.get("propDelay", None)

        # fallback propagation: fiber length
        if prop is None:
            length_km = ed.get("length_km", 0.0) or 0.0
            C_FIBER = 200_000.0  # km/s
            prop = (length_km / C_FIBER) if length_km else 0.0

        # fallback rate
        if not data_rate or not math.isfinite(data_rate) or data_rate <= 0:
            data_rate = (ed.get("capacity_bps", None) or
                         ed.get("dataRateOG", None) or
                         1e8)

        # if accidentally s/bit, normalize
        if data_rate < 1e-3:
            # s/bit → bps
            data_rate = 1.0 / data_rate

        tx = float(block_size_bits) / float(data_rate)

        return float(prop), float(tx), float(data_rate)

    # -------------------- SPAZIO (ISL/GSL) --------------------
    ed = _get_edge(getattr(earth, "space_graph", None), u, v)
    if ed:
        edge_type = ed.get("type", "")
        raw = ed.get("dataRateOG", None)
        # some graphs use 'dataRate'
        if raw is None:
            raw = ed.get("dataRate", None)

        # propagation from slant range: the edge can report meters or kilometers
        sl_raw = ed.get("slant_range")
        if sl_raw is None:
            sl_raw = ed.get("slant_range_km")

        if sl_raw is None:
            sr_m = ed.get("slant_range_m")
            sl_km = (sr_m / 1000.0) if sr_m else 0.0
        else:
            # if the value seems to be in meters (e.g. >1e5), convert
            sl_km = float(sl_raw)
            if sl_km > 1e5:
                sl_km /= 1000.0

        C_VAC = 299_792.458  # km/s
        prop = (sl_km / C_VAC) if sl_km else 0.0

        # normalize the raw value
        data_rate_bps = None
        tx = None

        # if raw is not finite or <=0, try fallback
        if (raw is None) or (not isinstance(raw, (int, float))) or (not math.isfinite(raw)) or (raw <= 0):
            # try capacity_bps as fallback
            cap = ed.get("capacity_bps", None)
            if cap and cap > 1e3 and math.isfinite(cap):
                data_rate_bps = float(cap)
                tx = float(block_size_bits) / data_rate_bps
            else:
                # conservative fallback: 100 Mbps
                data_rate_bps = 1e8
                tx = float(block_size_bits) / data_rate_bps
        else:
            if raw < 1e-3:
                # s/bit
                data_rate_bps = 1.0 / float(raw)
                tx = float(block_size_bits) * float(raw)
            elif raw >= 1e3:
                # bps
                data_rate_bps = float(raw)
                tx = float(block_size_bits) / data_rate_bps
            else:
                # gray zone (e.g. 0.001..1000), often s/bit in your sim
                data_rate_bps = 1.0 / float(raw)
                tx = float(block_size_bits) * float(raw)

        return float(prop), float(tx), float(data_rate_bps if data_rate_bps else 0.0)

    # No edge
    return None, None, None


def estimate_path_cost(earth, path, block_size_bits=None, verbose=False, label=None):
    """
    Returns total cost (s) = sum (prop + tx) on path edges.
    If an edge is missing or has invalid rate -> returns +inf.
    If verbose=True, prints detailed cost breakdown.
    """
    p = coerce_path_only(path)
    if not p or len(p) < 2:
        if verbose:
            print("[cost-verbose] path empty/too short")
        return float("inf")
    if block_size_bits is None:
        block_size_bits = BLOCK_SIZE

    if verbose and label:
        print(f"[cost-verbose] {label}")

    total_prop = 0.0
    total_tx = 0.0
    for a, b in zip(p[:-1], p[1:]):
        if verbose:
            # search first in space_graph to understand the type
            ed_s = _get_edge(getattr(earth, 'space_graph', None), a, b)
            ed_t = _get_edge(getattr(earth, 'terr_graph', None), a, b)
            if ed_s is not None:
                kind = ed_s.get('type', 'SPACE')
            elif ed_t is not None:
                kind = f"TERR-{ed_t.get('type', 'TERR')}"
            else:
                kind = "??"

        prop, tx, rate = edge_cost_from_graph(earth, a, b, block_size_bits=block_size_bits, verbose=verbose)
        if prop is None or tx is None:
            if verbose:
                print(f"[cost-verbose] MISSING  {a} -> {b}")
            return float("inf")

        if verbose:
            print(f"[cost-verbose] {kind:4}  {a} -> {b}  rate={rate: .2e}  prop={prop:.6f}  tx={tx:.6f}")

        total_prop += prop
        total_tx += tx

    total = total_prop + total_tx
    if verbose:
        print(f"[cost-verbose] TOTAL = {total:.6f}s")
    return total


def compute_hybrid_path(src_name, dst_name, earth, prefer="latency", block_size_bits=None, verbose=False):
    """
    Builds a hybrid path City -> (terrestrial) -> GWs -> (space) -> GWd -> (terrestrial) -> City.
    Chooses the gateway pair that minimizes the estimated cost.
    Returns the path as a list of mixed keys (terrestrial/gateway names + sat ID int).
    If no valid hybrid exists, returns [].
    """
    # local helpers already present in your code:
    # - getShortestPathTerrestrial(src, dst, G)
    # - getShortestPath(GT_A, GT_B, earth.pathParam, graph)  (spaziale fra gateway)
    Gt = getattr(earth, "terr_graph", None)
    Gs = getattr(earth, "space_graph", None)
    if block_size_bits is None:
        block_size_bits = BLOCK_SIZE

    if Gt is None or Gs is None:
        return []

    # available gateways
    gateways = list(getattr(earth, "gateways", []))
    if not gateways:
        return []

    best = (float("inf"), None)  # (cost, path)

    # try all pairs (GW_src, GW_dst)
    valid_combinations = 0
    for gws in gateways:
        gw_s = str(getattr(gws, "name", None))
        if not gw_s:
            continue
        # terrestrial path source -> GW_s
        p1 = getShortestPathTerrestrial(src_name, gw_s, Gt)
        if not p1:
            continue

        for gwd in gateways:
            gw_d = str(getattr(gwd, "name", None))
            if not gw_d:
                continue
            # terrestrial path GW_d -> destination
            p3 = getShortestPathTerrestrial(gw_d, dst_name, Gt)
            if not p3:
                continue

            # space path GW_s <-> GW_d (space graph)
            try:
                p2 = getShortestPath(gw_s, gw_d, earth.pathParam, earth.space_graph)
            except Exception:
                p2 = None
            if not p2 or len(p2) < 2:
                continue

            valid_combinations += 1

            # combine (avoid gateway duplicates at the border)
            p1c = coerce_path_only(p1)
            p2c = coerce_path_only(p2)
            p3c = coerce_path_only(p3)

            combo = []
            if p1c:
                combo.extend(p1c)
            if p2c:
                if combo and p2c[0] == combo[-1]:
                    combo.extend(p2c[1:])
                else:
                    combo.extend(p2c)
            if p3c:
                if combo and p3c[0] == combo[-1]:
                    combo.extend(p3c[1:])
                else:
                    combo.extend(p3c)

            # cost
            cost = estimate_path_cost(earth, combo, block_size_bits=block_size_bits, verbose=False)
            if verbose:
                print(f"[hyb] {src_name} -> {gw_s} ~~space~~ {gw_d} -> {dst_name}  cost={cost:.4f}")

            if cost < best[0]:
                best = (cost, combo)

    return best[1] if best[1] else []


def _norm(s):
    if s is None:
        return None
    _ud = unicodedata
    s = _ud.normalize('NFKD', str(s))
    s = ''.join(ch for ch in s if not _ud.combining(ch))
    s = s.lower().strip()
    s = re.sub(r'\s+', ' ', s)
    return s


def resolve_gateway_terr_key(G, gw_display_name):
    gwN = str(gw_display_name)
    gw_norm_full = _norm(gwN)
    gw_head = gwN.split(',')[0].strip()
    gw_norm_head = _norm(gw_head)

    if gwN in G and G.nodes[gwN].get('type') == 'Gateway':
        return gwN, 'exact'
    if gw_head in G and G.nodes[gw_head].get('type') == 'Gateway':
        return gw_head, 'head'
    for n, d in G.nodes(data=True):
        if d.get('type') != 'Gateway':
            continue
        if _norm(n) == gw_norm_full or _norm(n) == gw_norm_head:
            return n, 'norm'
    for n, d in G.nodes(data=True):
        if d.get('type') != 'Gateway':
            continue
        if gw_norm_head in _norm(n) or _norm(n) in gw_norm_head:
            return n, 'contains'
    return None, 'not-found'


def gw_space_key(terr_key, spaceG, default_display_name):
    return terr_key if terr_key in spaceG else default_display_name


def edge_latency_and_rate(earth, u, v):
    """
    Ritorna (propDelay_s, dataRate_bps) per l’arco u-v cercando nei grafi terrestri e spaziali.
    Calcola la prop se mancante usando distanza (terra: ~2e8 m/s, spazio: ~3e8 m/s).
    """
    # accesso diretto ai grafi
    GT = getattr(earth, "terr_graph", None)
    GS = getattr(earth, "space_graph", None)

    # try first terrestrial then space (or vice versa, order doesn't matter: we search both)
    ed = _get_edge(GT, u, v) or _get_edge(GS, u, v)
    if ed is None:
        return None, None

    etype = ed.get("type")
    data_rate = (ed.get("dataRate") or ed.get("data_rate") or ed.get("capacity") or ed.get("dataRateOG"))
    propTime = (ed.get("propDelay") or ed.get("prop_delay"))

    # if there's no propDelay, try to estimate it
    if propTime is None:
        # if we have known distance
        dist_km = ed.get("distance_km")
        slant = ed.get("slant_range") or ed.get("slant_range_km")
        if slant is not None:
            # space
            propTime = (float(slant) * 1000.0) / C_SPACE
        elif dist_km is not None:
            propTime = (float(dist_km) * 1000.0) / C_TER
        else:
            # try to estimate from coordinates of the graph that contained it
            g_used = GT if _get_edge(GT, u, v) is not None else GS
            if g_used is not None:
                nu = g_used.nodes.get(u, {})
                nv = g_used.nodes.get(v, {})
                lon1, lat1 = nu.get("lon"), nu.get("lat")
                lon2, lat2 = nv.get("lon"), nv.get("lat")
                if None not in (lon1, lat1, lon2, lat2):
                    dist_km = haversine(lon1, lat1, lon2, lat2, radius=6371.0088)
                    propTime = (dist_km * 1000.0) / (C_SPACE if etype in ("ISL", "GSL") else C_TER)

    # fallback data rate
    if data_rate is None or data_rate <= 0:
        data_rate = 1e8  # 100 Mbps fallback

    return float(propTime or 0.0), float(data_rate)


###############################################################################
###############################     Classes    ################################
###############################################################################

class Results:
    def __init__(self, finishedBlocks, constellation, GTs, meanTotalLatency, meanQueueLatency, meanTransLatency,
                 meanPropLatency, perQueueLatency, perPropLatency, perTransLatency):
        self.GTs = GTs
        self.finishedBlocks = finishedBlocks
        self.constellation = constellation
        self.meanTotalLatency = meanTotalLatency
        self.meanQueueLatency = meanQueueLatency
        self.meanPropLatency = meanPropLatency
        self.meanTransLatency = meanTransLatency
        self.perQueueLatency = perQueueLatency
        self.perPropLatency = perPropLatency
        self.perTransLatency = perTransLatency


class BlocksForPickle:
    def __init__(self, block):
        self.size = BLOCK_SIZE  # size in bits
        self.ID = block.ID  # a string which holds the source id, destination id, and index of the block, e.g. "1_2_12"
        self.timeAtFull = block.timeAtFull  # the simulation time at which the block was full and was ready to be sent.
        self.creationTime = block.creationTime  # the simulation time at which the block was created.
        self.timeAtFirstTransmission = block.timeAtFirstTransmission  # the simulation time at which the block left the GT.
        self.checkPoints = block.checkPoints  # list of simulation reception times at node with the first entry being the reception time at first sat - can be expanded to include the sat IDs at each checkpoint
        self.checkPointsSend = block.checkPointsSend  # list of times after the block was sent at each node
        self.path = block.path
        self.queueLatency = block.queueLatency  # total time acumulated in the queues
        self.txLatency = block.txLatency  # total transmission time
        self.propLatency = block.propLatency  # total propagation latency
        self.totLatency = block.totLatency  # total latency
        self.QPath = block.QPath  # path followed due to Q-Learning


class UserLink:
    def __init__(self, frequency, bandwidth, maxPtx, aDiameterTx, aDiameterRx,
                 pointingLoss, noiseFigure, noiseTemperature, min_rate):
        self.f = frequency
        self.B = bandwidth
        self.maxPtx = maxPtx
        self.maxPtx_db = 10 * math.log10(maxPtx)
        self.Gtx = 10 * math.log10(eff * ((math.pi * aDiameterTx * frequency / Vc) ** 2))
        self.Grx = 10 * math.log10(eff * ((math.pi * aDiameterRx * frequency / Vc) ** 2))
        self.G = self.Gtx + self.Grx - 2 * pointingLoss
        self.No = 10 * math.log10(bandwidth * k) + noiseFigure + 10 * math.log10(
            290 + (noiseTemperature - 290) * 10 ** (-noiseFigure / 10))
        self.GoT = self.Grx - noiseFigure - 10 * math.log10(
            290 + (noiseTemperature - 290) * 10 ** (-noiseFigure / 10))
        self.min_rate = min_rate

    def __repr__(self):
        return '\n Carrier frequency = {} GHz\n Bandwidth = {} MHz\n Transmission power = {} W\n Gain per antenna: Tx {}  Rx {}\n Total antenna gain = {} dB\n Noise power = {} dBW\n G/T = {} dB/K'.format(
            self.f / 1e9,
            self.B / 1e6,
            self.maxPtx,
            '%.2f' % self.Gtx,
            '%.2f' % self.Grx,
            '%.2f' % self.G,
            '%.2f' % self.No,
            '%.2f' % self.GoT,
        )


class RFlink:
    def __init__(self, frequency, bandwidth, maxPtx, aDiameterTx, aDiameterRx, pointingLoss, noiseFigure,
                 noiseTemperature, min_rate):
        self.f = frequency
        self.B = bandwidth
        self.maxPtx = maxPtx
        self.maxPtx_db = 10 * math.log10(self.maxPtx)
        self.Gtx = 10 * math.log10(eff * ((math.pi * aDiameterTx * self.f / Vc) ** 2))
        self.Grx = 10 * math.log10(eff * ((math.pi * aDiameterRx * self.f / Vc) ** 2))
        self.G = self.Gtx + self.Grx - 2 * pointingLoss
        self.No = 10 * math.log10(self.B * k) + noiseFigure + 10 * math.log10(
            290 + (noiseTemperature - 290) * (10 ** (-noiseFigure / 10)))
        self.GoT = 10 * math.log10(eff * ((math.pi * aDiameterRx * self.f / Vc) ** 2)) - noiseFigure - 10 * math.log10(
            290 + (noiseTemperature - 290) * (10 ** (-noiseFigure / 10)))
        self.min_rate = min_rate

    def __repr__(self):
        return '\n Carrier frequency = {} GHz\n Bandwidth = {} MHz\n Transmission power = {} W\n Gain per antenna: Tx {}  Rx {}\n Total antenna gain = {} dB\n Noise power = {} dBW\n G/T = {} dB/K'.format(
            self.f / 1e9,
            self.B / 1e6,
            self.maxPtx,
            '%.2f' % self.Gtx,
            '%.2f' % self.Grx,
            '%.2f' % self.G,
            '%.2f' % self.No,
            '%.2f' % self.GoT,
        )


class FSOlink:
    def __init__(self, data_rate, power, comm_range, weight):
        self.data_rate = data_rate
        self.power = power
        self.comm_range = comm_range
        self.weight = weight

    def __repr__(self):
        return '\n Data rate = {} Mbps\n Power = {} W\n Transmission range = {} km\n Weight = {} kg'.format(
            self.data_rate / 1e6,
            self.power,
            self.comm_range / 1e3,
            self.weight)


class OrbitalPlane:
    def __init__(self, ID, h, longitude, inclination, n_sat, min_elev, firstID, env, earth):
        self.ID = ID  # A unique ID given to every orbital plane = index in Orbital_planes, string
        self.h = h  # Altitude of deployment
        self.longitude = longitude  # Longitude angle where is intersects equator [radians]
        self.inclination = math.pi / 2 - inclination  # Inclination of the orbit form [radians]
        self.n_sat = n_sat  # Number of satellites in plane
        self.period = 2 * math.pi * math.sqrt(
            (self.h + Re) ** 3 / (G * Me))  # Orbital period of the satellites in seconds
        self.v = 2 * math.pi * (h + Re) / self.period  # Orbital velocity of the satellites in m/s
        self.min_elev = math.radians(min_elev)  # Minimum elevation angle for ground comm.
        self.max_alpha = math.acos(Re * math.cos(self.min_elev) / (
                self.h + Re)) - self.min_elev  # Maximum angle at the center of the Earth w.r.t. yaw
        self.max_beta = math.pi / 2 - self.max_alpha - self.min_elev  # Maximum angle at the satellite w.r.t. yaw
        self.max_distance_2_ground = Re * math.sin(self.max_alpha) / math.sin(
            self.max_beta)  # Maximum distance to a servable ground station
        self.earth = earth

        # Adding satellites
        self.first_sat_ID = firstID  # Unique ID of the first satellite in the orbital plane

        self.sats = []  # List of satellites in the orbital plane
        for i in range(n_sat):
            self.sats.append(
                Satellite(self.first_sat_ID + str(i), int(self.ID), int(i), self.h, self.longitude, self.inclination,
                          self.n_sat, env, self))

        self.last_sat_ID = self.first_sat_ID + str(
            len(self.sats) - 1)  # Unique ID of the last satellite in the orbital plane

    def __repr__(self):
        return '\nID = {}\n altitude= {} km\n longitude= {} deg\n inclination= {} deg\n number of satellites= {}\n period= {} hours\n satellite speed= {} km/s'.format(
            self.ID,
            self.h / 1e3,
            '%.2f' % math.degrees(self.longitude),
            '%.2f' % math.degrees(self.inclination),
            '%.2f' % self.n_sat,
            '%.2f' % (self.period / 3600),
            '%.2f' % (self.v / 1e3))

    def rotate(self, delta_t):
        """
        Rotates the orbit according to the elapsed time by adjusting the longitude. The amount the longitude is adjusted
        is based on the fraction the elapsed time makes up of the time it takes the Earth to complete a full rotation.
        """

        # Change in longitude and phi due to Earth's rotation
        self.longitude = self.longitude + 2 * math.pi * delta_t / Te
        self.longitude = self.longitude % (2 * math.pi)
        # Rotating every satellite in the orbital plane
        for sat in self.sats:
            sat.rotate(delta_t, self.longitude, self.period)


# @profile
class Satellite:
    def __init__(self, ID, in_plane, i_in_plane, h, longitude, inclination, n_sat, env, orbitalPlane, quota=500,
                 power=10):
        self.ID = ID  # A unique ID given to every satellite
        self.orbPlane = orbitalPlane  # Pointer to the orbital plane which the sat belongs to
        self.in_plane = in_plane  # Orbital plane where the satellite is deployed
        self.i_in_plane = i_in_plane  # Index in orbital plane
        self.quota = quota  # Quota of the satellite
        self.h = h  # Altitude of deployment
        self.power = power  # Transmission power
        self.minElevationAngle = minElAngle  # Value is taken from NGSO constellation design chapter

        # Spherical Coordinates before inclination (r,theta,phi)
        self.r = Re + self.h
        self.theta = 2 * math.pi * self.i_in_plane / n_sat
        self.phi = longitude

        # Inclination of the orbital plane
        self.inclination = inclination

        # Cartesian coordinates  (x,y,z)
        self.x = self.r * (
                math.sin(self.theta) * math.cos(self.phi) - math.cos(self.theta) * math.sin(self.phi) * math.sin(
            self.inclination))
        self.y = self.r * (
                math.sin(self.theta) * math.sin(self.phi) + math.cos(self.theta) * math.cos(self.phi) * math.sin(
            self.inclination))
        self.z = self.r * math.cos(self.theta) * math.cos(self.inclination)

        self.polar_angle = self.theta  # Angle within orbital plane [radians]
        self.latitude = math.asin(self.z / self.r)  # latitude corresponding to the satellite
        # longitude corresponding to satellite
        if self.x > 0:
            self.longitude = math.atan(self.y / self.x)
        elif self.x < 0 and self.y >= 0:
            self.longitude = math.pi + math.atan(self.y / self.x)
        elif self.x < 0 and self.y < 0:
            self.longitude = math.atan(self.y / self.x) - math.pi
        elif self.y > 0:
            self.longitude = math.pi / 2
        elif self.y < 0:
            self.longitude = -math.pi / 2
        else:
            self.longitude = 0

        self.waiting_list = {}
        self.applications = []
        self.n_sat = n_sat

        self.ngeo2gt = RFlink(f, B, maxPtx, Adtx, Adrx, pL, Nf, Tn, min_rate)
        self.downRate = 0

        # simpy
        self.env = env
        self.sendBufferGT = ([env.event()], [])  # ([self.env.event()], [DataBlock(0, 0, "0", 0)])
        self.sendBlocksGT = []  # env.process(self.sendBlock())  # simpy processes which send the data blocks
        self.sats = []
        self.linkedGT = None
        self.GTDist = None
        # list of data blocks waiting on their propagation delay.
        self.tempBlocks = [[],
                           []]  # This list is used to so the block can have their paths changed when the constellation is moved

        self.intraSats = []
        self.interSats = []
        self.sendBufferSatsIntra = []
        self.sendBufferSatsInter = []
        self.sendBlocksSatsIntra = []
        self.sendBlocksSatsInter = []
        self.newBuffer = [False]

        self.QLearning = None  # Q-learning table that will be updated in case the pathing is 'Q-Learning'
        self.DDQNA = None  # DDQN agent for each satellite. Only used in the online phase
        self.maxSlantRange = self.GetmaxSlantRange()

    def GetmaxSlantRange(self):
        """
        Maximum distance from satellite to edge of coverage area is calculated using the following formula:
        D_max(minElevationAngle, h) = sqrt(Re**2*sin**2(minElevationAngle) + 2*Re*h + h**2) - Re*sin(minElevationAngle)
        This formula is based on the NGSO constellation design chapter page 16.
        """
        eps = math.radians(self.minElevationAngle)

        distance = math.sqrt((Re + self.h) ** 2 - (Re * math.cos(eps)) ** 2) - Re * math.sin(eps)

        return distance

    def __repr__(self):
        return '\nID = {}\n orbital plane= {}, index in plane= {}, h={}\n pos r = {}, pos theta = {},' \
               ' pos phi = {},\n pos x= {}, pos y= {}, pos z= {}\n inclination = {}\n polar angle = {}' \
               '\n latitude = {}\n longitude = {}'.format(
            self.ID,
            self.in_plane,
            self.i_in_plane,
            '%.2f' % self.h,
            '%.2f' % self.r,
            '%.2f' % self.theta,
            '%.2f' % self.phi,
            '%.2f' % self.x,
            '%.2f' % self.y,
            '%.2f' % self.z,
            '%.2f' % math.degrees(self.inclination),
            '%.2f' % math.degrees(self.polar_angle),
            '%.2f' % math.degrees(self.latitude),
            '%.2f' % math.degrees(self.longitude))

    def createReceiveBlockProcess(self, block, propTime):
        """
        Function which starts a receiveBlock process upon receiving a block from a transmitter.
        """
        process = self.env.process(self.receiveBlock(block, propTime))

    def receiveBlock(self, block, propTime):
        """
        Simpy process function:

        This function is used to handle the propagation delay of data blocks. This is done simply by waiting the time
        of the propagation delay and adding the block to the send-buffer afterwards. Since there are multiple buffers,
        this function looks at the next step in the blocks path and adds the block to the correct send-buffer.
        When Q-Learning or Deep learning is used, this function is where the next step in the block's path is found.

        While the transmission delay is handled at the transmitter, the transmitter cannot also wait for the propagation
        delay, otherwise the send-buffer might be overfilled.

        Using this structure, if there are to be implemented limits on the sizes of the "receive-buffer" it could be
        handled by either limiting the amount of these processes that can occur at the same time, or limiting the size
        of the send-buffer.
        """
        # wait for block to fully propagate
        self.tempBlocks[0].append(block)

        yield self.env.timeout(propTime)

        if block.path == -1:
            return

        # KPI: propLatency receive block from sat
        block.propLatency += propTime

        for i, tempBlock in enumerate(self.tempBlocks[0]):
            # Skip if not a DataBlock (could be an Event)
            if not hasattr(tempBlock, 'ID'):
                continue
            if block.ID == tempBlock.ID:
                self.tempBlocks[0].pop(i)
                break

        try:  # ANCHOR Save Queue time csv
            block.queueTime.append(
                (block.checkPointsSend[len(block.checkPointsSend) - 1] - block.checkPoints[len(block.checkPoints) - 1]))
        except IndexError:  # Either it is the first satellite for the datablock or the datablock has no checkpoints appendeds
            # print('Index error')
            pass

        block.checkPoints.append(self.env.now)

        # if QLearning or Deep Q-Learning we:
        # Compute the next hop in the path and add it to the second last position (Last is the destination gateway)
        # we let the (Deep) Q-model choose the next hop and it will be added to the block.QPath as mentioned
        # if the next hop is the linked gateway it will simply not add anything and will let the model work normally
        if ((self.QLearning) or (self.orbPlane.earth.DDQNA is not None) or (self.DDQNA is not None)):
            if len(block.QPath) > 3:  # the block does not come from a gateway
                if self.QLearning:
                    nextHop = self.QLearning.makeAction(block, self, self.orbPlane.earth.gateways[0].graph,
                                                        self.orbPlane.earth, prevSat=(
                            findByID(self.orbPlane.earth, block.QPath[len(block.QPath) - 3][0])))
                elif self.DDQNA:
                    nextHop = self.DDQNA.makeDeepAction(block, self, self.orbPlane.earth.gateways[0].graph,
                                                        self.orbPlane.earth, prevSat=(
                            findByID(self.orbPlane.earth, block.QPath[len(block.QPath) - 3][0])))
                else:
                    nextHop = self.orbPlane.earth.DDQNA.makeDeepAction(block, self,
                                                                       self.orbPlane.earth.gateways[0].graph,
                                                                       self.orbPlane.earth, prevSat=(
                            findByID(self.orbPlane.earth, block.QPath[len(block.QPath) - 3][0])))
            else:
                if self.QLearning:
                    nextHop = self.QLearning.makeAction(block, self, self.orbPlane.earth.gateways[0].graph,
                                                        self.orbPlane.earth)
                elif self.DDQNA:
                    nextHop = self.DDQNA.makeDeepAction(block, self, self.orbPlane.earth.gateways[0].graph,
                                                        self.orbPlane.earth)
                else:
                    nextHop = self.orbPlane.earth.DDQNA.makeDeepAction(block, self,
                                                                       self.orbPlane.earth.gateways[0].graph,
                                                                       self.orbPlane.earth)

            if nextHop != 0:
                block.QPath.insert(len(block.QPath) - 1, nextHop)
                pathPlot = block.QPath.copy()
                pathPlot.pop()
            else:
                pathPlot = block.QPath.copy()

            # If plotPath plots an image for every action taken. Plots 1/10 of blocks. # ANCHOR plot action satellite
            #################################################################
            if self.orbPlane.earth.plotPaths:
                if int(block.ID[len(block.ID) - 1]) == 0:
                    os.makedirs(self.orbPlane.earth.outputPath + '/pictures/', exist_ok=True)  # create output path
                    outputPath = self.orbPlane.earth.outputPath + '/pictures/' + block.ID + '_' + str(
                        len(block.QPath)) + '_'
                    # plotShortestPath(self.orbPlane.earth, pathPlot, outputPath)
                    plotShortestPath(self.orbPlane.earth, pathPlot, outputPath, ID=block.ID, time=block.creationTime)
            #################################################################

            path = block.QPath  # if there is Q-Learning the path will be repalced with the QPath
        else:
            path = block.path  # if there is no Q-Learning we will work with the path as normally

        # get this satellites index in the blocks path

        index = None
        for i, step in enumerate(path):
            if self.ID == _step_key(step):
                index = i
                break

        if index is None:
            # print(f"Satellite {self.ID} not in block path {path}")
            return

        nxt = path[index + 1]
        nxt_id = nxt[0] if isinstance(nxt, (list, tuple)) else nxt

        # check if next step in path is GT (last step in path)
        earth = self.orbPlane.earth
        next_id = _step_key(path[index + 1])

        if next_id in getattr(earth, 'node_by_name', {}):
            if not self.sendBufferGT[0][0].triggered:
                self.sendBufferGT[0][0].succeed()
                self.sendBufferGT[1].append(block)
            else:
                newEvent = self.env.event().succeed()
                self.sendBufferGT[0].append(newEvent)
                self.sendBufferGT[1].append(block)
        else:
            ID = None
            isIntra = False
            for sat in self.intraSats:
                if sat[1].ID == next_id:
                    ID = sat[1].ID
                    isIntra = True
                    break
            if ID is None:
                for sat in self.interSats:
                    if sat[1].ID == next_id:
                        ID = sat[1].ID
                        break

            if ID is not None:
                sendBuffer = None
                # find send-buffer for the satellite
                if isIntra:
                    for buffer in self.sendBufferSatsIntra:
                        if ID == buffer[2]:
                            sendBuffer = buffer
                else:
                    for buffer in self.sendBufferSatsInter:
                        if ID == buffer[2]:
                            sendBuffer = buffer
                # ANCHOR save the queue length that the block found at its next hop
                self.orbPlane.earth.queues.append(len(sendBuffer[1]))
                block.queue.append(len(sendBuffer[1]))

                # add block to buffer
                if not sendBuffer[0][0].triggered:
                    sendBuffer[0][0].succeed()
                    sendBuffer[1].append(block)
                else:
                    newEvent = self.env.event().succeed()
                    sendBuffer[0].append(newEvent)
                    sendBuffer[1].append(block)

            else:
                print("ERROR! Sat {} tried to send block to {} but did not have it in its linked satellite list".format(
                    self.ID, next_id))

    def sendBlock(self, destination, isSat, isIntra=None):
        """
        Simpy process function:

        Sends data blocks that are filled and added to one of the send-buffers, a buffer which consists of a list of
        events and data blocks. Since there are multiple send-buffers, the function finds the correct buffer given
        information regarding the desired destination satellite or GT. The function monitors the send-buffer, and when
        the buffer contains one or more triggered events, the function will calculate the time it will take to send the
        block and trigger an event which notifies a separate process that a block has been sent.

        A process is running this method for each ISL and for the downLink GSL the satellite has. This will usually be
        4 ISL processes and 1 GSL process.
        """

        if isIntra is not None:
            sendBuffer = None
            if isSat:
                if isIntra:
                    for buffer in self.sendBufferSatsIntra:
                        if buffer[2] == destination[1].ID:
                            sendBuffer = buffer
                else:
                    for buffer in self.sendBufferSatsInter:
                        if buffer[2] == destination[1].ID:
                            sendBuffer = buffer
        else:
            sendBuffer = self.sendBufferGT

        while True:
            try:
                yield sendBuffer[0][0]

                # ANCHOR KPI: queueLatency at sat
                sendBuffer[1][0].checkPointsSend.append(self.env.now)

                if isSat:
                    timeToSend = sendBuffer[1][0].size / destination[2]

                    propTime = self.timeToSend(destination)
                    yield self.env.timeout(timeToSend)

                    receiver = destination[1]

                else:
                    propTime = self.timeToSend(self.linkedGT.linkedSat)
                    timeToSend = sendBuffer[1][0].size / self.downRate
                    yield self.env.timeout(timeToSend)

                    receiver = self.linkedGT

                # When the constellations move, the only case where this process can simply continue, is when the
                # receiver is the same, and there is a block already ready to be sent. The only place where the process
                # can continue from, is as a result right here. Furthermore, the only processes this can happen for are
                # the inter-ISL processes.
                # Due to having to remake buffers when the satellites move, it is necessary for the process to "find"
                # the correct buffer again - the process uses a reference to the buffer: "sendBuffer".
                # To avoid remaking the reference every time a block is sent, the list of boolean values: self.newBuffer
                # is used to indicate when the constellation is moved,

                if True in self.newBuffer and not isIntra and isSat:  # remake reference to buffer
                    if isIntra is not None:
                        sendBuffer = None
                        if isSat:
                            if isIntra:
                                for buffer in self.sendBufferSatsIntra:
                                    if buffer[2] == destination[1].ID:
                                        sendBuffer = buffer
                            else:
                                for buffer in self.sendBufferSatsInter:
                                    if buffer[2] == destination[1].ID:
                                        sendBuffer = buffer
                    else:
                        sendBuffer = self.sendBufferGT

                    for index, val in enumerate(self.newBuffer):
                        if val:  # each process will one by one remake their reference, and change one value to True.
                            # After all processes has done this, all values are back to False
                            self.newBuffer[index] = False
                            break

                # ANCHOR KPI: txLatency ISL
                if sendBuffer[1] and len(sendBuffer[1]) > 0:  # Check if buffer has blocks
                    sendBuffer[1][0].txLatency += timeToSend
                    receiver.createReceiveBlockProcess(sendBuffer[1][0], propTime)
                else:
                    # Buffer is empty, skip this transmission
                    continue

                # remove from own buffer
                if sendBuffer[1] and len(sendBuffer[1]) > 0:  # Only remove if buffer has blocks
                    if len(sendBuffer[0]) == 1:
                        sendBuffer[0].pop(0)
                        sendBuffer[1].pop(0)
                        sendBuffer[0].append(self.env.event())
                    else:
                        sendBuffer[0].pop(0)
                        sendBuffer[1].pop(0)
            except simpy.Interrupt:
                # print(f'Simpy interrupt at sending block at satellite {self.ID} to {destination[1].ID}') # FIXME Are they really lost blocks?
                # self.orbPlane.earth.lostBlocks+=1
                break

    def adjustDownRate(self):

        speff_thresholds = np.array(
            [0, 0.434841, 0.490243, 0.567805, 0.656448, 0.789412, 0.889135, 0.988858, 1.088581, 1.188304, 1.322253,
             1.487473, 1.587196, 1.647211, 1.713601, 1.779991, 1.972253, 2.10485, 2.193247, 2.370043, 2.458441,
             2.524739, 2.635236, 2.637201, 2.745734, 2.856231, 2.966728, 3.077225, 3.165623, 3.289502, 3.300184,
             3.510192, 3.620536, 3.703295, 3.841226, 3.951571, 4.206428, 4.338659, 4.603122, 4.735354, 4.933701,
             5.06569, 5.241514, 5.417338, 5.593162, 5.768987, 5.900855])
        lin_thresholds = np.array(
            [1e-10, 0.5188000389, 0.5821032178, 0.6266138647, 0.751622894, 0.9332543008, 1.051961874, 1.258925412,
             1.396368361, 1.671090614, 2.041737945, 2.529297996, 2.937649652, 2.971666032, 3.25836701, 3.548133892,
             3.953666201, 4.518559444, 4.83058802, 5.508076964, 6.45654229, 6.886522963, 6.966265141, 7.888601176,
             8.452788452, 9.354056741, 10.49542429, 11.61448614, 12.67651866, 12.88249552, 14.48771854, 14.96235656,
             16.48162392, 18.74994508, 20.18366364, 23.1206479, 25.00345362, 30.26913428, 35.2370871, 38.63669771,
             45.18559444, 49.88844875, 52.96634439, 64.5654229, 72.27698036, 76.55966069, 90.57326009])
        db_thresholds = np.array(
            [-100.00000, -2.85000, -2.35000, -2.03000, -1.24000, -0.30000, 0.22000, 1.00000, 1.45000, 2.23000, 3.10000,
             4.03000, 4.68000, 4.73000, 5.13000, 5.50000, 5.97000, 6.55000, 6.84000, 7.41000, 8.10000, 8.38000, 8.43000,
             8.97000, 9.27000, 9.71000, 10.21000, 10.65000, 11.03000, 11.10000, 11.61000, 11.75000, 12.17000, 12.73000,
             13.05000, 13.64000, 13.98000, 14.81000, 15.47000, 15.87000, 16.55000, 16.98000, 17.24000, 18.10000,
             18.59000, 18.84000, 19.57000])

        pathLoss = 10 * np.log10((4 * math.pi * self.linkedGT.linkedSat[0] * self.ngeo2gt.f / Vc) ** 2)
        snr = 10 ** ((self.ngeo2gt.maxPtx_db + self.ngeo2gt.G - pathLoss - self.ngeo2gt.No) / 10)
        shannonRate = self.ngeo2gt.B * np.log2(1 + snr)

        feasible_speffs = speff_thresholds[np.nonzero(lin_thresholds <= snr)]
        speff = self.ngeo2gt.B * feasible_speffs[-1]

        self.downRate = speff

    def timeToSend(self, linkedSat):
        """
        Calculates the propagation time of a block going from satellite to satellite.
        """
        distance = linkedSat[0]
        pTime = distance / Vc
        return pTime

    def findIntraNeighbours(self, earth):
        '''
        Finds intra-plane neighbours
        '''
        self.linked = None  # Closest sat linked
        self.upper = earth.LEO[self.in_plane].sats[self.i_in_plane - 1]  # Previous sat in the same plane
        if self.i_in_plane < self.n_sat - 1:
            self.lower = earth.LEO[self.in_plane].sats[self.i_in_plane + 1]  # Following sat in the same plane
        else:
            self.lower = earth.LEO[self.in_plane].sats[0]  # last satellite of the plane

    def findInterNeighbours(self, earth):
        '''
        Sets the inter plane neighbors for each satellite that will be used for DRL
        '''
        g = earth.graph
        self.right = None
        self.left = None
        # Find inter-plane neighbours (right and left)
        for edge in list(g.edges(self.ID)):
            if edge[1][0].isdigit():
                satB = findByID(earth, edge[1])
                dir = getDirection(self, satB)
                if (dir == 3):  # Found Satellite at East
                    # if self.right is not None:
                    #     print(f"{self.ID} east satellite duplicated! Replacing {self.right.ID} with {satB.ID}.")
                    self.right = satB

                elif (dir == 4):  # Found Satellite at West
                    # if self.left is not None:
                    #     print(f"{self.ID} west satellite duplicated! Replacing {self.left.ID} with {satB.ID}.")
                    self.left = satB
                elif (dir == 1 or dir == 2):
                    pass
                else:
                    print(f'Sat: {satB.ID} direction not found with respect to {self.ID}')
            else:  # it is a GT
                pass

    def rotate(self, delta_t, longitude, period):
        """
        Rotates the satellite by re-calculating the sperical coordinates, Cartesian coordinates, and longitude and
        latitude adjusted for the new longitude of the orbit, and fraction the elapsed time makes up of the orbit time
        of the satellite.
        """
        # Updating spherical coordinates upon rotation (these are phi, theta before inclination)
        self.phi = longitude
        self.theta = self.theta + 2 * math.pi * delta_t / period
        self.theta = self.theta % (2 * math.pi)

        # Calculating x,y,z coordinates with inclination
        self.x = self.r * (
                math.sin(self.theta) * math.cos(self.phi) - math.cos(self.theta) * math.sin(self.phi) * math.sin(
            self.inclination))
        self.y = self.r * (
                math.sin(self.theta) * math.sin(self.phi) + math.cos(self.theta) * math.cos(self.phi) * math.sin(
            self.inclination))
        self.z = self.r * math.cos(self.theta) * math.cos(self.inclination)
        self.polar_angle = self.theta  # Angle within orbital plane [radians]
        # updating latitude and longitude after rotation [degrees]
        self.latitude = math.asin(self.z / self.r)  # latitude corresponding to the satellite
        # longitude corresponding to satellite
        if self.x > 0:
            self.longitude = math.atan(self.y / self.x)
        elif self.x < 0 and self.y >= 0:
            self.longitude = math.pi + math.atan(self.y / self.x)
        elif self.x < 0 and self.y < 0:
            self.longitude = math.atan(self.y / self.x) - math.pi
        elif self.y > 0:
            self.longitude = math.pi / 2
        elif self.y < 0:
            self.longitude = -math.pi / 2
        else:
            self.longitude = 0


class edge:
    def __init__(self, sati, satj, slant_range, dji, dij, shannonRate):
        '''
        dji && dij are deprecated. We do not use them anymore to decide which neighbour is at the right or left direction. We are using their coordinates.
        It is used in the markovian matching only
        '''
        self.i = sati  # sati ID
        self.j = satj  # satj ID
        self.slant_range = slant_range  # distance between both sats
        self.dji = dji  # direction from sati to satj
        self.dij = dij  # direction from sati to satj
        self.shannonRate = shannonRate  # max dataRate between sat1 and satj

    def __repr__(self):
        return '\n node i: {}, node j: {}, slant_range: {}, shannonRate: {}'.format(
            self.i,
            self.j,
            self.slant_range,
            self.shannonRate)

    def __cmp__(self, other):
        if hasattr(other, 'slant_range'):  # returns true if has 'weight' attribute
            return self.slant_range.__cmp__(other.slant_range)


class DataBlock:
    """
    Unità di traffico aggregato. Tiene traccia di tempi di coda, TX e propagazione.
    """

    def __init__(self, source, destination, ID, creationTime):
        self.size = BLOCK_SIZE  # bits
        self.destination = destination
        self.source = source
        self.ID = ID
        self.creationTime = creationTime

        # tempi/checkpoint
        self.timeAtFull = None
        self.timeAtFirstTransmission = None
        self.checkPoints = []  # arrivo ai nodi
        self.checkPointsSend = []  # istante di inizio TX ai nodi (compat vecchia)

        # path
        self.path = []
        self.isNewPath = False
        self.oldPath = []
        self.newPath = []
        self.QPath = []

        # KPI
        self.queueLatency = (None, None)  # compat vecchia: (tot, [per-hop])
        self.txLatency = 0.0
        self.propLatency = 0.0
        self.totLatency = 0.0

        # coda: marcatori per-hop
        self._queue_enq_times = []  # istanti di ingresso in coda
        self._queue_deq_times = []  # istanti di inizio trasmissione

        # RL (se usi)
        self.queue = []
        self.queueTime = []
        self.oldState = None
        self.oldAction = None

    # ---------- queue helpers ----------
    def mark_enqueued(self, t):
        self._queue_enq_times.append(float(t))

    def mark_dequeued(self, t):
        self._queue_deq_times.append(float(t))

    def queue_times(self):
        n = min(len(self._queue_enq_times), len(self._queue_deq_times))
        return [(self._queue_enq_times[i], self._queue_deq_times[i]) for i in range(n)]

    def queue_latencies(self):
        return [deq - enq for (enq, deq) in self.queue_times() if deq >= enq]

    # ---------- compat vecchia ----------
    def getQueueTime(self):
        """
        Compat con plotting vecchio: ritorna (tot_que, [per-hop]).
        Se hai usato i nuovi marcatori, ricostruisco dai queue_times().
        """
        per_hop = self.queue_latencies()
        if per_hop:
            tot = sum(per_hop)
            self.queueLatency = (tot, per_hop)
            return self.queueLatency

        # fallback: se sei nel flusso originale
        queueLatency = [0, []]
        if self.timeAtFirstTransmission is not None:
            first = self.timeAtFirstTransmission - self.creationTime
            queueLatency[0] += first
            queueLatency[1].append(first)
        for arrived, sendReady in zip(self.checkPoints, self.checkPointsSend):
            dt = sendReady - arrived
            queueLatency[0] += dt
            queueLatency[1].append(dt)
        self.queueLatency = queueLatency
        return queueLatency

    def getTotalTransmissionTime(self):
        totalTime = 0
        if len(self.checkPoints) == 1:
            totalTime = self.checkPoints[0] - self.timeAtFirstTransmission
        else:
            lastTime = self.creationTime
            for t in self.checkPoints:
                totalTime += t - lastTime
                lastTime = t
        self.totLatency = totalTime
        return totalTime

    def __repr__(self):
        return f'ID = {self.ID}\n Source:\n {self.source}\n Destination:\n {self.destination}\nTotal latency: {self.totLatency}'


class TerrestrialNode:
    """
    Terrestrial node: source/destination and transit.
    Can generate traffic with the "cell-based" model of original Gateways
    (without becoming a gateway and without touching cell.gateway).
    """

    def __init__(self, name: str, ID: int, latitude: float, longitude: float,
                 totalX: int, totalY: int, totalNodes, env, totalLocations, earth, graph=None):
        self.name = name  # can be int or str
        self.ID = ID  # matches the graph key (int/str)
        self.earth = earth
        self.latitude = latitude
        self.longitude = longitude
        self.totalNodes = totalNodes
        self.totalX = totalX
        self.totalY = totalY

        # --- robust totalLocations handling
        if totalLocations is None:
            tl = []
        elif isinstance(totalLocations, pd.Series):
            tl = [x for x in totalLocations.tolist() if pd.notna(x)]
        elif isinstance(totalLocations, np.ndarray):
            tl = [x for x in totalLocations.tolist() if x is not None]
        else:
            tl = list(totalLocations)
        self.totalLocations = tl
        self.peer_nodes = list(self.totalLocations)

        # Grid position (for ring-scan cells)
        self.gridLocationX = int((0.5 + longitude / 360) * totalX)
        self.gridLocationY = int((0.5 - latitude / 180) * totalY)

        # Cartesian coordinates
        self.polar_angle = (math.pi / 2 - math.radians(latitude) + 2 * math.pi) % (2 * math.pi)
        self.x = Re * math.cos(math.radians(longitude)) * math.sin(self.polar_angle)
        self.y = Re * math.sin(math.radians(longitude)) * math.sin(self.polar_angle)
        self.z = Re * math.cos(self.polar_angle)

        # Network structure
        self.graph = graph  # typically G (TopoHub) or combined

        self.connectedUsers = []

        self.cellsInRange = []
        self.totalAvgFlow = 0.0

        # SimPy
        self.env = env
        self.datBlocks = []
        self.fillBlocks = []
        self.sendBlocks = env.process(self.sendBlock())
        self.sendBuffer = ([env.event()], [])  # ([events], [blocks])
        self.paths = {}  # dest.name -> path (etichette int/str)
        self.dataRate = 1e9  # 1 Gbps default

        # Link “utente↔nodo” (lasciato com’era)
        self.usr2node = UserLink(
            frequency=3.5e9,
            bandwidth=20e6,
            maxPtx=0.1,
            aDiameterTx=0.05,
            aDiameterRx=0.2,
            pointingLoss=2.0,
            noiseFigure=7,
            noiseTemperature=300,
            min_rate=1e6
        )

    def _resolve_node(self, key):
        earth = getattr(self, "earth", None)
        if earth is None:
            return None
        if hasattr(key, "name") or hasattr(key, "ID"):
            return key
        if isinstance(key, int):  # satellite
            return earth.sat_by_id.get(key, None)
        if isinstance(key, str):  # terrestrial/gateway
            node = earth.sat_by_id.get(str(key))
            if node is None:
                node = earth.node_by_name.get(key)
            return node
        return None

    def _drop_head_from_sendbuffer(self):
        try:
            self.sendBuffer[0].pop(0)
            self.sendBuffer[1].pop(0)
        except Exception:
            pass
        finally:
            self.sendBuffer[0].append(self.env.event())

    def cellDistance(self, cell) -> float:
        """Distanza geodesica (km) centro-cella ↔ nodo."""
        cellCoord = (math.degrees(cell.latitude), math.degrees(cell.longitude))
        nCoord = (self.latitude, self.longitude)
        return geopy.distance.geodesic(cellCoord, nCoord).km

    def clearCells(self):
        self.cellsInRange = []

    def findCellsWithinRange(self, earth, maxDistance_km: float, assign_to_cell=False):
        """
        Scan “ad anelli” come nel Gateway.
        Se assign_to_cell=True, scrive anche cell.gateway (di solito False qui).
        Compila self.cellsInRange con: [(lat_deg, lon_deg), users, distanza_km].
        """
        self.cellsInRange = []

        def _consider_cell(x, y):
            cell = earth.cells[x][y]
            d = self.cellDistance(cell)
            if d <= maxDistance_km:
                if assign_to_cell:
                    if cell.gateway is None or (cell.gateway is not None and d < cell.gateway[1]):
                        cell.gateway = (self, d)
                self.cellsInRange.append([(math.degrees(cell.latitude),
                                           math.degrees(cell.longitude)), cell.users, d])
                return True
            return False

        # Up right:
        isWithinRangeX = True
        x = self.gridLocationX
        while isWithinRangeX:
            y = self.gridLocationY
            isWithinRangeY = True
            if x == earth.total_x:
                x = 0
            if not _consider_cell(x, y):
                isWithinRangeY = False
                isWithinRangeX = False
            while isWithinRangeY:
                if y == -1:
                    y = earth.total_y - 1
                if not _consider_cell(x, y):
                    isWithinRangeY = False
                y -= 1
            x += 1

        # Down right:
        isWithinRangeX = True
        x = self.gridLocationX
        while isWithinRangeX:
            y = self.gridLocationY + 1
            isWithinRangeY = True
            if x == earth.total_x:
                x = 0
            if not _consider_cell(x, y):
                isWithinRangeY = False
                isWithinRangeX = False
            while isWithinRangeY:
                if y == earth.total_y:
                    y = 0
                if not _consider_cell(x, y):
                    isWithinRangeY = False
                y += 1
            x += 1

        # Up left:
        isWithinRangeX = True
        x = self.gridLocationX - 1
        while isWithinRangeX:
            y = self.gridLocationY
            isWithinRangeY = True
            if x == -1:
                x = earth.total_x - 1
            if not _consider_cell(x, y):
                isWithinRangeY = False
                isWithinRangeX = False
            while isWithinRangeY:
                if y == -1:
                    y = earth.total_y - 1
                if not _consider_cell(x, y):
                    isWithinRangeY = False
                y -= 1
            x -= 1

        # Down left:
        isWithinRangeX = True
        x = self.gridLocationX - 1
        while isWithinRangeX:
            y = self.gridLocationY + 1
            isWithinRangeY = True
            if x == -1:
                x = earth.total_x - 1
            if not _consider_cell(x, y):
                isWithinRangeY = False
                isWithinRangeX = False
            while isWithinRangeY:
                if y == earth.total_y:
                    y = 0
                if not _consider_cell(x, y):
                    isWithinRangeY = False
                y += 1
            x -= 1

    def addCell(self, cellInfo):
        self.cellsInRange.append(cellInfo)

    def removeCell(self, cell):
        for i, cellInfo in enumerate(self.cellsInRange):
            if cell.latitude == cellInfo[0][0] and cell.longitude == cellInfo[0][1]:
                cellInfo.pop(i)
                return True
        return False

    def getTotalFlow_gateway_style(self, distanceFunc="Step", maxDistance_km=30, capacity=1e12, fraction=1.0,
                                   avgFlowPerUser=None, override_capacity=None):

        # 1) Make sure to have cells in range
        if not self.cellsInRange:
            self.findCellsWithinRange(self.earth, maxDistance_km, assign_to_cell=False)

        if avgFlowPerUser is None:
            try:
                flow_per_user = avUserLoad  # se definito globalmente come nel progetto originale
            except NameError:
                flow_per_user = 5e6  # 5 Mbps fallback
        else:
            flow_per_user = float(avgFlowPerUser)

        totalAvgFlow = 0.0
        total_users = 0
        if distanceFunc == "Step":
            for cell in self.cellsInRange:
                # cell = [(lat,lon), users, distance_km]
                totalAvgFlow += cell[1] * flow_per_user
                total_users += cell[1]
        elif distanceFunc == "Slope":
            gradient = (0.0 - flow_per_user) / (maxDistance_km - 0.0)
            for cell in self.cellsInRange:
                totalAvgFlow += (gradient * cell[2] + flow_per_user) * cell[1]
                total_users += cell[1]
        else:
            print(f"[WARN] distanceFunc={distanceFunc} non riconosciuta, uso 'Step'.")
            for cell in self.cellsInRange:
                totalAvgFlow += cell[1] * flow_per_user
                total_users += cell[1]

        # 4) capacity: min(backhaul, access via UserLink) if not provided
        if capacity is None:
            cap_backhaul = self._capacity_from_backhaul()
            cap_access = self._capacity_from_userlink(percentile=0.75)  # use 0.5 for median, 0.9 more conservative
            capacity = min(cap_backhaul, cap_access)
            # print(f"[DEBUG CAPACITY] {self.name}: backhaul={cap_backhaul/1e6:.1f} Mbps, access={cap_access/1e6:.1f} Mbps, final={capacity/1e6:.1f} Mbps")
            if capacity <= 0:
                capacity = 1e12  # fallback 1 Tbps (increased from 1 Gbps)

        # 5) clamp finale e set degli attributi
        self.totalAvgFlow = min(totalAvgFlow, capacity * fraction)
        self.dataRate = capacity  # utile per timeToSend/diagnostica

        # stampa stile Gateway: valore in miliardi (Gbit/s)
        try:
            print(f"{self.name}: {self.totalAvgFlow / 1e9:.7f}")
        except Exception:
            pass

        return self.totalAvgFlow

    def _capacity_from_backhaul(self, agg="sum"):
        """
        Stima capacità backhaul verso la rete terrestre.
        Legge i dataRate degli archi uscenti dal grafo terrestre.
        agg: "sum" (default) per sommare i link paralleli, "min" per il collo-di-bottiglia.
        """
        G = getattr(getattr(self, "earth", None), "terr_graph", None)
        if G is None or self.name not in G:
            return 1e9  # fallback 1 Gbps

        rates = []
        for nb in G.neighbors(self.name):
            ed = G.get_edge_data(self.name, nb) or G.get_edge_data(nb, self.name)
            if ed is None:
                continue
            # se MultiGraph, prendi il primo dict di attributi
            if isinstance(ed, dict) and ed and all(isinstance(v, dict) for v in ed.values()):
                ed = next(iter(ed.values()))
            r = ed.get("dataRate") or ed.get("data_rate") or ed.get("capacity")
            if r and r > 0:
                rates.append(float(r))

        if not rates:
            return 1e9

        if agg == "min":
            return min(rates)
        else:
            return sum(rates)

    def _capacity_from_userlink(self, percentile=0.75):
        """
        Stima una capacità lato accesso usando UserLink e una distanza utente tipica.
        Per semplicità prende la distanza p-percentile delle celle nel raggio
        e calcola una sola portante tipo Shannon per quel pathloss.
        """
        if not getattr(self, "cellsInRange", None):
            return 1e9

        dists = [c[2] for c in self.cellsInRange if len(c) >= 3]
        if not dists:
            return 1e9

        d = np.percentile(dists, percentile * 100.0)

        # Free-space path loss at distance d (km) - improved for real distances
        f = self.usr2node.f
        B = self.usr2node.B

        # More realistic path loss for urban networks (not pure free-space)
        # Use a more realistic model with reduced attenuation
        pathLoss_db = 20 * np.log10(4 * math.pi * (d * 1000) * f / Vc)

        # Riduci il path loss per reti urbane reali (fattore di correzione)
        pathLoss_db = pathLoss_db * 0.7  # 30% riduzione per ambiente urbano

        snr_lin = 10 ** ((self.usr2node.maxPtx_db + self.usr2node.G - pathLoss_db - self.usr2node.No) / 10.0)
        shannon = B * np.log2(1.0 + snr_lin)

        # Evita valori assurdi/negativi
        if not np.isfinite(shannon) or shannon <= 0:
            shannon = self.usr2node.min_rate

        # More realistic reuse factor for modern networks
        reuse = 0.8  # Increased from 0.5 to 0.8
        shannon_improved = shannon * reuse

        # Additional correction factor for dense urban networks
        urban_factor = 10.0  # 10x improvement for urban networks

        return max(self.usr2node.min_rate, shannon_improved * urban_factor)

    def setup_coverage_and_flow(self,
                                earth,
                                maxDistance_km=30,
                                distanceFunc="Step",
                                capacity=None,
                                fraction=1.0,
                                avgFlowPerUser=None,
                                clear=True):
        """Calcola celle nel raggio e poi totalAvgFlow (gateway-style)."""
        if clear:
            self.clearCells()
        self.findCellsWithinRange(earth, maxDistance_km, assign_to_cell=False)
        return self.getTotalFlow_gateway_style(distanceFunc=distanceFunc,
                                               maxDistance_km=maxDistance_km,
                                               capacity=capacity,
                                               fraction=fraction,
                                               avgFlowPerUser=avgFlowPerUser)

    # ------------------------------------------------------------
    # traffic (fillBlock) — usa totalAvgFlow gateway-style
    # ------------------------------------------------------------

    def makeFillBlockProcesses(self, terrestrial_nodes=None, target_nodes=None):
        if target_nodes is not None:
            allowed = [n for n in target_nodes if n is not self]
        elif terrestrial_nodes is not None:
            allowed = [n for n in terrestrial_nodes if n is not self]
        else:
            allowed = []
        if not allowed:
            return
        self.totalNodes = len(allowed)
        self.peer_nodes = list(allowed)
        self.fillBlocks = getattr(self, "fillBlocks", [])
        for dest in allowed:
            self.fillBlocks.append(self.env.process(self.fillBlock(dest)))

    def fillBlock(self, destination):
        index = 0
        unavailableDestinationBuffer = []
        while True:
            try:
                graph = getattr(self, 'graph', None)
                src_keys = [getattr(self, 'name', None), getattr(self, 'ID', None)]
                dst_keys = [getattr(destination, 'name', None), getattr(destination, 'ID', None)]
                src_ok = graph is not None and any(k is not None and graph.has_node(k) for k in src_keys)
                dst_ok = graph is not None and any(k is not None and graph.has_node(k) for k in dst_keys)
                if not (src_ok and dst_ok):
                    print(
                        f"Cannot create DataBlock: source {getattr(self, 'name', self)} or destination {getattr(destination, 'name', destination)} not in graph")
                    return
                block = DataBlock(self, destination, f"{self.ID}_{destination.ID}_{index}", self.env.now)
                timeToFull = self.timeToFullBlock(block)
                yield self.env.timeout(timeToFull)

                if destination.name not in self.paths or not self.paths[destination.name]:
                    unavailableDestinationBuffer.append(block)
                else:
                    # svuota eventuali blocchi in attesa di path
                    while unavailableDestinationBuffer:
                        if not self.sendBuffer[0][0].triggered:
                            self.sendBuffer[0][0].succeed()
                        else:
                            self.sendBuffer[0].append(self.env.event().succeed())
                        self.sendBuffer[1].append(unavailableDestinationBuffer.pop(0))

                    # assegna path al blocco
                    block.path = self.paths[destination.name]
                    if self.earth.pathParam in ['Q-Learning', 'Deep Q-Learning']:
                        block.QPath = [block.path[0], block.path[1], block.path[-1]]

                    block.timeAtFull = self.env.now
                    createdBlocks.append(block)

                    # *** QUEUE: entra in coda al sorgente ***
                    block.mark_enqueued(self.env.now)

                    if not self.sendBuffer[0][0].triggered:
                        self.sendBuffer[0][0].succeed()
                    else:
                        self.sendBuffer[0].append(self.env.event().succeed())
                    self.sendBuffer[1].append(block)
                    index += 1

            except simpy.Interrupt:
                print('Simpy interrupt during fillBlock at node {}'.format(self.name))
                break

    # ------------------------------------------------------------
    # transmission
    # ------------------------------------------------------------

    def sendBlock(self):
        """
        Processo di invio per i blocchi in coda del TerrestrialNode.
        - Normalizza sempre il path (per evitare chiavi non hashable).
        - Calcola tempo di TX dall'attributo dataRate del link (fallback 100 Mbps se mancante).
        - Somma la propagation delay del link.
        - Aggiorna le metriche del DataBlock (queue/tx/prop) in modo compatibile coi plot esistenti.
        """
        while True:
            # wait for event "there's something to send"
            yield self.sendBuffer[0][0]
            if not self.sendBuffer[1]:
                self._drop_head_from_sendbuffer()
                continue

            block = self.sendBuffer[1][0]

            p = coerce_path_only(getattr(block, "path", None))
            if not p or len(p) < 2:
                try:
                    path_log = coerce_path_only(list(getattr(block, 'path', [])))
                    safe_print("[ERROR] Invalid/missing path from {} to {}: {}".format(
                        self.name,
                        getattr(getattr(block, 'destination', None), 'name', '?'),
                        path_log
                    ))
                except Exception:
                    print("[ERROR] Invalid/missing path.")
                self._drop_head_from_sendbuffer()
                continue

            # trova il mio indice nel path e il prossimo hop
            my_keys = {str(getattr(self, "name", "")), str(getattr(self, "ID", ""))}
            cur_idx = None
            for i, k in enumerate(p):
                if str(k) in my_keys:
                    cur_idx = i
                    break

            if cur_idx is None or cur_idx + 1 >= len(p):
                # print("[ERROR] Current node {} not in path or last-hop inconsistency: {}".format(self.name, p))
                self._drop_head_from_sendbuffer()
                continue

            nxt_key = p[cur_idx + 1]
            next_hop_node = self._resolve_node(nxt_key)
            if next_hop_node is None:
                print("[ERROR] Next hop not found for {} -> {} (key={})".format(self.name, p, nxt_key))
                self._drop_head_from_sendbuffer()
                continue

            # parametri link (propDelay e dataRate)
            u_key = str(getattr(self, "name", getattr(self, "ID", None)))
            v_key = str(getattr(next_hop_node, "name", getattr(next_hop_node, "ID", None)))
            propTime, data_rate = edge_latency_and_rate(self.earth, u_key, v_key)

            if data_rate is None or data_rate <= 0:
                dst_name = getattr(next_hop_node, "name", getattr(next_hop_node, "ID", "?"))
                print("[ERROR] Missing/invalid data rate {} on edge {} -> {}".format(data_rate, self.name, dst_name))
                self._drop_head_from_sendbuffer()
                continue

            # --- QUEUE: esce dalla coda e inizia la trasmissione
            if hasattr(block, "mark_dequeued"):
                # se esiste, segna l'uscita dalla coda (compat con i tuoi plot)
                block.mark_dequeued(self.env.now)
            block.checkPointsSend.append(self.env.now)  # historical compatibility

            if block.timeAtFirstTransmission is None:
                block.timeAtFirstTransmission = self.env.now

            # trasmissione (solo tempo di TX, la propagazione la aggiunge il next hop)
            timeToSend = BLOCK_SIZE / data_rate
            yield self.env.timeout(timeToSend)
            block.txLatency = getattr(block, "txLatency", 0.0) + timeToSend

            # invia al prossimo hop aggiungendo la propagation delay su quel link
            next_hop_node.createReceiveBlockProcess(block, propTime if propTime is not None else 0.0)

            # rimuovi dalla testa e ri-arma l'evento
            self._drop_head_from_sendbuffer()

    def timeToSend(self):
        return BLOCK_SIZE / self.dataRate

    # ------------------------------------------------------------
    # receiving
    # ------------------------------------------------------------

    def createReceiveBlockProcess(self, block, propTime=0):
        self.env.process(self.receiveBlock(block, propTime))

    def receiveBlock(self, block, propTime):
        # ritardo di propagazione
        yield self.env.timeout(propTime)
        block.propLatency += propTime
        block.checkPoints.append(self.env.now)

        def _same_node(dest, me_name, me_id):
            cand = {str(dest)}
            if hasattr(dest, "name"):
                cand.add(str(getattr(dest, "name")))
            if hasattr(dest, "ID"):
                cand.add(str(getattr(dest, "ID")))
            my_ids = {str(me_name), str(me_id)}
            return len(my_ids.intersection(cand)) > 0

        # consegna oppure re-queue
        if _same_node(block.destination, self.name, getattr(self, "ID", self.name)):
            receivedDataBlocks.append(block)
        else:
            # *** QUEUE: entra in coda nel nodo intermedio ***
            block.mark_enqueued(self.env.now)

            if not self.sendBuffer[0][0].triggered:
                self.sendBuffer[0][0].succeed()
            else:
                self.sendBuffer[0].append(self.env.event().succeed())
            self.sendBuffer[1].append(block)

    # ------------------------------------------------------------
    # traffic model selection
    # ------------------------------------------------------------

    def timeToFullBlock(self, block):
        """
        Usa SEMPRE self.totalAvgFlow (ottienila con setup_coverage_and_flow / getTotalFlow_gateway_style).
        """
        peers = getattr(self, 'peer_nodes', [])
        n_dests = len(peers)
        if n_dests <= 0 or getattr(self, 'totalAvgFlow', 0) <= 0:
            return 1.0
        flow_per_dest = self.totalAvgFlow / n_dests
        if flow_per_dest <= 0:
            return 1.0
        avgTime = block.size / flow_per_dest
        return np.random.exponential(scale=avgTime)

    # (compat for old model based on "connectedUsers")
    def getTotalFlow(self, avgFlowPerUser=1e6, capacity=1e12, fraction=1.0):
        num_users = len(self.connectedUsers)
        totalAvgFlow = num_users * avgFlowPerUser
        self.totalAvgFlow = min(totalAvgFlow, capacity * fraction)
        self.dataRate = capacity
        if self.totalAvgFlow == 0:
            self.totalAvgFlow = 1e-6


# @profile
class Gateway:
    """
    Class for the gateways (or concentrators). Each gateway will exist as an instance of this class
    which means that each ground station will have separate processes filling and sending blocks to all other GTs.
    """

    def __init__(self, name: str, ID: int, latitude: float, longitude: float, totalX: int, totalY: int, totalGTs, env,
                 totalLocations, earth):
        self.name = name
        self.ID = ID
        self.earth = earth
        self.latitude = latitude  # number is already in degrees
        self.longitude = longitude  # number is already in degrees

        # using the formulas from the set_window() function in the Earth class to the location in terms of cell grid.
        self.gridLocationX = int((0.5 + longitude / 360) * totalX)
        self.gridLocationY = int((0.5 - latitude / 180) * totalY)
        self.cellsInRange = []  # format: [ [(lat,long), userCount, distance], [..], .. ]
        self.totalGTs = totalGTs  # number of GTs including itself
        self.totalLocations = totalLocations  # number of possible GTs
        self.totalAvgFlow = None  # total combined average flow from all users in bits per second
        self.totalX = totalX
        self.totalY = totalY

        # cartesian coordinates
        self.polar_angle = (math.pi / 2 - math.radians(self.latitude) + 2 * math.pi) % (
                2 * math.pi)  # Polar angle in radians
        self.x = Re * math.cos(math.radians(self.longitude)) * math.sin(self.polar_angle)
        self.y = Re * math.sin(math.radians(self.longitude)) * math.sin(self.polar_angle)
        self.z = Re * math.cos(self.polar_angle)

        # satellite linking structure
        self.satsOrdered = []
        self.satIndex = 0
        self.linkedSat = (None, None)  # (distance, sat)
        self.graph = nx.Graph()

        # simpy attributes
        self.env = env  # simulation environment
        self.datBlocks = []  # list of outgoing data blocks - one for each destination GT
        self.fillBlocks = []  # list of simpy processes which fills up the data blocks
        self.sendBlocks = env.process(self.sendBlock())  # simpy process which sends the data blocks
        self.sendBuffer = ([env.event()], [])  # queue of blocks that are ready to be sent
        self.paths = {}  # dictionary for destination: path pairs

        # # --- Buffer/process TERRESTRIAL (new, doesn't touch satellites) ---
        # if not hasattr(self, 't_sendBuffer'):
        #     self.t_sendBuffer = ([self.env.event()], [])  # [list of events], [queue of blocks]
        # if not hasattr(self, 'sendBlocksTerrestrial'):
        #     self.sendBlocksTerrestrial = []
        #     self.sendBlocksTerrestrial.append(
        #         self.env.process(self._sendBlockTerrestrial())
        #     )

        # comm attributes
        self.dataRate = None
        self.gs2ngeo = RFlink(
            frequency=30e9,
            bandwidth=500e6,
            maxPtx=20,
            aDiameterTx=0.33,
            aDiameterRx=0.26,
            pointingLoss=0.3,
            noiseFigure=2,
            noiseTemperature=290,
            min_rate=10e3
        )

    # ---------------------------
    # helpers
    # ---------------------------

    def _resolve_node(self, key):
        """
        Resolve a next hop that could be:
        - int (satellite ID)
        - str (gateway name/terrestrial node)
        - object node
        Use the Earth lookup.
        """
        earth = getattr(self, "earth", None)
        if earth is None:
            return None
        if hasattr(key, "name") or hasattr(key, "ID"):
            return key
        if isinstance(key, int):  # satellite
            return getattr(earth, "sat_by_id", {}).get(key)
        if isinstance(key, str):  # terrestrial/gateway
            return getattr(earth, "node_by_name", {}).get(key)
        return None

    # ---------------------------
    #
    # ---------------------------

    def makeFillBlockProcesses(self, GTs):
        """
        Creates the processes for filling the data blocks and adding them to the send-buffer. A separate process for
        each destination gateway is created.
        """

        self.totalGTs = len(GTs)

        for gt in GTs:
            if gt != self:
                # add a process for each destination which runs the function 'fillBlock'
                self.fillBlocks.append(self.env.process(self.fillBlock(gt)))

    def fillBlock(self, destination):
        """
        Simpy process function:

        Creates a block headed for a given destination, finds the time for a block to be full and adds the block to the
        send-buffer after the calculated time.

        A separate process for each destination gateway will be running this function.
        """
        index = 0
        unavailableDestinationBuffer = []

        while True:
            try:
                graph = getattr(self, 'graph', None)
                src_keys = [getattr(self, 'name', None), getattr(self, 'ID', None)]
                dst_keys = [getattr(destination, 'name', None), getattr(destination, 'ID', None)]
                src_ok = graph is not None and any(k is not None and graph.has_node(k) for k in src_keys)
                dst_ok = graph is not None and any(k is not None and graph.has_node(k) for k in dst_keys)
                if not (src_ok and dst_ok):
                    print(
                        f"Cannot create DataBlock: source {getattr(self, 'name', self)} or destination {getattr(destination, 'name', destination)} not in graph")
                    return
                # create a new block to be filled
                block = DataBlock(self, destination, str(self.ID) + "_" + str(destination.ID) + "_" + str(index),
                                  self.env.now)

                timeToFull = self.timeToFullBlock(block)  # calculate time to fill block

                yield self.env.timeout(timeToFull)  # wait until block is full

                if block.destination.linkedSat[0] is None:
                    unavailableDestinationBuffer.append(block)
                else:
                    while unavailableDestinationBuffer:  # empty buffer before adding new block
                        if not self.sendBuffer[0][0].triggered:
                            self.sendBuffer[0][0].succeed()
                            self.sendBuffer[1].append(unavailableDestinationBuffer[0])
                            unavailableDestinationBuffer.pop(0)
                        else:
                            newEvent = self.env.event().succeed()
                            self.sendBuffer[0].append(newEvent)
                            self.sendBuffer[1].append(unavailableDestinationBuffer[0])
                            unavailableDestinationBuffer.pop(0)

                    block.path = self.paths[destination.name]

                    if self.earth.pathParam == 'Q-Learning' or self.earth.pathParam == 'Deep Q-Learning':
                        block.QPath = [block.path[0], block.path[1], block.path[len(block.path) - 1]]
                        # We add a Qpath field for the Q-Learning case. Only source and destination will be added
                        # after that, every hop will be added at the second last position.

                    if not block.path:
                        print(self.name, destination.name)
                        exit()
                    block.timeAtFull = self.env.now
                    createdBlocks.append(block)
                    # add block to send-buffer
                    if not self.sendBuffer[0][0].triggered:
                        self.sendBuffer[0][0].succeed()
                        self.sendBuffer[1].append(block)
                    else:
                        newEvent = self.env.event().succeed()
                        self.sendBuffer[0].append(newEvent)
                        self.sendBuffer[1].append(block)
                    index += 1
            except simpy.Interrupt:
                print(f'Simpy interrupt at filling block at gateway{self.name}')
                break

    def sendBlock(self):
        """
        Simpy process function:

        Sends data blocks that are filled and added to the send-buffer which is a list of events and data blocks. The
        function monitors the send-buffer, and when the buffer contains one or more triggered events, the function will
        calculate the time it will take to send the block (yet to be implemented), and trigger an event which notifies
        a separate process that a block has been sent (yet to be implemented).

        After a block is sent, the function will send the next, if any more blocks are ready to be sent.

        (While it is assumed that if a buffer is full and ready to be sent it will always be at the first index,
        the method simpy.AnyOf is used. The end result is the same and this method is simple to implement.
        Furthermore, it allows for handling of such errors where a later index is ready but the first is not.
        this case is, however, not handled.)

        Since there is only one link on the GT for sending, there will only be one process running this method.
        """
        # while True:
        #     yield self.sendBuffer[0][0]     # event 0 of block 0
        #
        #     # wait until a satellite is linked
        #     while self.linkedSat[0] is None:
        #         yield self.env.timeout(0.1)
        #
        #     # calculate propagation time and transmission time
        #     propTime = self.timeToSend(self.linkedSat)
        #     timeToSend = BLOCK_SIZE/self.dataRate
        #
        #     self.sendBuffer[1][0].timeAtFirstTransmission = self.env.now
        #     yield self.env.timeout(timeToSend)
        #     # ANCHOR KPI: txLatency send block from GT
        #     self.sendBuffer[1][0].txLatency += timeToSend
        #
        #     if not self.sendBuffer[1][0].path:
        #         print(self.sendBuffer[1][0].source.name, self.sendBuffer[1][0].destination.name)
        #         exit()
        #
        #     self.linkedSat[1].createReceiveBlockProcess(self.sendBuffer[1][0], propTime)
        #
        #     # remove from own sendBuffer
        #     if len(self.sendBuffer[0]) == 1:
        #         self.sendBuffer[0].pop(0)
        #         self.sendBuffer[1].pop(0)
        #         self.sendBuffer[0].append(self.env.event())
        #     else:
        #         self.sendBuffer[0].pop(0)
        #         self.sendBuffer[1].pop(0)
        while True:
            yield self.sendBuffer[0][0]
            block = self.sendBuffer[1][0]

            # 1) Path sanity
            p = getattr(block, "path", None)
            if not p or len(p) < 2:
                print(f"[ERROR] Invalid/missing path from {self.name} to "
                      f"{getattr(getattr(block, 'destination', None), 'name', '?')}: {p}")
                self.sendBuffer[0].pop(0)
                self.sendBuffer[1].pop(0)
                self.sendBuffer[0].append(self.env.event())
                continue

            # helper to extract comparable key from step
            def key_of(x):
                if isinstance(x, (tuple, list)) and x:
                    return x[0]
                return x

            path_keys = [key_of(x) for x in p]

            # 2) find current index in path (match by name or ID)
            cur_idx = None
            me_keys = {getattr(self, 'name', None), getattr(self, 'ID', None)}
            for i, k in enumerate(path_keys):
                if k in me_keys:
                    cur_idx = i
                    break

            if cur_idx is None:
                print(f"[ERROR] Current node {self.name} not in path {path_keys}")
                self.sendBuffer[0].pop(0)
                self.sendBuffer[1].pop(0)
                self.sendBuffer[0].append(self.env.event())
                continue

            if cur_idx + 1 >= len(path_keys):
                print(f"[ERROR] {self.name} is last hop but path not delivered for block to "
                      f"{getattr(block.destination, 'name', '?')}")
                self.sendBuffer[0].pop(0)
                self.sendBuffer[1].pop(0)
                self.sendBuffer[0].append(self.env.event())
                continue

            # 3) resolve next hop
            nxt_key = path_keys[cur_idx + 1]
            next_hop_node = None

            earth = getattr(self, 'earth', None) or getattr(getattr(self, 'orbPlane', None), 'earth', None)

            if isinstance(nxt_key, int) and earth is not None:
                # could be a satellite (int ID)
                next_hop_node = getattr(earth, "sat_by_id", {}).get(nxt_key)

            if next_hop_node is None:
                # try as terrestrial/gateway node (string); also use normalized forms if available
                s = str(nxt_key)
                nb = getattr(earth, "node_by_name", {}) if earth else {}
                next_hop_node = nb.get(s)
                if next_hop_node is None:
                    try:
                        import unicodedata as _ud
                        next_hop_node = nb.get(_ud.normalize('NFC', s)) or nb.get(_ud.normalize('NFKC', s))
                    except Exception:
                        pass

            # fallback: use linked sat if exists
            if next_hop_node is None and getattr(self, "linkedSat", (None, None))[1] is not None:
                next_hop_node = self.linkedSat[1]

            if next_hop_node is None:
                print(f"[ERROR] Next hop not found for {self.name} -> {path_keys}")
                self.sendBuffer[0].pop(0)
                self.sendBuffer[1].pop(0)
                self.sendBuffer[0].append(self.env.event())
                continue

            # 4) prendi prop delay e data rate usando i GRAFI SEPARATI
            #    NB: _edge_latency_and_rate(u, v) si aspetta **chiavi** di grafo (int/str)
            u_key = getattr(self, 'name', getattr(self, 'ID', None))
            v_key = getattr(next_hop_node, 'name', getattr(next_hop_node, 'ID', None))
            propTime, data_rate = edge_latency_and_rate(self.earth, u_key, v_key)

            # se sto andando al sat collegato e non ho rate sul grafo, usa l'uplink calcolato del gateway
            if (data_rate is None or data_rate <= 0) and getattr(self, "linkedSat", (None, None))[1] is next_hop_node:
                if getattr(self, "dataRate", None):
                    data_rate = self.dataRate

            if data_rate is None or data_rate <= 0:
                dst_name = getattr(next_hop_node, "name", getattr(next_hop_node, "ID", "?"))
                print(f"[ERROR] Missing/invalid data rate on edge {self.name} -> {dst_name}")
                self.sendBuffer[0].pop(0)
                self.sendBuffer[1].pop(0)
                self.sendBuffer[0].append(self.env.event())
                continue

            # 5) invio
            timeToSend = BLOCK_SIZE / data_rate
            if not hasattr(block, 'timeAtFirstTransmission') or block.timeAtFirstTransmission is None:
                block.timeAtFirstTransmission = self.env.now

            yield self.env.timeout(timeToSend)
            block.txLatency = getattr(block, "txLatency", 0.0) + timeToSend

            # deliver to next hop (propTime can be None)
            next_hop_node.createReceiveBlockProcess(block, propTime if propTime is not None else 0.0)

            # 6) pulizia buffer
            self.sendBuffer[0].pop(0)
            self.sendBuffer[1].pop(0)
            self.sendBuffer[0].append(self.env.event())

    def timeToSend(self, linkedSat):
        distance = linkedSat[0]
        pTime = distance / Vc
        return pTime

    def createReceiveBlockProcess(self, block, propTime):
        """
        Function which starts a receiveBlock process upon receiving a block from a transmitter.
        Adds the propagation time to the block attribute
        """

        process = self.env.process(self.receiveBlock(block, propTime))

    # def receiveBlock(self, block, propTime):
    #     """
    #     Simpy process function:
    #
    #     This function is used to handle the propagation delay of data blocks. This is done simply by waiting the time
    #     of the propagation delay. As a GT will always be the last step in a block's path, there is no need to send the
    #     block further. After the propagation delay, the block is simply added to a list of finished blocks so the KPIs
    #     can be tracked at the end of the simulation.
    #
    #     While the transmission delay is handled at the transmitter, the transmitter cannot also wait for the propagation
    #     delay, otherwise the send-buffer might be overfilled.
    #     """
    #     # clamp per sicurezza (niente tempi negativi)
    #     if propTime is None or propTime < 0:
    #         propTime = 0
    #
    #     # attesa propagazione
    #     yield self.env.timeout(propTime)
    #
    #     # KPI: propagation latency
    #     if not hasattr(block, 'propLatency') or block.propLatency is None:
    #         block.propLatency = 0.0
    #     block.propLatency += propTime
    #
    #     # checkpoint list always present
    #     if not hasattr(block, 'checkPoints') or block.checkPoints is None:
    #         block.checkPoints = []
    #     block.checkPoints.append(self.env.now)
    #
    #     # optional: small warning if gateway is not the expected destination
    #     try:
    #         dest_name = getattr(block.destination, 'name', None)
    #         if dest_name is not None and dest_name != self.name:
    #             print(f"[WARN] Block destined to {dest_name} received by {self.name}.")
    #     except Exception:
    #         pass
    #
    #     # registra il blocco ricevuto
    #     receivedDataBlocks.append(block)

    def receiveBlock(self, block, propTime):
        """Gateway: può ricevere da terra o dallo spazio e inoltrare ulteriormente."""
        yield self.env.timeout(propTime)
        block.propLatency += propTime
        block.checkPoints.append(self.env.now)

        # if (rarely) the destination is this gateway itself:
        if block.destination == self:
            receivedDataBlocks.append(block)
            return

        # otherwise enqueue in my buffer; my sendBlock will use block.path
        if not self.sendBuffer[0][0].triggered:
            self.sendBuffer[0][0].succeed()
        else:
            self.sendBuffer[0].append(self.env.event().succeed())
        self.sendBuffer[1].append(block)

    def cellDistance(self, cell) -> float:
        """
        Calculates the distance to the specified cell (assumed the center of the cell).
        Calculation is based on the geopy package which uses the 'WGS-84' model for earth shape.
        """
        cellCoord = (math.degrees(cell.latitude),
                     math.degrees(cell.longitude))  # cell lat and long is saved in a format which is not degrees
        gTCoord = (self.latitude, self.longitude)

        return geopy.distance.geodesic(cellCoord, gTCoord).km

    def distance_GSL(self, satellite):
        """
        Distance between GT and satellite is calculated using the distance formula based on the cartesian coordinates
        in 3D space.
        """

        satCoords = [satellite.x, satellite.y, satellite.z]
        GTCoords = [self.x, self.y, self.z]

        distance = math.dist(satCoords, GTCoords)
        return distance

    def adjustDataRate(self):

        speff_thresholds = np.array(
            [0, 0.434841, 0.490243, 0.567805, 0.656448, 0.789412, 0.889135, 0.988858, 1.088581, 1.188304, 1.322253,
             1.487473, 1.587196, 1.647211, 1.713601, 1.779991, 1.972253, 2.10485, 2.193247, 2.370043, 2.458441,
             2.524739, 2.635236, 2.637201, 2.745734, 2.856231, 2.966728, 3.077225, 3.165623, 3.289502, 3.300184,
             3.510192, 3.620536, 3.703295, 3.841226, 3.951571, 4.206428, 4.338659, 4.603122, 4.735354, 4.933701,
             5.06569, 5.241514, 5.417338, 5.593162, 5.768987, 5.900855])
        lin_thresholds = np.array(
            [1e-10, 0.5188000389, 0.5821032178, 0.6266138647, 0.751622894, 0.9332543008, 1.051961874, 1.258925412,
             1.396368361, 1.671090614, 2.041737945, 2.529297996, 2.937649652, 2.971666032, 3.25836701, 3.548133892,
             3.953666201, 4.518559444, 4.83058802, 5.508076964, 6.45654229, 6.886522963, 6.966265141, 7.888601176,
             8.452788452, 9.354056741, 10.49542429, 11.61448614, 12.67651866, 12.88249552, 14.48771854, 14.96235656,
             16.48162392, 18.74994508, 20.18366364, 23.1206479, 25.00345362, 30.26913428, 35.2370871, 38.63669771,
             45.18559444, 49.88844875, 52.96634439, 64.5654229, 72.27698036, 76.55966069, 90.57326009])
        db_thresholds = np.array(
            [-100.00000, -2.85000, -2.35000, -2.03000, -1.24000, -0.30000, 0.22000, 1.00000, 1.45000, 2.23000, 3.10000,
             4.03000, 4.68000, 4.73000, 5.13000, 5.50000, 5.97000, 6.55000, 6.84000, 7.41000, 8.10000, 8.38000, 8.43000,
             8.97000, 9.27000, 9.71000, 10.21000, 10.65000, 11.03000, 11.10000, 11.61000, 11.75000, 12.17000, 12.73000,
             13.05000, 13.64000, 13.98000, 14.81000, 15.47000, 15.87000, 16.55000, 16.98000, 17.24000, 18.10000,
             18.59000, 18.84000, 19.57000])

        pathLoss = 10 * np.log10((4 * math.pi * self.linkedSat[0] * self.gs2ngeo.f / Vc) ** 2)
        snr = 10 ** ((self.gs2ngeo.maxPtx_db + self.gs2ngeo.G - pathLoss - self.gs2ngeo.No) / 10)
        shannonRate = self.gs2ngeo.B * np.log2(1 + snr)

        feasible_speffs = speff_thresholds[np.nonzero(lin_thresholds <= snr)]
        speff = self.gs2ngeo.B * feasible_speffs[-1]

        self.dataRate = speff

    def orderSatsByDist(self, constellation):
        """
        Calculates the distance from the GT to all satellites and saves a sorted (least to greatest distance) list of
        all the satellites that are within range of the GT.
        """
        sats = []
        index = 0
        for orbitalPlane in constellation:
            for sat in orbitalPlane.sats:
                d_GSL = self.distance_GSL(sat)
                # ensure that the satellite is within range
                if d_GSL <= sat.maxSlantRange * 10:  # FIXME this x10 is for small constellations
                    sats.append((d_GSL, sat, [index]))
                index += 1
        sats.sort()
        self.satsOrdered = sats

    def addRefOnSat(self):
        """
        Adds a reference of the GT on a satellite based on the local list of satellites that are within range of the GT.
        This function is used in the greedy version of the 'linkSats2GTs()' method in the Earth class.
        The function uses a local indexing number to choose which satellite to add a reference to. If the satellite
        already has a reference, the GT checks if it is closer than the existing reference. If it is closer, it
        overwrites the reference and forces the other GT to add a reference to the next satellite it its own list.
        """
        if self.satIndex >= len(self.satsOrdered):
            self.linkedSat = (None, None)
            print("No satellite for GT {}".format(self.name))
            return

        # check if satellite has reference
        if self.satsOrdered[self.satIndex][1].linkedGT is None:
            # add self as reference on satellite
            self.satsOrdered[self.satIndex][1].linkedGT = self
            self.satsOrdered[self.satIndex][1].GTDist = self.satsOrdered[self.satIndex][0]

        # check if satellites reference is further away than this GT
        elif self.satsOrdered[self.satIndex][1].GTDist < self.satsOrdered[self.satIndex][0]:
            # force other GT to increment satIndex and check next satellite in its local ordered list
            self.satsOrdered[self.satIndex][1].linkedGT.satIndex += 1
            self.satsOrdered[self.satIndex][1].linkedGT.addRefOnSat()

            # add self as reference on satellite
            self.satsOrdered[self.satIndex][1].linkedGT = self
            self.satsOrdered[self.satIndex][1].GTDist = self.satsOrdered[self.satIndex][0]
        else:
            self.satIndex += 1
            if self.satIndex == len(self.satsOrdered):
                self.linkedSat = (None, None)
                print("No satellite for GT {}".format(self.name))
                return

            self.addRefOnSat()

    def link2Sat(self, dist, sat):
        """
        Links the GT to the satellite chosen in the 'linkSats2GTs()' method in the Earth class and makes sure that the
        data rate for the RFlink to the satellite is updated.
        """
        self.linkedSat = (dist, sat)
        sat.linkedGT = self
        sat.GTDist = dist
        self.adjustDataRate()

    def addCell(self, cellInfo):
        """
        Links a cell to the GT by adding the relevant information of the cell to the local list "cellsInRange".
        """
        self.cellsInRange.append(cellInfo)

    def removeCell(self, cell):
        """
        Unused function
        """
        for i, cellInfo in enumerate(self.cellsInRange):
            if cell.latitude == cellInfo[0][0] and cell.longitude == cellInfo[0][1]:
                cellInfo.pop(i)
                return True
        return False

    def findCellsWithinRange(self, earth, maxDistance):
        """
        This function finds the cells that are within the coverage area of the gateway instance. The cells are
        found by checking cells one at a time from the location of the gateway moving outward in a circle until
        the edge of the circle around the terminal exclusively consists of cells that border cells which are outside the
        coverage area. This is an optimized way of finding the cells within the coverage area, as only a limited number
        of cells outside the coverage is checked.

        The size of the area that is checked for is based on the parameter 'maxDistance' which can be seen as the radius
        of the coverage area in kilometers.

        The function will not "link" the cells and the gateway. Instead, it will only add a reference in the
        cells to the closest GT. As a result, all GTs must run this function before any linking is performed. The
        linking is done in the function: "linkCells2GTs()", in the Earth class, which also runs this function. This is
        done to handle cases where the coverage areas of two or more GTs are overlapping and the cells must only link to
        one of the GTs.

        The information added to the "cellsWithinRange" list is used for generating flows from the cells to each GT.
        """

        # Up right:
        isWithinRangeX = True
        x = self.gridLocationX
        while isWithinRangeX:
            y = self.gridLocationY
            isWithinRangeY = True
            if x == earth.total_x:  # "roll over" to opposite side of grid.
                x = 0
            cell = earth.cells[x][y]
            distance = self.cellDistance(cell)
            if distance > maxDistance:
                isWithinRangeY = False
                isWithinRangeX = False
            while isWithinRangeY:
                if y == -1:  # "roll over" to opposite side of grid.
                    y = earth.total_y - 1
                cell = earth.cells[x][y]
                distance = self.cellDistance(cell)
                if distance > maxDistance:
                    isWithinRangeY = False
                else:
                    # check if any GT has been added to cell, and if any has check if current GT is closer.
                    if cell.gateway is None or cell.gateway is not None and distance < cell.gateway[1]:
                        # No GT is added to cell or current GT is closer - add current GT.
                        cell.gateway = (self, distance)
                y -= 1  # the y-axis is flipped in the cell grid.
            x += 1

        # Down right:
        isWithinRangeX = True
        x = self.gridLocationX
        while isWithinRangeX:
            y = self.gridLocationY + 1
            isWithinRangeY = True
            if x == earth.total_x:  # "roll over" to opposite side of grid.
                x = 0
            cell = earth.cells[x][y]
            distance = self.cellDistance(cell)
            if distance > maxDistance:
                isWithinRangeY = False
                isWithinRangeX = False
            while isWithinRangeY:
                if y == earth.total_y:  # "roll over" to opposite side of grid.
                    y = 0
                cell = earth.cells[x][y]
                distance = self.cellDistance(cell)
                if distance > maxDistance:
                    isWithinRangeY = False
                else:
                    # check if any GT has been added to cell, and if any has check if current GT is closer.
                    if cell.gateway is None or cell.gateway is not None and distance < cell.gateway[1]:
                        # No GT is added to cell or current GT is closer - add current GT.
                        cell.gateway = (self, distance)
                y += 1  # the y-axis is flipped in the cell grid.
            x += 1

        # up left:
        isWithinRangeX = True
        x = self.gridLocationX - 1
        while isWithinRangeX:
            y = self.gridLocationY
            isWithinRangeY = True
            if x == -1:  # "roll over" to opposite side of grid.
                x = earth.total_x - 1
            cell = earth.cells[x][y]
            distance = self.cellDistance(cell)
            if distance > maxDistance:
                isWithinRangeY = False
                isWithinRangeX = False
            while isWithinRangeY:
                if y == -1:  # "roll over" to opposite side of grid.
                    y = earth.total_y - 1
                cell = earth.cells[x][y]
                distance = self.cellDistance(cell)
                if distance > maxDistance:
                    isWithinRangeY = False
                else:
                    # check if any GT has been added to cell, and if any has check if current GT is closer.
                    if cell.gateway is None or cell.gateway is not None and distance < cell.gateway[1]:
                        # No GT is added to cell or current GT is closer - add current GT.
                        cell.gateway = (self, distance)
                y -= 1  # the y-axis is flipped in the cell grid.
            x -= 1

        # down left:
        isWithinRangeX = True
        x = self.gridLocationX - 1
        while isWithinRangeX:
            y = self.gridLocationY + 1
            isWithinRangeY = True
            if x == -1:  # "roll over" to opposite side of grid.
                x = earth
            cell = earth.cells[x][y]
            distance = self.cellDistance(cell)
            if distance > maxDistance:
                isWithinRangeY = False
                isWithinRangeX = False
            while isWithinRangeY:
                if y == -1:  # "roll over" to opposite side of grid.
                    y = earth.total_y - 1
                cell = earth.cells[x][y]
                distance = self.cellDistance(cell)
                if distance > maxDistance:
                    isWithinRangeY = False
                else:
                    # check if any GT has been added to cell, and if any has check if current GT is closer.
                    if cell.gateway is None or cell.gateway is not None and distance < cell.gateway[1]:
                        # No GT is added to cell or current GT is closer - add current GT.
                        cell.gateway = (self, distance)
                y += 1  # the y-axis is flipped in the cell grid.
            x -= 1

    def timeToFullBlock(self, block):
        """
        Calculates the average time it will take to fill up a data block and returns the actual time based on a
        random variable following an exponential distribution.
        Different from the non reinforcement version of the simulator, this does not include different methods for
        setting the fractions of the data generation to each destination gateway.
        """

        # split the traffic evenly among the active gateways while keeping the fraction to each gateway the same
        # regardless of number of active gateways
        flow = self.totalAvgFlow / (len(self.totalLocations) - 1)

        avgTime = block.size / flow  # the average time to fill the buffer in seconds

        time = np.random.exponential(scale=avgTime)  # the actual time to fill the buffer after adjustment by exp dist.

        return time

    def getTotalFlow(self, avgFlowPerUser, distanceFunc, maxDistance, capacity=None, fraction=1.0):
        """
        This function is used as a precursor for the 'timeToFillBlock' method. Based on one of two distance functions
        this function finds the combined average flow from the combined users within the ground coverage area of the GT.

        Calculates the average combined flow from all cells scaling with distance in one of two ways:
            For the step function this means that it essentially just counts the number of users from the local list and
            multiplies with the flowPerUser value.

            For the slope it means that the slope is found using the flowPerUser and maxDistance as the gradient where
            the function gives 0 at the maximum distance.

            If this logic should be changed, it is important that it is done so in accordance with the
            "findCellsWithinRange" method.
        """
        if balancedFlow:
            self.totalAvgFlow = totalFlow
        else:
            totalAvgFlow = 0
            avgFlowPerUser = avUserLoad

            if distanceFunc == "Step":
                for cell in self.cellsInRange:
                    totalAvgFlow += cell[1] * avgFlowPerUser

            elif distanceFunc == "Slope":
                gradient = (0 - avgFlowPerUser) / (maxDistance - 0)
                for cell in self.cellsInRange:
                    totalAvgFlow += (gradient * cell[2] + avgFlowPerUser) * cell[1]

            else:
                print(
                    "Error, distance function not recognized. Provided function = {}. Allowed functions: {} or {}".format(
                        distanceFunc,
                        "Step",
                        "slope"))
                exit()

            if self.linkedSat[0] is None:
                self.dataRate = self.gs2ngeo.min_rate

            if not capacity:
                capacity = self.dataRate

            if totalAvgFlow < capacity * fraction:
                self.totalAvgFlow = totalAvgFlow
            else:
                self.totalAvgFlow = capacity * fraction

        print(self.name + ': ' + str(self.totalAvgFlow / 1000000000))

    def _hop_key(self, h):
        """Restituisce la chiave confrontabile dello step del path."""
        if isinstance(h, (int, str)):
            return h
        if isinstance(h, (list, tuple)) and h:
            return h[0]
        return getattr(h, "ID", getattr(h, "name", h))

    def _idx_in_path(self, path):
        """Trova l'indice del gateway corrente nel path (match per ID o name)."""
        me = {str(self.ID), str(self.name)}
        for i, h in enumerate(path):
            k = str(self._hop_key(h))
            if k in me:
                return i
        return None

    def __eq__(self, other):
        if self.latitude == other.latitude and self.longitude == other.longitude:
            return True
        else:
            return False

    def __repr__(self):
        return 'Location = {}\n Longitude = {}\n Latitude = {}\n pos x= {}, pos y= {}, pos z= {}'.format(
            self.name,
            self.longitude,
            self.latitude,
            self.x,
            self.y,
            self.z)


# A single cell on earth
class Cell:
    def __init__(self, total_x, total_y, cell_x, cell_y, users, Re=6378e3, f=20e9, bw=200e6, noise_power=1 / (1e11)):
        # X and Y coordinates of the cell on the dataset map
        self.map_x = cell_x
        self.map_y = cell_y
        # Latitude and longitude of the cell as per dataset map
        self.latitude = math.pi * (0.5 - cell_y / total_y)
        self.longitude = (cell_x / total_x - 0.5) * 2 * math.pi
        if self.latitude < -5 or self.longitude < -5:
            print("less than 0")
            print(self.longitude, self.latitude)
            print(cell_x, cell_y)
            # exit()
        # Actual area the cell covers on earth (scaled for)
        self.area = 4 * math.pi * Re * Re * math.cos(self.latitude) / (total_x * total_y)
        # X,Y,Z coordinates to the center of the cell (assumed)
        self.x = Re * math.cos(self.latitude) * math.cos(self.longitude)
        self.y = Re * math.cos(self.latitude) * math.sin(self.longitude)
        self.z = Re * math.sin(self.latitude)

        self.users = users  # Population in the cell
        self.f = f  # Frequency used by the cell
        self.bw = bw  # Bandwidth used for the cell
        self.noise_power = noise_power  # Noise power for the cell
        self.rejected = True  # Usefulfor applications process to show if the cell is rejected or accepted
        self.gateway = None  # (groundstation, distance)

    def __repr__(self):
        return 'Users = {}\n area = {} km^2\n longitude = {} deg\n latitude = {} deg\n pos x = {}\n pos y = {}\n pos ' \
               'z = {}\n x position on map = {}\n y position on map = {}'.format(
            self.users,
            '%.2f' % (self.area / 1e6),
            '%.2f' % math.degrees(self.longitude),
            '%.2f' % math.degrees(self.latitude),
            '%.2f' % self.x,
            '%.2f' % self.y,
            '%.2f' % self.z,
            '%.2f' % self.map_x,
            '%.2f' % self.map_y)

    def setGT(self, gateways, maxDistance=30):
        """
        Finds the closest gateway and updates the internal attribute 'self.gateway' as a tuple:
        (Gateway, distance to terminal). If the distance to the closest gateway is less than some maximum
        distance, the cell information is added to the gateway.
        """
        closestGT = (gateways[0], gateways[0].cellDistance(self))
        for gateway in gateways[1:]:
            distanceToGT = gateway.cellDistance(self)
            if distanceToGT < closestGT[1]:
                closestGT = (gateway, distanceToGT)
        self.gateway = closestGT

        if closestGT[1] <= maxDistance:
            closestGT[0].addCell(
                [(math.degrees(self.latitude), math.degrees(self.longitude)), self.users, closestGT[1]])
        else:
            self.users = 0
        return closestGT


# Earth consisting of cells
# @profile
class Earth:
    def __init__(self, env, img_path, gt_path, constellation, inputParams, deltaT, totalLocations, getRates=False,
                 window=None, outputPath='/', terrestrial_nodes_path=None, enable_gateway_traffic=False):
        # Input the population count data
        # img_path = 'Population Map/gpw_v4_population_count_rev11_2020_15_min.tif'
        self.outputPath = outputPath
        self.plotPaths = plotPath
        self.lostBlocks = 0
        self.queues = []
        self.loss = []
        self.lossAv = []
        self.DDQNA = None

        self.step = 0
        self.nMovs = 0  # number of total movements done by the constellation
        self.epsilon = []  # set of epsilon values
        self.rewards = []  # set of rewards
        self.trains = []  # Set of times when a fit to any dnn has happened
        self.graph = None
        self.CKA = []

        pop_count_data = Image.open(img_path)

        pop_count = np.array(pop_count_data)
        pop_count[pop_count < 0] = 0  # ensure there are no negative values

        # total image sizes
        [self.total_x, self.total_y] = pop_count_data.size

        self.total_cells = self.total_x * self.total_y

        # List of all cells stored in a 2d array as per the order in dataset
        self.cells = []
        # Scale factor to convert population density to realistic user counts
        # TIF files typically contain very small density values, so we scale them up significantly
        POPULATION_SCALE_FACTOR = 100  # Scale up population density to user counts (more realistic)

        for i in range(self.total_x):
            self.cells.append([])
            for j in range(self.total_y):
                # Scale the population data to represent realistic user counts
                scaled_users = int(pop_count[j][i] * POPULATION_SCALE_FACTOR)
                self.cells[i].append(Cell(self.total_x, self.total_y, i, j, scaled_users))

        # window is a list with the coordinate bounds of our window of interest
        # format for window = [western longitude, eastern longitude, southern latitude, northern latitude]
        if window is not None:  # if window provided
            # latitude, longitude bounds:
            self.lati = [window[2], window[3]]
            self.longi = [window[0], window[1]]
            # dataset pixel bounds:
            self.windowx = (
                (int)((0.5 + window[0] / 360) * self.total_x), (int)((0.5 + window[1] / 360) * self.total_x))
            self.windowy = (
                (int)((0.5 - window[3] / 180) * self.total_y), (int)((0.5 - window[2] / 180) * self.total_y))
        else:  # set window size as entire world if no window provided
            self.lati = [-90, 90]
            self.longi = [-179, 180]
            self.windowx = (0, self.total_x)
            self.windowy = (0, self.total_y)

        # import gateways from .csv
        self.gateways = []

        # import users from .csv
        self.terrestrial_nodes = []

        gateways = pd.read_csv(gt_path)
        # Load terrestrial nodes from CSV if provided
        if terrestrial_nodes_path:
            terrestrial_nodes = pd.read_csv(terrestrial_nodes_path)
            tn_nodes = terrestrial_nodes['Location'].tolist()

            for i, row in terrestrial_nodes.iterrows():
                name = row['Location']
                lat = row['Latitude']
                lon = row['Longitude']
                node = TerrestrialNode(
                    name,
                    i,
                    lat,
                    lon,
                    self.total_x,
                    self.total_y,
                    len(terrestrial_nodes),
                    env,
                    totalLocations,
                    self,
                    graph=None
                )
                self.terrestrial_nodes.append(node)

        for node in self.terrestrial_nodes:
            node.getTotalFlow(avgFlowPerUser=5e6, capacity=1e12, fraction=1.0)

        length = 0
        for i, location in enumerate(gateways['Location']):
            for name in inputParams['Locations']:
                if name in location.split(","):
                    length += 1

        if inputParams['Locations'][0] != 'All':
            for i, location in enumerate(gateways['Location']):
                for name in inputParams['Locations']:
                    if name in location.split(","):
                        lName = gateways['Location'][i]
                        gtLati = gateways['Latitude'][i]
                        gtLongi = gateways['Longitude'][i]
                        self.gateways.append(Gateway(lName, i, gtLati, gtLongi, self.total_x, self.total_y,
                                                     length, env, totalLocations, self))
                        break
        else:
            for i in range(len(gateways['Latitude'])):
                name = gateways['Location'][i]
                gtLati = gateways['Latitude'][i]
                gtLongi = gateways['Longitude'][i]
                self.gateways.append(Gateway(name, i, gtLati, gtLongi, self.total_x, self.total_y,
                                             len(gateways['Latitude']), env, totalLocations, self))

        self.pathParam = pathing

        # build dizionari base
        self._tn_by_name = {}
        self._gt_by_name = {}

        def _add(dct, key, obj):
            dct[key] = obj
            dct[str(key)] = obj  # alias stringa
            # se l'oggetto ha un ID numerico distinto, aggiungi anche quello
            if hasattr(obj, "ID"):
                dct[obj.ID] = obj
                dct[str(obj.ID)] = obj

        for n in self.terrestrial_nodes:
            _add(self._tn_by_name, getattr(n, "name", None), n)

        for g in self.gateways:
            _add(self._gt_by_name, getattr(g, "name", None), g)

        # merge unico per risoluzione rapida
        self.node_by_name = {}
        self.node_by_name.update(self._tn_by_name)
        self.node_by_name.update(self._gt_by_name)

        # create data Blocks on all GTs.
        if (not getRates) and enable_gateway_traffic:
            for gt in self.gateways:
                gt.makeFillBlockProcesses(self.gateways)

        # create constellation of satellites
        self.LEO = create_Constellation(constellation, env, self)

        if rotateFirst:
            print('Rotating constellation...')
            for constellation in self.LEO:
                constellation.rotate(ndeltas * deltaT)

        # Simpy process for handling moving the constellation and the satellites within the constellation
        self.moveConstellation = env.process(self.moveConstellation(env, deltaT, getRates))

        self.build_lookups()

    def getNodeByName(self, key):
        """
        solve a key that could be:
          - int: prima satellite per ID, altrimenti nodo terrestre con key int
          - str: nodo terrestre/gateway per nome
        Ritorna l'oggetto (Satellite | Gateway | TerrestrialNode) oppure None.
        """
        # 1) int -> prova satellite
        if isinstance(key, int):
            # se hai un dict sat_by_id meglio ancora; altrimenti usa findByID
            try:
                if hasattr(self, 'sat_by_id'):
                    sat = self.sat_by_id.get(key)
                    if sat is not None:
                        return sat
                # fallback
                from math import inf  # if needed
                sat = findByID(self, key)  # already in your base code
                if sat is not None:
                    return sat
            except Exception:
                pass
            # poi prova nodo terrestre con chiave int
            if hasattr(self, 'node_by_name') and isinstance(self.node_by_name, dict):
                n = self.node_by_name.get(key) or self.node_by_name.get(str(key))
                if n is not None:
                    return n
            return None

        # 2) str -> nodo terrestre/gateway
        if isinstance(key, str):
            # terrestri/gateway-come-terrestrial
            if hasattr(self, 'node_by_name') and isinstance(self.node_by_name, dict):
                n = self.node_by_name.get(key)
                if n is not None:
                    return n
            # gateway oggetto Gateway (se vuoi preferire quello)
            if hasattr(self, 'gateways'):
                for g in self.gateways:
                    if str(g.name) == key:
                        return g
            return None

        # 3) already valid object
        if hasattr(key, 'name') or hasattr(key, 'ID'):
            return key
        return None

    def set_window(self, window):  # function to change/set window for the earth
        """
        Unused function
        """
        self.lati = [window[2], window[3]]
        self.longi = [window[0], window[1]]
        self.windowx = ((int)((0.5 + window[0] / 360) * self.total_x), (int)((0.5 + window[1] / 360) * self.total_x))
        self.windowy = ((int)((0.5 - window[3] / 180) * self.total_y), (int)((0.5 - window[2] / 180) * self.total_y))

    def linkCells2GTs(self, distance):
        """
        Finds the cells that are within the coverage areas of all GTs and links them ensuring that a cell only links to
        a single GT.
        """
        start = time.time()

        # Find cells that are within range of all GTs
        for i, gt in enumerate(self.gateways):
            print("Finding cells within coverage area of GT {} of {}".format(i + 1, len(self.gateways)), end='\r')
            gt.findCellsWithinRange(self, distance)
        print('\r')
        print("Time taken to find cells that are within range of all GTs: {} seconds".format(time.time() - start))

        start = time.time()

        # Add reference for cells to the GT they are closest to
        for cells in self.cells:
            for cell in cells:
                if cell.gateway is not None:
                    cell.gateway[0].addCell([(math.degrees(cell.latitude),
                                              math.degrees(cell.longitude)),
                                             cell.users,
                                             cell.gateway[1]])

        print("Time taken to add cell information to all GTs: {} seconds".format(time.time() - start))
        print()

    def linkSats2GTs(self, method):
        """
        Links GTs to satellites. One satellite is only allowed to link to one GT.
        """
        sats = []
        for orbit in self.LEO:
            for sat in orbit.sats:
                sat.linkedGT = None
                sat.GTDist = None
                sats.append(sat)

        if method == "Greedy":
            for GT in self.gateways:
                GT.orderSatsByDist(self.LEO)
                GT.addRefOnSat()

            for orbit in self.LEO:
                for sat in orbit.sats:
                    if sat.linkedGT is not None:
                        sat.linkedGT.link2Sat(sat.GTDist, sat)
        elif method == "Optimize":
            # make cost matrix
            SxGT = np.array([[99999 for _ in range(len(sats))] for _ in range(len(self.gateways))])
            for i, GT in enumerate(self.gateways):
                GT.orderSatsByDist(self.LEO)
                for val, entry in enumerate(GT.satsOrdered):
                    SxGT[i][entry[2][0]] = val

            # find assignment of GSL which minimizes the cost from the cost matrix
            rowInd, colInd = linear_sum_assignment(SxGT)

            # link satellites and GTs
            for i, GT in enumerate(self.gateways):
                if SxGT[rowInd[i]][colInd[i]] < len(GT.satsOrdered):
                    sat = GT.satsOrdered[SxGT[rowInd[i]][colInd[i]]]
                    GT.link2Sat(sat[0], sat[1])
                else:
                    GT.linkedSat = (None, None)
                    print("no satellite for GT {}".format(GT.name))

    def getCellUsers(self):
        """
        Used for plotting the population map.
        """
        temp = []
        for i, cellList in enumerate(self.cells):
            temp.append([])
            for cell in cellList:
                temp[i].append(cell.users)
        return temp

    def updateSatelliteProcessesSimpler(self, graph):
        """

        Function from the non-reinforcement implementation. However, due to the paths not existing between transmitter
        and destination gateways (they get created as the blocks travel through the constellation), this version does
        work with Q-Learning and Deep-Learning.

        Can be used for a simpler version of updating the processes on satellites. However, it does not take into
        account that some processes may be able to continue without being stopped. Stopping the processes may lose
        time of the transmission of a block.

        Function which ensures all processes on all satellites are updated after constellation movement. This is done in
        several steps:
            - All blocks waiting to be sent or currently being sent has their paths updated.
            - All processes are stopped and remade according to current links - all transmission progress is lost on
            blocks currently being transmitted.
            - All buffers are emptied and blocks are redistributed to new buffers according to the blocks' arrival time
            at the satellite.
        """

        # update ISL references in all satellites, adjust data rate to GTs and ensure send-processes are correct
        sats = []
        for plane in self.LEO:
            for sat1 in plane.sats:
                sats.append(sat1)
        for plane in self.LEO:
            for sat in plane.sats:

                # remake path for all blocks
                for buffer in sat.sendBufferSatsIntra:
                    for block in buffer[1]:
                        destination = block.destination.name
                        newPath = getShortestPath(sat.ID, destination, self.pathParam, graph)
                        path = None
                        # splice old and new path
                        for i, step in enumerate(block.path):
                            if step[0] == sat.ID:
                                path = block.path[:i] + newPath
                                break
                        if path is None:
                            print("no path to sat:")
                            print(block)
                            exit()
                        block.path = path
                for buffer in sat.sendBufferSatsInter:
                    for block in buffer[1]:
                        destination = block.destination.name
                        newPath = getShortestPath(sat.ID, destination, self.pathParam, graph)
                        path = None
                        # splice old and new path
                        for i, step in enumerate(block.path):
                            if step[0] == sat.ID:
                                path = block.path[:i] + newPath
                                break
                        if path is None:
                            print("no path to sat:")
                            print(block)
                            exit()
                        block.path = path
                for block in sat.sendBufferGT[1]:
                    destination = block.destination.name
                    newPath = getShortestPath(sat.ID, destination, self.pathParam, graph)
                    path = None
                    # splice old and new path
                    for i, step in enumerate(block.path):
                        if step[0] == sat.ID:
                            path = block.path[:i] + newPath
                            break
                    if path is None:
                        print("no path to GT:")
                        print(block)
                        exit()
                    block.path = path
                for block in sat.tempBlocks[0]:
                    # Skip if not a DataBlock (could be an Event)
                    if not hasattr(block, 'destination'):
                        continue
                    destination = block.destination.name
                    newPath = getShortestPath(sat.ID, destination, self.pathParam, graph)
                    path = None
                    # splice old and new path
                    for i, step in enumerate(block.path):
                        if step[0] == sat.ID:
                            path = block.path[:i] + newPath
                            break
                    if path is None:
                        print("no path from Temp:")
                        print(block)
                        exit()
                    block.path = path

                # find neighboring satellites
                neighbors = list(nx.neighbors(graph, sat.ID))
                itt = 0
                neighborSats = []
                for sat2 in sats:
                    if sat2.ID in neighbors:
                        dataRate = nx.path_weight(graph, [sat2.ID, sat.ID], "dataRateOG")
                        distance = nx.path_weight(graph, [sat2.ID, sat.ID], "slant_range")
                        neighborSats.append((distance, sat2, dataRate))
                        itt += 1
                        if itt == len(neighbors):
                            break

                sat.intraSats = []
                sat.interSats = []

                # add new satellites as references
                for neighbor in neighborSats:
                    if neighbor[1].in_plane == sat.in_plane:
                        sat.intraSats.append(neighbor)
                    else:
                        sat.interSats.append(neighbor)

                # stop all processes
                for process in sat.sendBlocksSatsInter:
                    process.interrupt()
                for process in sat.sendBlocksSatsIntra:
                    process.interrupt()
                for process in sat.sendBlocksGT:
                    process.interrupt()
                sat.sendBlocksSatsIntra = []
                sat.sendBlocksSatsInter = []
                sat.sendBlocksGT = []

                # add all blocks to list and reset queues
                blocksToDistribute = []
                for buffer in sat.sendBufferSatsIntra:
                    for block in buffer[1]:
                        blocksToDistribute.append((block.checkPoints[-1], block))
                sat.sendBufferSatsIntra = []
                for buffer in sat.sendBufferSatsInter:
                    for block in buffer[1]:
                        blocksToDistribute.append((block.checkPoints[-1], block))
                sat.sendBufferSatsInter = []
                for block in sat.sendBufferGT[1]:
                    blocksToDistribute.append((block.checkPoints[-1], block))
                sat.sendBufferGT = ([sat.env.event()], [])

                # remake all processes
                if sat.linkedGT is not None:
                    sat.adjustDownRate()
                    # make a process for the GSL from sat to GT
                    sat.sendBlocksGT.append(sat.env.process(sat.sendBlock((sat.GTDist, sat.linkedGT), False)))

                for neighbor in sat.intraSats:
                    # make a send buffer for each ISL ([self.env.event()], [DataBlock(0, 0, "0", 0)], 0)
                    sat.sendBufferSatsIntra.append(([sat.env.event()], [], neighbor[1].ID))

                    # make a process for each ISL
                    sat.sendBlocksSatsIntra.append(sat.env.process(sat.sendBlock(neighbor, True, True)))

                for neighbor in sat.interSats:
                    # make a send buffer for each ISL ([self.env.event()], [DataBlock(0, 0, "0", 0)], 0)
                    sat.sendBufferSatsInter.append(([sat.env.event()], [], neighbor[1].ID))

                    # make a process for each ISL
                    sat.sendBlocksSatsInter.append(sat.env.process(sat.sendBlock(neighbor, True, False)))

                # sort blocks by arrival time at satellite
                blocksToDistribute.sort()
                # add blocks to the correct queues based on next step in their path
                # since the blocks list is sorted by arrival time, the order in the new queues is correct
                for block in blocksToDistribute:
                    # get this satellite's index in the blocks path
                    index = None
                    for i, step in enumerate(block[1].path):
                        if sat.ID == step[0]:
                            index = i

                    # check if next step in path is GT (last step in path)
                    if index == len(block[1].path) - 2:
                        # add block to GT send-buffer
                        if not sat.sendBufferGT[0][0].triggered:
                            sat.sendBufferGT[0][0].succeed()
                            sat.sendBufferGT[1].append(block[1])
                        else:
                            newEvent = sat.env.event().succeed()
                            sat.sendBufferGT[0].append(newEvent)
                            sat.sendBufferGT[1].append(block[1])
                    else:
                        # get ID of next sat and find if it is intra or inter
                        ID = None
                        isIntra = False
                        for neighborSat in sat.intraSats:
                            id = neighborSat[1].ID
                            if id == block[1].path[index + 1][0]:
                                ID = neighborSat[1].ID
                                isIntra = True
                        for neighborSat in sat.interSats:
                            id = neighborSat[1].ID
                            if id == block[1].path[index + 1][0]:
                                ID = neighborSat[1].ID

                        if ID is not None:
                            sendBuffer = None
                            # find send-buffer for the satellite
                            if isIntra:
                                for buffer in sat.sendBufferSatsIntra:
                                    if ID == buffer[2]:
                                        sendBuffer = buffer
                            else:
                                for buffer in sat.sendBufferSatsInter:
                                    if ID == buffer[2]:
                                        sendBuffer = buffer

                            # add block to buffer
                            if not sendBuffer[0][0].triggered:
                                sendBuffer[0][0].succeed()
                                sendBuffer[1].append(block[1])
                            else:
                                newEvent = sat.env.event().succeed()
                                sendBuffer[0].append(newEvent)
                                sendBuffer[1].append(block[1])
                        else:
                            print("buffer for next satellite in path could not be found")

    def updateSatelliteProcessesCorrect(self, graph):
        """

        Function from the non-reinforcement implementation. However, due to the paths not existing between transmitter
        and destination gateways (they get created as the blocks travel through the constellation), this version does
        work with Q-Learning and Deep-Learning.

        Function which ensures all processes on all satellites are updated after constellation movement. This is done in
        several steps:
            - All blocks waiting to be sent or currently being sent has their paths updated.
            - ISLs are updated with references to new inter-orbit satellites (intra-orbit links do not change).
                - This includes updating buffer if ISL is changed
                - It also includes remaking send-process if ISL is changed
                - Despite intra-orbit links not changing, blocks in an intra-orbit buffer may have to be moved.
            - GSL is updated:
                - Depending on new status - whether the satellite has a GSL or not - and past status - whether the
                satellite had a GSL or not - GSL buffer and process is handled accordingly.
            - All blocks not currently being transmitted to a satellite/GT, which is still present as a ISL or GSL, are
            redistributed to send-buffers according to their arrival time at the satellite.

        This function differentiates from the simple version by allowing continued operation of send-processes after
        constellation movement if the link is not broken.
        """
        sats = []
        for plane in self.LEO:
            for sat1 in plane.sats:
                sats.append(sat1)

        for plane in self.LEO:
            for sat in plane.sats:
                # remake path for all blocks

                for buffer in sat.sendBufferSatsIntra:
                    index = 0
                    while index < len(buffer[1]):
                        block = buffer[1][index]
                        destination = block.destination.name
                        newPath = getShortestPath(sat.ID, destination, self.pathParam, graph)

                        if newPath == -1:
                            if len(buffer[0]) == 1:
                                buffer[0].pop(index)
                                buffer[1].pop(index)
                                buffer[0].append(sat.env.event())
                            else:
                                buffer[0].pop(index)
                                buffer[1].pop(index)
                            continue
                        path = None
                        # splice old and new path
                        for i, step in enumerate(block.path):
                            if step[0] == sat.ID:
                                path = block.path[:i] + newPath
                                break
                        if path is None:
                            print("no path to sat:")
                            print(block)
                            exit()
                        block.isNewPath = True
                        block.oldPath = block.path
                        block.newPath = newPath
                        block.path = path
                        index += 1

                for buffer in sat.sendBufferSatsInter:
                    index = 0
                    while index < len(buffer[1]):
                        block = buffer[1][index]
                        destination = block.destination.name
                        newPath = getShortestPath(sat.ID, destination, self.pathParam, graph)

                        if newPath == -1:
                            if len(buffer[0]) == 1:
                                buffer[0].pop(index)
                                buffer[1].pop(index)
                                buffer[0].append(sat.env.event())
                            else:
                                buffer[0].pop(index)
                                buffer[1].pop(index)
                            continue
                        path = None
                        # splice old and new path
                        for i, step in enumerate(block.path):
                            if step[0] == sat.ID:
                                path = block.path[:i] + newPath
                                break
                        if path is None:
                            print("no path to sat:")
                            print(block)
                            exit()
                        block.isNewPath = True
                        block.oldPath = block.path
                        block.newPath = newPath
                        block.path = path
                        index += 1

                index = 0
                while index < len(sat.sendBufferGT[1]):
                    block = sat.sendBufferGT[1][index]
                    destination = block.destination.name
                    newPath = getShortestPath(sat.ID, destination, self.pathParam, graph)

                    if newPath == -1:
                        if len(sat.sendBufferGT[0]) == 1:
                            sat.sendBufferGT[0].pop(index)
                            sat.sendBufferGT[1].pop(index)
                            sat.sendBufferGT[0].append(sat.env.event())
                        else:
                            sat.sendBufferGT[0].pop(index)
                            sat.sendBufferGT[1].pop(index)
                        continue
                    path = None
                    # splice old and new path
                    for i, step in enumerate(block.path):
                        if step[0] == sat.ID:
                            path = block.path[:i] + newPath
                            break
                    if path is None:
                        print("no path to GT:")
                        print(block)
                        exit()
                    block.isNewPath = True
                    block.oldPath = block.path
                    block.newPath = newPath
                    block.path = path
                    index += 1

                index = 0
                while index < len(sat.tempBlocks[0]):
                    block = sat.tempBlocks[0][index]

                    # Skip if not a DataBlock (could be an Event)
                    if not hasattr(block, 'destination'):
                        index += 1
                        continue

                    destination = block.destination.name
                    newPath = getShortestPath(sat.ID, destination, self.pathParam, graph)

                    if newPath == -1:
                        block.path = -1
                        if len(sat.tempBlocks[0]) == 1:
                            sat.tempBlocks[0].pop(index)
                            if len(sat.tempBlocks[1]) > index:
                                sat.tempBlocks[1].pop(index)
                            sat.tempBlocks[0].append(sat.env.event())
                        else:
                            sat.tempBlocks[0].pop(index)
                            if len(sat.tempBlocks[1]) > index:
                                sat.tempBlocks[1].pop(index)
                        continue

                    path = None
                    # splice old and new path
                    for i, step in enumerate(block.path):
                        if step[0] == sat.ID:
                            path = block.path[:i] + newPath
                            break
                    if path is None:
                        print("no path from Temp:")
                        print(block)
                        exit()
                    block.isNewPath = True
                    block.oldPath = block.path
                    block.newPath = newPath
                    block.path = path
                    index += 1

                # find neighboring satellites
                neighbors = list(nx.neighbors(graph, sat.ID))
                itt = 0
                neighborSatsInter = []
                for sat2 in sats:
                    if sat2.ID in neighbors:
                        # we only care about the satellite if it is an inter-plane ISL
                        # we assume intra-plane ISLs will not change
                        if sat2.in_plane != sat.in_plane:
                            dataRate = nx.path_weight(graph, [sat2.ID, sat.ID], "dataRateOG")
                            distance = nx.path_weight(graph, [sat2.ID, sat.ID], "slant_range")
                            neighborSatsInter.append((distance, sat2, dataRate))
                        itt += 1
                        if itt == len(neighbors):
                            break
                sat.interSats = neighborSatsInter
                # list of blocks to be redistributed
                blocksToDistribute = []

                ### inter-plane ISLs ###

                sat.newBuffer = [True for _ in range(len(neighborSatsInter))]

                # make a list of False entries for each current neighbor
                sameSats = [False for _ in range(len(neighborSatsInter))]

                buffers = [None for _ in range(len(neighborSatsInter))]
                processes = [None for _ in range(len(neighborSatsInter))]

                # go through each process/buffer
                #   - check if the satellite is still there:
                #       - if it is, change the corresponding False to True, handle blocks and add process and buffer references to temporary list
                #       - if it is not, remove blocks from buffer and stop process
                for bufferIndex, buffer in enumerate(sat.sendBufferSatsInter):
                    # check if the satellite is still there
                    isPresent = False
                    for neighborIndex, neighbor in enumerate(neighborSatsInter):
                        if buffer[2] == neighbor[1].ID:
                            isPresent = True
                            sameSats[neighborIndex] = True

                            ## handle blocks
                            # check if there are blocks in the buffer
                            if buffer[1]:
                                # find index of satellite in block's path
                                index = None
                                for i, step in enumerate(buffer[1][0].path):
                                    if sat.ID == step[0]:
                                        index = i
                                        break

                                # check if next step in path corresponds to buffer's satellite
                                if buffer[1][0].path[index + 1][0] == buffer[2]:
                                    # add all but the first block to redistribution list
                                    for block in buffer[1][1:]:
                                        blocksToDistribute.append((block.checkPoints[-1], block))

                                    # add buffer with only first block present to temp list
                                    buffers[neighborIndex] = (
                                        [sat.env.event().succeed()], [sat.sendBufferSatsInter[bufferIndex][1][0]],
                                        buffer[2])
                                    processes[neighborIndex] = sat.sendBlocksSatsInter[bufferIndex]
                                else:
                                    # add all blocks to redistribution list
                                    for block in buffer[1]:
                                        blocksToDistribute.append((block.checkPoints[-1], block))
                                    # reset buffer
                                    buffers[neighborIndex] = ([sat.env.event()], [], buffer[2])

                                    # reset process
                                    sat.sendBlocksSatsInter[bufferIndex].interrupt()
                                    processes[neighborIndex] = sat.env.process(sat.sendBlock(neighbor, True, False))

                            else:  # there are no blocks in the buffer
                                # add buffer and remake process
                                buffers[neighborIndex] = sat.sendBufferSatsInter[bufferIndex]
                                sat.sendBlocksSatsInter[bufferIndex].interrupt()
                                processes[neighborIndex] = sat.env.process(sat.sendBlock(neighbor, True, False))
                                # sendBlocksSatsInter[bufferIndex]

                            break
                    if not isPresent:
                        # add blocks to redistribution list
                        for block in buffer[1]:
                            blocksToDistribute.append((block.checkPoints[-1], block))
                        # stop process
                        sat.sendBlocksSatsInter[bufferIndex].interrupt()

                # make buffer and process for new neighbors(s)
                # - go through list of previously false entries:
                #   - check  entry for each neighbor:
                #       - if False, create buffer and process for new neighbor
                # - clear temporary list of processes and buffers
                for entryIndex, entry in enumerate(sameSats):
                    if not entry:
                        buffers[entryIndex] = ([sat.env.event()], [], neighborSatsInter[entryIndex][1].ID)
                        processes[entryIndex] = sat.env.process(
                            sat.sendBlock(neighborSatsInter[entryIndex], True, False))

                # overwrite buffers and processes
                sat.sendBlocksSatsInter = processes
                sat.sendBufferSatsInter = buffers

                ### intra-plane ISLs ###
                # check blocks for each buffer
                for bufferIndex, buffer in enumerate(sat.sendBufferSatsIntra):
                    ## handle blocks
                    # check if there are blocks in the buffer
                    if buffer[1]:
                        # find index of satellite in block's path
                        index = None
                        for i, step in enumerate(buffer[1][0].path):
                            if sat.ID == step[0]:
                                index = i
                                break

                        # check if next step in path corresponds to buffer's satellite
                        if buffer[1][0].path[index + 1][0] == buffer[2]:
                            # add all but the first block to redistribution list
                            for block in buffer[1][1:]:
                                blocksToDistribute.append((block.checkPoints[-1], block))

                            # remove all but the first block and event from the buffer
                            length = len(sat.sendBufferSatsIntra[bufferIndex][1]) - 1
                            for _ in range(length):
                                sat.sendBufferSatsIntra[bufferIndex][1].pop(1)
                                sat.sendBufferSatsIntra[bufferIndex][0].pop(1)

                        else:
                            # add all blocks to redistribution list
                            for block in buffer[1]:
                                blocksToDistribute.append((block.checkPoints[-1], block))
                            # reset buffer
                            sat.sendBufferSatsIntra[bufferIndex] = ([sat.env.event()], [], buffer[2])

                            # reset process
                            sat.sendBlocksSatsIntra[bufferIndex].interrupt()
                            sat.sendBlocksSatsIntra[bufferIndex] = sat.env.process(
                                sat.sendBlock(sat.intraSats[bufferIndex], True, True))

                ### GSL ###
                # check if satellite has a linked GT
                if sat.linkedGT is not None:
                    sat.adjustDownRate()

                    # check if it had a sendBlocksGT process
                    if sat.sendBlocksGT:
                        # check if there are any blocks in the buffer
                        if sat.sendBufferGT[1]:
                            # check if linked GT is the same as the destination of first block in sendBufferGT
                            if sat.sendBufferGT[1][0].destination != sat.linkedGT:
                                sat.sendBlocksGT[0].interrupt()
                                sat.sendBlocksGT = []

                                # remove blocks from queue and add to list of blocks which should be redistributed
                                for block in sat.sendBufferGT[1]:
                                    blocksToDistribute.append(
                                        (block.checkPoints[-1], block))  # (latest checkpoint time, block)
                                sat.sendBufferGT = ([sat.env.event()], [])

                                # make new send process for new linked GT
                                sat.sendBlocksGT.append(
                                    sat.env.process(sat.sendBlock((sat.GTDist, sat.linkedGT), False)))
                            else:
                                # keep the first block in the buffer and let process continue
                                for block in sat.sendBufferGT[1][1:]:
                                    blocksToDistribute.append(
                                        (block.checkPoints[-1], block))  # (latest checkpoint time, block)
                                length = len(sat.sendBufferGT[1]) - 1
                                for _ in range(length):
                                    sat.sendBufferGT[1].pop(1)  # pop all but the first block
                                    sat.sendBufferGT[0].pop(1)  # pop all but the first event

                        else:  # there are no blocks in the buffer
                            sat.sendBlocksGT[0].interrupt()
                            sat.sendBlocksGT = []
                            sat.sendBufferGT = ([sat.env.event()], [])
                            # make new send process for new linked GT
                            sat.sendBlocksGT.append(sat.env.process(sat.sendBlock((sat.GTDist, sat.linkedGT), False)))

                    else:  # it had no process running
                        # there should be no blocks in the GT buffer, but just in case - if there are none, then the for loop will not run
                        # remove blocks from queue and add to list of blocks which should be redistributed
                        for block in sat.sendBufferGT[1]:
                            blocksToDistribute.append((block.checkPoints[-1], block))  # (latest checkpoint time, block)
                        sat.sendBufferGT = ([sat.env.event()], [])

                        # make new send process for new linked GT
                        sat.sendBlocksGT.append(sat.env.process(sat.sendBlock((sat.GTDist, sat.linkedGT), False)))

                else:  # no linked GT
                    # check if there is a sendBlocksGT process
                    if sat.sendBlocksGT:
                        sat.sendBlocksGT[0].interrupt()
                        sat.sendBlocksGT = []

                        # remove blocks from queue and add to list of blocks which should be redistributed
                        for block in sat.sendBufferGT[1]:
                            blocksToDistribute.append((block.checkPoints[-1], block))  # (latest checkpoint time, block)
                        sat.sendBufferGT = ([sat.env.event()], [])

                # sort blocks by arrival time at satellite
                blocksToDistribute.sort()
                # add blocks to the correct queues based on next step in their path
                # since the blocks list is sorted by arrival time, the order in the new queues is correct
                for block in blocksToDistribute:
                    # get this satellite's index in the blocks path
                    index = None
                    for i, step in enumerate(block[1].path):
                        if sat.ID == step[0]:
                            index = i

                    # check if next step in path is GT (last step in path)
                    if index == len(block[1].path) - 2:
                        # add block to GT send-buffer
                        if not sat.sendBufferGT[0][0].triggered:
                            sat.sendBufferGT[0][0].succeed()
                            sat.sendBufferGT[1].append(block[1])
                        else:
                            newEvent = sat.env.event().succeed()
                            sat.sendBufferGT[0].append(newEvent)
                            sat.sendBufferGT[1].append(block[1])
                    else:
                        # get ID of next sat and find if it is intra or inter
                        ID = None
                        isIntra = False
                        for neighborSat in sat.intraSats:
                            id = neighborSat[1].ID
                            if id == block[1].path[index + 1][0]:
                                ID = neighborSat[1].ID
                                isIntra = True
                        for neighborSat in sat.interSats:
                            id = neighborSat[1].ID
                            if id == block[1].path[index + 1][0]:
                                ID = neighborSat[1].ID

                        if ID is not None:
                            sendBuffer = None
                            # find send-buffer for the satellite
                            if isIntra:
                                for buffer in sat.sendBufferSatsIntra:
                                    if ID == buffer[2]:
                                        sendBuffer = buffer
                            else:
                                for buffer in sat.sendBufferSatsInter:
                                    if ID == buffer[2]:
                                        sendBuffer = buffer

                            # add block to buffer
                            if not sendBuffer[0][0].triggered:
                                sendBuffer[0][0].succeed()
                                sendBuffer[1].append(block[1])
                            else:
                                newEvent = sat.env.event().succeed()
                                sendBuffer[0].append(newEvent)
                                sendBuffer[1].append(block[1])
                        else:
                            print("buffer for next satellite in path could not be found")

    def updateSatelliteProcessesRL(self, graph):
        """
        Update: This function works now. The issue is that all the inter-plane packets that were in a queue to be sent are discarded
        when the graph is updated and those links stop existing.
        This function does not work correctly! The remaking of processes and queues fails when the satellites move
        enough so that new links must be formed.

        This function takes into account that the paths are not complete and the next step may not have been chosen yet.

        Function which ensures all processes on all satellites are updated after constellation movement. This is done in
        several steps:
            - All blocks waiting to be sent or currently being sent has their paths updated.
            - ISLs are updated with references to new inter-orbit satellites (intra-orbit links do not change).
                - This includes updating buffer if ISL is changed
                - It also includes remaking send-process if ISL is changed
                - Despite intra-orbit links not changing, blocks in an intra-orbit buffer may have to be moved.
            - GSL is updated:
                - Depending on new status - whether the satellite has a GSL or not - and past status - whether the
                satellite had a GSL or not - GSL buffer and process is handled accordingly.
            - All blocks not currently being transmitted to a satellite/GT, which is still present as a ISL or GSL, are
            redistributed to send-buffers according to their arrival time at the satellite.

        This function differentiates from the simple version by allowing continued operation of send-processes after
        constellation movement if the link is not broken.
        """
        # update linked sats
        sats = []
        for plane in self.LEO:
            for sat in plane.sats:
                sats.append(sat)
                if self.pathParam == 'Q-Learning':
                    # Update ISL
                    linkedSats = getLinkedSats(sat, graph, self)
                    sat.QLearning.linkedSats = {'U': linkedSats['U'],
                                                'D': linkedSats['D'],
                                                'R': linkedSats['R'],
                                                'L': linkedSats['L']}
                elif self.pathParam == 'Deep Q-Learning':
                    # update ISL. Intra-plane should not change
                    sat.findIntraNeighbours(self)
                    sat.findInterNeighbours(self)

        for plane in self.LEO:
            for sat in plane.sats:
                # get next step for all blocks
                # doing this here assumes that the constellation movement will have a limited effect on the links
                # and that the queue sizes will not change significantly.

                # intra satellite buffers
                for buffer in sat.sendBufferSatsIntra:
                    index = 0
                    while index < len(buffer[1]):
                        block = buffer[1][index]
                        nextHop = None

                        if len(block.QPath) > 3:  # the block does not come from a gateway
                            if sat.QLearning is not None:  # Q-Learning
                                nextHop = sat.QLearning.makeAction(block, sat,
                                                                   sat.orbPlane.earth.gateways[0].graph,
                                                                   sat.orbPlane.earth, prevSat=(
                                        findByID(sat.orbPlane.earth, block.QPath[len(block.QPath) - 3][0])))
                            elif sat.DDQNA is not None:  # Deep Q-Learning-Online phase
                                nextHop = sat.DDQNA.makeDeepAction(block, sat,
                                                                   sat.orbPlane.earth.gateways[0].graph,
                                                                   sat.orbPlane.earth, prevSat=(
                                        findByID(sat.orbPlane.earth, block.QPath[len(block.QPath) - 3][0])))
                            elif self.DDQNA is not None:  # Deep Q-Learning-Offline phase
                                # nextHop = sat.orbPlane.earth.DDQNA.makeDeepAction(block, sat,
                                nextHop = self.DDQNA.makeDeepAction(block, sat,
                                                                    sat.orbPlane.earth.gateways[
                                                                        0].graph,
                                                                    sat.orbPlane.earth, prevSat=(
                                        findByID(sat.orbPlane.earth, block.QPath[len(block.QPath) - 3][0])))
                            else:
                                print(f'No learning model for sat: {sat.ID}')
                        else:
                            if sat.QLearning is not None:  # Q-Learning
                                nextHop = sat.QLearning.makeAction(block, sat,
                                                                   sat.orbPlane.earth.gateways[0].graph,
                                                                   sat.orbPlane.earth)
                            elif sat.DDQNA is not None:  # Deep Q-Learning-Offline phase
                                nextHop = sat.DDQNA.makeDeepAction(block, sat,
                                                                   sat.orbPlane.earth.gateways[
                                                                       0].graph,
                                                                   sat.orbPlane.earth)
                            elif self.DDQNA is not None:  # Deep Q-Learning-Offline phase
                                # nextHop = sat.orbPlane.earth.DDQNA.makeDeepAction(block, sat,
                                nextHop = self.DDQNA.makeDeepAction(block, sat,
                                                                    sat.orbPlane.earth.gateways[
                                                                        0].graph,
                                                                    sat.orbPlane.earth)
                            else:
                                print(f'No learning model for sat: {sat.ID}')

                        if nextHop is None:
                            print(f'Something wrong with block: {block}')

                        elif nextHop != 0:
                            block.QPath[-2] = nextHop
                            pathPlot = block.QPath.copy()
                            pathPlot.pop()
                        else:
                            pathPlot = block.QPath.copy()

                        # If plotPath plots an image for every action taken. Prints 1/10 of blocks. # ANCHOR plot action earth 1
                        #################################################################
                        if sat.orbPlane.earth.plotPaths:
                            if int(block.ID[len(block.ID) - 1]) == 0:
                                os.makedirs(sat.orbPlane.earth.outputPath + '/pictures/',
                                            exist_ok=True)  # create output path
                                outputPath = sat.orbPlane.earth.outputPath + '/pictures/' + block.ID + '_' + str(
                                    len(block.QPath)) + '_'
                                # plotShortestPath(sat.orbPlane.earth, pathPlot, outputPath)
                                plotShortestPath(sat.orbPlane.earth, pathPlot, outputPath, ID=block.ID,
                                                 time=block.creationTime)
                        #################################################################

                        # path = block.QPath  # if there is Q-Learning the path will be repalced with the QPath
                        index += 1

                # inter satellite buffers
                for buffer in sat.sendBufferSatsInter:
                    index = 0
                    while index < len(buffer[1]):
                        block = buffer[1][index]

                        if len(block.QPath) > 3:  # the block does not come from a gateway
                            if sat.QLearning:
                                nextHop = sat.QLearning.makeAction(block, sat,
                                                                   sat.orbPlane.earth.gateways[0].graph,
                                                                   sat.orbPlane.earth, prevSat=(
                                        findByID(sat.orbPlane.earth, block.QPath[len(block.QPath) - 3][0])))
                            elif sat.DDQNA is not None:
                                nextHop = sat.DDQNA.makeDeepAction(block, sat,
                                                                   sat.orbPlane.earth.gateways[
                                                                       0].graph,
                                                                   sat.orbPlane.earth, prevSat=(
                                        findByID(sat.orbPlane.earth, block.QPath[len(block.QPath) - 3][0])))
                            else:
                                nextHop = sat.orbPlane.earth.DDQNA.makeDeepAction(block, sat,
                                                                                  sat.orbPlane.earth.gateways[
                                                                                      0].graph,
                                                                                  sat.orbPlane.earth, prevSat=(
                                        findByID(sat.orbPlane.earth, block.QPath[len(block.QPath) - 3][0])))
                        else:
                            if sat.QLearning:
                                nextHop = sat.QLearning.makeAction(block, sat,
                                                                   sat.orbPlane.earth.gateways[0].graph,
                                                                   sat.orbPlane.earth)
                            elif sat.DDQNA is not None:
                                nextHop = sat.DDQNA.makeDeepAction(block, sat,
                                                                   sat.orbPlane.earth.gateways[
                                                                       0].graph,
                                                                   sat.orbPlane.earth)

                            else:
                                nextHop = sat.orbPlane.earth.DDQNA.makeDeepAction(block, sat,
                                                                                  sat.orbPlane.earth.gateways[
                                                                                      0].graph,
                                                                                  sat.orbPlane.earth)

                        if nextHop != 0:
                            block.QPath[-2] = nextHop
                            pathPlot = block.QPath.copy()
                            pathPlot.pop()
                        else:
                            pathPlot = block.QPath.copy()

                        # If plotPath plots an image for every action taken. Prints 1/10 of blocks. # ANCHOR plot action earth 2
                        #################################################################
                        if sat.orbPlane.earth.plotPaths:
                            if int(block.ID[len(block.ID) - 1]) == 0:
                                os.makedirs(sat.orbPlane.earth.outputPath + '/pictures/',
                                            exist_ok=True)  # create output path
                                outputPath = sat.orbPlane.earth.outputPath + '/pictures/' + block.ID + '_' + str(
                                    len(block.QPath)) + '_'
                                # plotShortestPath(sat.orbPlane.earth, pathPlot, outputPath)
                                plotShortestPath(sat.orbPlane.earth, pathPlot, outputPath, ID=block.ID,
                                                 time=block.creationTime)
                        #################################################################

                        # path = block.QPath  # if there is Q-Learning the path will be repalced with the QPath
                        index += 1

                # down link buffers
                index = 0
                while index < len(sat.sendBufferGT[1]):
                    block = sat.sendBufferGT[1][index]

                    if len(block.QPath) > 3:  # the block does not come from a gateway
                        if sat.QLearning:
                            nextHop = sat.QLearning.makeAction(block, sat,
                                                               sat.orbPlane.earth.gateways[0].graph,
                                                               sat.orbPlane.earth, prevSat=(
                                    findByID(sat.orbPlane.earth, block.QPath[len(block.QPath) - 3][0])))
                        elif sat.DDQNA is not None:
                            nextHop = sat.DDQNA.makeDeepAction(block, sat,
                                                               sat.orbPlane.earth.gateways[
                                                                   0].graph,
                                                               sat.orbPlane.earth, prevSat=(
                                    findByID(sat.orbPlane.earth, block.QPath[len(block.QPath) - 3][0])))
                        else:
                            nextHop = sat.orbPlane.earth.DDQNA.makeDeepAction(block, sat,
                                                                              sat.orbPlane.earth.gateways[
                                                                                  0].graph,
                                                                              sat.orbPlane.earth, prevSat=(
                                    findByID(sat.orbPlane.earth, block.QPath[len(block.QPath) - 3][0])))
                    else:
                        if sat.QLearning:
                            nextHop = sat.QLearning.makeAction(block, sat,
                                                               sat.orbPlane.earth.gateways[0].graph,
                                                               sat.orbPlane.earth)
                        elif sat.DDQNA is not None:
                            nextHop = sat.DDQNA.makeDeepAction(block, sat,
                                                               sat.orbPlane.earth.gateways[
                                                                   0].graph,
                                                               sat.orbPlane.earth)
                        else:
                            nextHop = sat.orbPlane.earth.DDQNA.makeDeepAction(block, sat,
                                                                              sat.orbPlane.earth.gateways[
                                                                                  0].graph,
                                                                              sat.orbPlane.earth)

                    if nextHop != 0:
                        block.QPath[-2] = nextHop
                        pathPlot = block.QPath.copy()
                        pathPlot.pop()
                    else:
                        pathPlot = block.QPath.copy()

                    # If plotPath plots an image for every action taken. Prints 1/10 of blocks. # ANCHOR plot action earth 3
                    #################################################################
                    if sat.orbPlane.earth.plotPaths:
                        if int(block.ID[len(block.ID) - 1]) == 0:
                            os.makedirs(sat.orbPlane.earth.outputPath + '/pictures/',
                                        exist_ok=True)  # create output path
                            outputPath = sat.orbPlane.earth.outputPath + '/pictures/' + block.ID + '_' + str(
                                len(block.QPath)) + '_'
                            # plotShortestPath(sat.orbPlane.earth, pathPlot, outputPath)
                            plotShortestPath(sat.orbPlane.earth, pathPlot, outputPath, ID=block.ID,
                                             time=block.creationTime)
                    #################################################################

                    # path = block.QPath  # if there is Q-Learning the path will be repalced with the QPath
                    index += 1

                # find neighboring satellites
                neighbors = list(nx.neighbors(graph, sat.ID))
                itt = 0
                neighborSatsInter = []
                for sat2 in sats:
                    if sat2.ID in neighbors:
                        # we only care about the satellite if it is an inter-plane ISL
                        # we assume intra-plane ISLs will not change
                        if sat2.in_plane != sat.in_plane:
                            dataRate = nx.path_weight(graph, [sat2.ID, sat.ID], "dataRateOG")
                            distance = nx.path_weight(graph, [sat2.ID, sat.ID], "slant_range")
                            neighborSatsInter.append((distance, sat2, dataRate))
                        itt += 1
                        if itt == len(neighbors):
                            break
                sat.interSats = neighborSatsInter
                # list of blocks to be redistributed
                blocksToDistribute = []

                ### inter-plane ISLs ###

                sat.newBuffer = [True for _ in range(len(neighborSatsInter))]

                # make a list of False entries for each current neighbor
                sameSats = [False for _ in range(len(neighborSatsInter))]

                buffers = [None for _ in range(len(neighborSatsInter))]
                processes = [None for _ in range(len(neighborSatsInter))]

                # go through each process/buffer
                #   - check if the satellite is still there:
                #       - if it is, change the corresponding False to True, handle blocks and add process and buffer references to temporary list
                #       - if it is not, remove blocks from buffer and stop process
                for bufferIndex, buffer in enumerate(sat.sendBufferSatsInter):
                    # check if the satellite is still there
                    isPresent = False
                    for neighborIndex, neighbor in enumerate(neighborSatsInter):
                        if buffer[2] == neighbor[1].ID:
                            isPresent = True
                            sameSats[neighborIndex] = True

                            ## handle blocks
                            # check if there are blocks in the buffer
                            if buffer[1]:
                                # find index of satellite in block's path
                                index = None
                                for i, step in enumerate(buffer[1][0].QPath):
                                    if sat.ID == step[0]:
                                        index = i
                                        break

                                # check if next step in path corresponds to buffer's satellite
                                if buffer[1][0].QPath[index + 1][0] == buffer[2]:
                                    # add all but the first block to redistribution list
                                    for block in buffer[1][1:]:
                                        blocksToDistribute.append((block.checkPoints[-1], block))

                                    # add buffer with only first block present to temp list
                                    buffers[neighborIndex] = (
                                        [sat.env.event().succeed()], [sat.sendBufferSatsInter[bufferIndex][1][0]],
                                        buffer[2])
                                    processes[neighborIndex] = sat.sendBlocksSatsInter[bufferIndex]
                                else:
                                    # add all blocks to redistribution list
                                    for block in buffer[1]:
                                        blocksToDistribute.append((block.checkPoints[-1], block))
                                    # reset buffer
                                    buffers[neighborIndex] = ([sat.env.event()], [], buffer[2])

                                    # reset process
                                    sat.sendBlocksSatsInter[bufferIndex].interrupt()
                                    processes[neighborIndex] = sat.env.process(sat.sendBlock(neighbor, True, False))

                            else:  # there are no blocks in the buffer
                                # add buffer and remake process
                                buffers[neighborIndex] = sat.sendBufferSatsInter[bufferIndex]
                                sat.sendBlocksSatsInter[bufferIndex].interrupt()
                                processes[neighborIndex] = sat.env.process(sat.sendBlock(neighbor, True, False))
                                # sendBlocksSatsInter[bufferIndex]

                            break
                    if not isPresent:
                        # add blocks to redistribution list
                        for block in buffer[1]:
                            blocksToDistribute.append((block.checkPoints[-1], block))
                        # stop process
                        sat.sendBlocksSatsInter[bufferIndex].interrupt()

                # make buffer and process for new neighbors(s)
                # - go through list of previously false entries:
                #   - check  entry for each neighbor:
                #       - if False, create buffer and process for new neighbor
                # - clear temporary list of processes and buffers
                for entryIndex, entry in enumerate(sameSats):
                    if not entry:
                        buffers[entryIndex] = ([sat.env.event()], [], neighborSatsInter[entryIndex][1].ID)
                        processes[entryIndex] = sat.env.process(
                            sat.sendBlock(neighborSatsInter[entryIndex], True, False))

                # overwrite buffers and processes
                sat.sendBlocksSatsInter = processes
                sat.sendBufferSatsInter = buffers

                ### intra-plane ISLs ###
                # check blocks for each buffer
                for bufferIndex, buffer in enumerate(sat.sendBufferSatsIntra):
                    ## handle blocks
                    # check if there are blocks in the buffer
                    if buffer[1]:
                        # find index of satellite in block's path
                        index = None
                        for i, step in enumerate(buffer[1][0].QPath):
                            if sat.ID == step[0]:
                                index = i
                                break

                        # check if next step in path corresponds to buffer's satellite
                        if buffer[1][0].QPath[index + 1][0] == buffer[2]:
                            # add all but the first block to redistribution list
                            for block in buffer[1][1:]:
                                blocksToDistribute.append((block.checkPoints[-1], block))

                            # remove all but the first block and event from the buffer
                            length = len(sat.sendBufferSatsIntra[bufferIndex][1]) - 1
                            for _ in range(length):
                                sat.sendBufferSatsIntra[bufferIndex][1].pop(1)
                                sat.sendBufferSatsIntra[bufferIndex][0].pop(1)

                        else:
                            # add all blocks to redistribution list
                            for block in buffer[1]:
                                blocksToDistribute.append((block.checkPoints[-1], block))
                            # reset buffer
                            sat.sendBufferSatsIntra[bufferIndex] = ([sat.env.event()], [], buffer[2])

                            # reset process
                            sat.sendBlocksSatsIntra[bufferIndex].interrupt()
                            sat.sendBlocksSatsIntra[bufferIndex] = sat.env.process(
                                sat.sendBlock(sat.intraSats[bufferIndex], True, True))

                ### GSL ###
                # check if satellite has a linked GT
                if sat.linkedGT is not None:
                    sat.adjustDownRate()

                    # check if it had a sendBlocksGT process
                    if sat.sendBlocksGT:
                        # check if there are any blocks in the buffer
                        if sat.sendBufferGT[1]:
                            # check if linked GT is the same as the destination of first block in sendBufferGT
                            if sat.sendBufferGT[1][0].destination != sat.linkedGT:
                                sat.sendBlocksGT[0].interrupt()
                                sat.sendBlocksGT = []

                                # remove blocks from queue and add to list of blocks which should be redistributed
                                for block in sat.sendBufferGT[1]:
                                    blocksToDistribute.append(
                                        (block.checkPoints[-1], block))  # (latest checkpoint time, block)
                                sat.sendBufferGT = ([sat.env.event()], [])

                                # make new send process for new linked GT
                                sat.sendBlocksGT.append(
                                    sat.env.process(sat.sendBlock((sat.GTDist, sat.linkedGT), False)))
                            else:
                                # keep the first block in the buffer and let process continue
                                for block in sat.sendBufferGT[1][1:]:
                                    blocksToDistribute.append(
                                        (block.checkPoints[-1], block))  # (latest checkpoint time, block)
                                length = len(sat.sendBufferGT[1]) - 1
                                for _ in range(length):
                                    sat.sendBufferGT[1].pop(1)  # pop all but the first block
                                    sat.sendBufferGT[0].pop(1)  # pop all but the first event

                        else:  # there are no blocks in the buffer
                            sat.sendBlocksGT[0].interrupt()
                            sat.sendBlocksGT = []
                            sat.sendBufferGT = ([sat.env.event()], [])
                            # make new send process for new linked GT
                            sat.sendBlocksGT.append(sat.env.process(sat.sendBlock((sat.GTDist, sat.linkedGT), False)))

                    else:  # it had no process running
                        # there should be no blocks in the GT buffer, but just in case - if there are none, then the for loop will not run
                        # remove blocks from queue and add to list of blocks which should be redistributed
                        for block in sat.sendBufferGT[1]:
                            blocksToDistribute.append((block.checkPoints[-1], block))  # (latest checkpoint time, block)
                        sat.sendBufferGT = ([sat.env.event()], [])

                        # make new send process for new linked GT
                        sat.sendBlocksGT.append(sat.env.process(sat.sendBlock((sat.GTDist, sat.linkedGT), False)))

                else:  # no linked GT
                    # check if there is a sendBlocksGT process
                    if sat.sendBlocksGT:
                        sat.sendBlocksGT[0].interrupt()
                        sat.sendBlocksGT = []

                        # remove blocks from queue and add to list of blocks which should be redistributed
                        for block in sat.sendBufferGT[1]:
                            blocksToDistribute.append((block.checkPoints[-1], block))  # (latest checkpoint time, block)
                        sat.sendBufferGT = ([sat.env.event()], [])

                # sort blocks by arrival time at satellite
                try:
                    blocksToDistribute.sort()
                except Exception as e:
                    print(f"Caught an exception: {e}")
                    print(f'Something wrong with: \n{blocksToDistribute}')
                # add blocks to the correct queues based on next step in their path
                # since the blocks list is sorted by arrival time, the order in the new queues is correct
                for block in blocksToDistribute:
                    # get this satellite's index in the blocks path
                    index = None
                    for i, step in enumerate(block[1].QPath):
                        if sat.ID == step[0]:
                            index = i

                    # check if next step in path is GT (last step in path)
                    if index is None:
                        print(
                            f'Satellite {sat.ID} not found in the QPath: {block[1].QPath}')  # FIXME This should not happen. Debugging I realized when this happens the previous satellite is twice in last positions of QPath, instead of prevSat and currentSat. The current sat was the linked to the gateways bu after the movement it is not anymore.
                        self.lostBlocks += 1
                    elif index == len(block[1].QPath) - 2:
                        # add block to GT send-buffer
                        if not sat.sendBufferGT[0][0].triggered:
                            sat.sendBufferGT[0][0].succeed()
                            sat.sendBufferGT[1].append(block[1])
                        else:
                            newEvent = sat.env.event().succeed()
                            sat.sendBufferGT[0].append(newEvent)
                            sat.sendBufferGT[1].append(block[1])
                    else:
                        # get ID of next sat and find if it is intra or inter
                        ID = None
                        isIntra = False
                        for neighborSat in sat.intraSats:
                            id = neighborSat[1].ID
                            if id == block[1].QPath[index + 1][0]:
                                ID = neighborSat[1].ID
                                isIntra = True
                        for neighborSat in sat.interSats:
                            id = neighborSat[1].ID
                            if id == block[1].QPath[index + 1][0]:
                                ID = neighborSat[1].ID

                        if ID is not None:
                            sendBuffer = None
                            # find send-buffer for the satellite
                            if isIntra:
                                for buffer in sat.sendBufferSatsIntra:
                                    if ID == buffer[2]:
                                        sendBuffer = buffer
                            else:
                                for buffer in sat.sendBufferSatsInter:
                                    if ID == buffer[2]:
                                        sendBuffer = buffer

                            # add block to buffer
                            if not sendBuffer[0][0].triggered:
                                sendBuffer[0][0].succeed()
                                sendBuffer[1].append(block[1])
                            else:
                                newEvent = sat.env.event().succeed()
                                sendBuffer[0].append(newEvent)
                                sendBuffer[1].append(block[1])
                        else:
                            print("buffer for next satellite in path could not be found")

    def updateGTPaths(self):
        """
        Updates all paths for all GTs going to all other GTs and ensures that all blocks waiting to be sent has the
        correct path.
        """
        # make new paths for all GTs
        for GT in self.gateways:
            for destination in self.gateways:
                if GT != destination:
                    if destination.linkedSat[0] is not None and GT.linkedSat[0] is not None:
                        # Check if both nodes exist in the graph before calculating path
                        if GT.name in GT.graph and destination.name in GT.graph:
                            path = getShortestPath(GT.name, destination.name, self.pathParam, GT.graph)
                        else:
                            print(f"Warning: Gateway nodes not found in graph: {GT.name} or {destination.name}")
                            path = []
                    else:
                        path = []
                        # No path from gateway

                    GT.paths.update({destination.name: path})

            # update paths for all blocks in send-buffer
            for block in GT.sendBuffer[1]:
                if block.destination.name in GT.paths:
                    block.path = GT.paths[block.destination.name]
                    block.isNewPath = True
                    if block.path and block.path != -1:  # Check for valid path
                        block.QPath = [block.path[0], block.path[1], block.path[len(block.path) - 1]]
                    else:
                        block.QPath = []

    def getGSLDataRates(self):
        upDataRates = []
        downDataRates = []
        for GT in self.gateways:
            if GT.linkedSat[0] is not None:
                upDataRates.append(GT.dataRate)

        for orbit in self.LEO:
            for satellite in orbit.sats:
                if satellite.linkedGT is not None:
                    downDataRates.append(satellite.downRate)

        return upDataRates, downDataRates

    def getISLDataRates(self):
        interDataRates = []
        highRates = 0
        for orbit in self.LEO:
            for satellite in orbit.sats:
                for satData in satellite.interSats:
                    if satData[2] > 3e9:
                        highRates += 1
                    interDataRates.append(satData[2])
        return interDataRates

    def moveConstellation(self, env, deltaT=3600, getRates=False):
        """
        Simpy process function:

        Moves the constellations in terms of the Earth's rotation and moves the satellites within the constellations.
        The movement is based on the time that has passed since last constellation movement and is defined by the
        "deltaT" variable.

        After the satellites have been moved a process of re-linking all links, both GSLs and ISLs, is conducted where
        the paths for all blocks are re-made, the blocks are moved (if necessary) to the correct buffers, and all
        processes managing the send-buffers are checked to ensure they will still work correctly.
        """

        # Get the data rate for a intra plane ISL - used for testing
        if getRates:
            intraRate.append(self.LEO[0].sats[0].intraSats[0][2])

        while True:
            print('Creating/Moving constellation: Updating satellites position and links.')
            if getRates:
                # get data rates for all inter plane ISLs and all GSLs (up and down) - used for testing
                upDataRates, downDataRates = self.getGSLDataRates()
                inter = self.getISLDataRates()

                for val in upDataRates:
                    upGSLRates.append(val)

                for val in downDataRates:
                    downGSLRates.append(val)

                for val in inter:
                    interRates.append(val)

            yield env.timeout(deltaT)

            # clear satellite references on all GTs
            for GT in self.gateways:
                GT.satsOrdered = []
                GT.linkedSat = (None, None)

            # rotate constellation and satellites
            for plane in self.LEO:
                plane.rotate(ndeltas * deltaT)

            # relink satellites and GTs
            self.linkSats2GTs("Optimize")

            # create new graph and add references to all GTs for every rotation
            # prevGraph = self.graph
            graph = createGraph(self, matching=matching)
            self.graph = graph
            self.space_graph = graph  # Update space_graph as well
            for GT in self.gateways:
                GT.graph = graph

            if self.pathParam == 'Deep Q-Learning' or self.pathParam == 'Q-Learning':
                self.updateSatelliteProcessesRL(graph)
            else:
                self.updateSatelliteProcessesCorrect(graph)

            self.updateGTPaths()

            # Recalculate hybrid paths between terrestrial nodes after constellation movement
            # This uses the same logic as initialization
            if hasattr(self, 'active_terrestrial_nodes') and self.active_terrestrial_nodes:
                print("Recalculating hybrid paths between terrestrial nodes...")
                for src in self.active_terrestrial_nodes:
                    for dst in self.active_terrestrial_nodes:
                        if src != dst:
                            # Use the same logic as initialization: compare terrestrial vs hybrid
                            p_terr = getShortestPathTerrestrial(src.name, dst.name, self.terr_graph)
                            p_hyb = compute_hybrid_path(src.name, dst.name, self, prefer="latency")

                            terr_cost = estimate_path_cost(self, p_terr) if p_terr else float("inf")
                            hyb_cost = estimate_path_cost(self, p_hyb) if p_hyb else float("inf")

                            # Choose the better path (same logic as initialization)
                            if p_hyb and hyb_cost < terr_cost:
                                chosen_path = coerce_path_only(p_hyb)
                                chosen_type = "hyb"
                                print(
                                    f"[MOVEMENT] Updated hybrid path: {src.name} -> {dst.name} (length: {len(chosen_path)}, cost: {hyb_cost:.6f} vs terrestrial: {terr_cost:.6f})")
                            else:
                                chosen_path = coerce_path_only(p_terr)
                                chosen_type = "terr"
                                print(
                                    f"[MOVEMENT] Updated terrestrial path: {src.name} -> {dst.name} (length: {len(chosen_path)}, cost: {terr_cost:.6f} vs hybrid: {hyb_cost:.6f})")

                            src.paths[dst.name] = chosen_path
                            src.path_types[dst.name] = chosen_type if chosen_path else "none"
            self.nMovs += 1
            if saveISLs:
                print('Constellation moved! Saving ISLs map...')
                islpath = outputPath + '/ISL_maps/'
                os.makedirs(islpath, exist_ok=True)
                self.plotMap(plotGT=True, plotSat=True, edges=True, save=True, outputPath=islpath, n=self.nMovs)
                plt.close()

            try:
                if hasattr(self, 'active_terrestrial_nodes') and len(self.active_terrestrial_nodes) >= 2:
                    src_name = self.active_terrestrial_nodes[0].name
                    dst_name = self.active_terrestrial_nodes[1].name

                    # Usa il path effettivo che seguono i blocchi nella simulazione
                    path_nodes = None
                    path_type = "none"

                    try:
                        # Calcola sempre il path ibrido per vedere come cambia con il movimento
                        p_terr = getShortestPathTerrestrial(src_name, dst_name, self.terr_graph)
                        terr_cost = estimate_path_cost(self, p_terr) if p_terr else float('inf')

                        p_hyb = compute_hybrid_path(src_name, dst_name, self)
                        hyb_cost = estimate_path_cost(self, p_hyb) if p_hyb else float('inf')

                        if terr_cost <= hyb_cost and p_terr:
                            path_nodes = p_terr
                            path_type = "terrestrial"
                            print(
                                f'[Movement {self.nMovs}] Using TERRESTRIAL path: {len(path_nodes)} nodes (cost: {terr_cost:.4f})')
                        elif p_hyb:
                            path_nodes = p_hyb
                            path_type = "hybrid"
                            print(
                                f'[Movement {self.nMovs}] Using HYBRID path: {len(path_nodes)} nodes (cost: {hyb_cost:.4f})')
                        else:
                            path_nodes = []
                            print(f'[Movement {self.nMovs}] No valid path found')

                        # Check if path changed compared to previous movement
                        if hasattr(self, 'last_path') and self.last_path and path_nodes:
                            if path_nodes != self.last_path:
                                pass  # Path changed

                        # Salva il path per il confronto successivo
                        self.last_path = path_nodes.copy() if path_nodes else []

                    except Exception as e:
                        path_nodes = []

                    # Salva il path se trovato
                    if path_nodes:
                        path_maps_dir = outputPath + '/path_maps/'
                        os.makedirs(path_maps_dir, exist_ok=True)
                        output_file = path_maps_dir + f"path_movement_{self.nMovs}_{src_name}_to_{dst_name}_{path_type}.png"
                        plotPathClean(self, path_nodes, src_name, dst_name, output_file)
            except Exception as e:
                pass

            # Perform Federated Learning
            if FL_Test:
                global const_moved
                const_moved = True
                CKA_before, CKA_after = perform_FL(self)  # , outputPath)
                self.CKA.append([CKA_before, CKA_after, env.now])

        # Test functions removed - not needed for production
        totalFailed = 0

        for GT in self.gateways[1:]:
            failed = False
            path = getShortestPath(self.gateways[0].name, GT.name, self.pathParam, graph)
            try:
                firstStep = GT.linkedSat[0]
            except KeyError:
                firstStep = edgeWeights[(path[1][0], path[0][0])]
                print(f'Keyerror in: {GT.name}')

            for index in range(1, len(path) - 2):
                try:
                    if edgeWeights[(path[index][0], path[index + 1][0])] > firstStep:
                        failed = True
                except KeyError:
                    print(f'Keyerror 2 in: {GT.name}')
                    if edgeWeights[(path[index + 1][0], path[index][0])] > firstStep:
                        failed = True
            if failed:
                print("{} could not create a path which adheres to flow constraints".format(GT.name))
                totalFailed += 1

        print("number of GT paths that cannot meet flow restraints: {}".format(totalFailed))

    def plotMap(self, plotGT=True, plotSat=True, path=None, bottleneck=None,
                save=False, ID=None, time=None, edges=False, arrow_gap=0.008,
                outputPath='', paths=None, fileName="map.png", n=None):

        def _normalize_path_for_plot(path_in):
            norm = []
            if not path_in:
                return norm

            def _resolve_coords_by_key(key):
                # 1) usa getNodeByName se disponibile
                node_fn = getattr(self, "getNodeByName", None)
                node = node_fn(key) if callable(node_fn) else None

                # If not found and key is string that looks like satellite ID (e.g. "6_3"), try to convert
                if node is None and isinstance(key, str) and '_' in key:
                    try:
                        # Prova a convertire in int per satelliti
                        parts = key.split('_')
                        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                            # Per satelliti, cerca direttamente nei piani LEO
                            for plane in self.LEO:
                                for sat in plane.sats:
                                    if str(sat.ID) == key:
                                        node = sat
                                        break
                                if node:
                                    break
                    except Exception:
                        pass

                if node is not None:
                    lon = getattr(node, "longitude", getattr(node, "lon", None))
                    lat = getattr(node, "latitude", getattr(node, "lat", None))
                    if isinstance(lon, (int, float)) and abs(lon) <= math.pi + 1e-6:
                        lon = math.degrees(lon)
                    if isinstance(lat, (int, float)) and abs(lat) <= (math.pi / 2) + 1e-6:
                        lat = math.degrees(lat)
                    if lon is not None and lat is not None:
                        name = getattr(node, "name", getattr(node, "ID", str(key)))
                        return (str(name), float(lon), float(lat))
                    else:
                        pass
                else:
                    pass

                # 2) grafo terrestre
                if hasattr(self, "terr_graph") and self.terr_graph is not None and self.terr_graph.has_node(key):
                    nd = self.terr_graph.nodes[key]
                    lon = nd.get("lon");
                    lat = nd.get("lat")
                    if lon is not None and lat is not None:
                        return (str(key), float(lon), float(lat))

                if hasattr(self, "space_graph") and self.space_graph is not None and self.space_graph.has_node(key):
                    nd = self.space_graph.nodes[key]
                    lon = nd.get("lon");
                    lat = nd.get("lat")
                    if lon is not None and lat is not None:
                        # print(f"[plotMap DEBUG] Resolved {key} via space_graph at ({lon:.2f}, {lat:.2f})")
                        return (str(key), float(lon), float(lat))
                return None

            for hop in path_in:
                if isinstance(hop, (list, tuple)):
                    if len(hop) >= 3 and isinstance(hop[1], (int, float)) and isinstance(hop[2], (int, float)):
                        norm.append((str(hop[0]), float(hop[1]), float(hop[2])))
                        continue
                    key = hop[0]
                elif hasattr(hop, "name") or hasattr(hop, "ID"):
                    key = getattr(hop, "name", None) or getattr(hop, "ID", None)
                    lon = getattr(hop, "longitude", getattr(hop, "lon", None))
                    lat = getattr(hop, "latitude", getattr(hop, "lat", None))
                    if lon is not None and lat is not None:
                        if isinstance(lon, (int, float)) and abs(lon) <= math.pi + 1e-6:
                            lon = math.degrees(lon)
                        if isinstance(lat, (int, float)) and abs(lat) <= (math.pi / 2) + 1e-6:
                            lat = math.degrees(lat)
                        name = getattr(hop, "name", getattr(hop, "ID", str(key)))
                        norm.append((str(name), float(lon), float(lat)))
                        continue
                else:
                    key = hop

                res = _resolve_coords_by_key(key)
                if res is not None:
                    norm.append(res)

            dedup = []
            for t in norm:
                if not dedup or (dedup[-1][1], dedup[-1][2]) != (t[1], t[2]):
                    dedup.append(t)
            return dedup

        norm_path = _normalize_path_for_plot(path) if path else []

        # Debug: stampa informazioni sul path
        if path:

            satellite_ids = [str(sat.ID) for plane in self.LEO for sat in plane.sats]

            terrestrial_node_names = []
            if hasattr(self, 'terr_graph') and self.terr_graph:
                terrestrial_node_names = list(self.terr_graph.nodes())

            for i, hop in enumerate(norm_path[:5]):
                is_sat = hop[0] in satellite_ids
                is_terr = hop[0] in terrestrial_node_names
        else:
            pass

        if paths is None or (hasattr(paths, '__len__') and len(paths) == 0) or (
                isinstance(paths, np.ndarray) and paths.size == 0):
            plt.figure()
        else:
            plt.figure(figsize=(6, 3))

        legend_properties = {'size': 10, 'weight': 'bold'}
        markerscale = 1.5
        usage_threshold = 10  # percent

        # Compute link usage per congestion map
        def calculate_link_usage(paths_in):
            link_usage = {}
            for p in paths_in:
                # normalizza anche qui per sicurezza
                npth = _normalize_path_for_plot(p)
                for i in range(len(npth) - 1):
                    s, e = npth[i], npth[i + 1]
                    link_str = f'{s[0]}_{e[0]}'
                    coordinates = [(s[1], s[2]), (e[1], e[2])]
                    if link_str in link_usage:
                        link_usage[link_str]['count'] += 1
                    else:
                        link_usage[link_str] = {'count': 1, 'coordinates': coordinates}
            return link_usage

        # Regola inizio/fine frecce
        def adjust_arrow_points(start, end, gap_value):
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            dist = math.sqrt(dx ** 2 + dy ** 2)
            if dist == 0:
                return start, end
            gap_scaled = gap_value * 1440  # la mappa usa [0..1440]x[0..720]
            new_start = (start[0] + gap_scaled * dx / dist, start[1] + gap_scaled * dy / dist)
            new_end = (end[0] - gap_scaled * dx / dist, end[1] - gap_scaled * dy / dist)
            return new_start, new_end

        # --- plot ISL + GSL edges ------------------------------------------------
        if edges:
            if n is not None:
                fileName = os.path.join(outputPath, f"ISLs_map_{n}.png")
            else:
                fileName = os.path.join(outputPath, "ISLs_map.png")

            for plane in self.LEO:
                for sat in plane.sats:
                    orig_start_x = int((0.5 + math.degrees(sat.longitude) / 360.0) * 1440)
                    orig_start_y = int((0.5 - math.degrees(sat.latitude) / 180.0) * 720)

                    for connected_sat in sat.intraSats + sat.interSats:
                        orig_end_x = int((0.5 + math.degrees(connected_sat[1].longitude) / 360.0) * 1440)
                        orig_end_y = int((0.5 - math.degrees(connected_sat[1].latitude) / 180.0) * 720)
                        adj_start, adj_end = adjust_arrow_points(
                            (orig_start_x, orig_start_y),
                            (orig_end_x, orig_end_y),
                            arrow_gap
                        )
                        plt.arrow(adj_start[0], adj_start[1],
                                  adj_end[0] - adj_start[0], adj_end[1] - adj_start[1],
                                  shape='full', lw=0.5, length_includes_head=True, head_width=5)

            # GSL: gateway->satelliti (solo se linkati)
            for GT in getattr(self, "gateways", []):
                if getattr(GT, "linkedSat", None) and GT.linkedSat[1]:
                    gt_x = GT.gridLocationX
                    gt_y = GT.gridLocationY
                    sat_x = int((0.5 + math.degrees(GT.linkedSat[1].longitude) / 360.0) * 1440)
                    sat_y = int((0.5 - math.degrees(GT.linkedSat[1].latitude) / 180.0) * 720)
                    _, adj_end = adjust_arrow_points((gt_x, gt_y), (sat_x, sat_y), arrow_gap)
                    plt.arrow(gt_x, gt_y,
                              adj_end[0] - gt_x, adj_end[1] - gt_y,
                              shape='full', lw=0.5, length_includes_head=True, head_width=5)

        # --- plot satellites / gateways -----------------------------------------
        scat1 = None
        scat2 = None

        path_gates = set()
        if norm_path:
            for hop in norm_path:
                try:
                    node = self.getNodeByName(hop[0])
                    if isinstance(node, Gateway):
                        path_gates.add(str(node.name))
                except Exception:
                    continue

        # plotSat={plotSat}, plotGT={plotGT}
        if plotSat:
            # ENTERING SATELLITE SECTION
            # Visualizzazione satellitare ORIGINALE - griglia colorata come nell'immagine
            colors = cm.rainbow(np.linspace(0, 1, len(self.LEO))) if len(self.LEO) > 0 else [(0, 0, 0, 1)]
            plotSatID = globals().get('plotSatID', False)

            for plane, c in zip(self.LEO, colors):
                for sat in plane.sats:
                    gridSatX = int((0.5 + math.degrees(sat.longitude) / 360.0) * 1440)
                    gridSatY = int((0.5 - math.degrees(sat.latitude) / 180.0) * 720)

                    # Satelliti colorati come nell'originale
                    scat2 = plt.scatter(gridSatX, gridSatY, marker='o', s=15, linewidth=0.4,
                                        edgecolors='black', color=c, alpha=0.8,
                                        label='Satellites' if sat.ID == plane.sats[0].ID else "")

                    if plotSatID:
                        plt.text(gridSatX + 8, gridSatY - 8, str(sat.ID), fontsize=5, ha='left', va='center')

        if plotGT:
            if path_gates:
                for GT in getattr(self, "gateways", []):
                    if str(GT.name) in path_gates:
                        scat1 = plt.scatter(GT.gridLocationX, GT.gridLocationY,
                                            marker='x', c='r', s=28, linewidth=1.5, label=str(GT.name))
            else:
                for GT in getattr(self, "gateways", []):
                    scat1 = plt.scatter(GT.gridLocationX, GT.gridLocationY,
                                        marker='x', c='r', s=28, linewidth=1.5, label=str(GT.name))

        # --- disegna path se fornito (robusto a liste di nomi) -------------------
        if path:
            # print(f"[plotMap DEBUG] Drawing path: {len(norm_path)} normalized hops")
            if len(norm_path) >= 2:
                if bottleneck:
                    # Using bottleneck plotting mode
                    xValues = [[], [], []]
                    yValues = [[], [], []]
                    minimum = np.amin(bottleneck[1])
                    length = len(norm_path)
                    index = 0
                    arr = 0
                    minFound = False

                    while index < length:
                        hop = norm_path[index]
                        xValues[arr].append(int((0.5 + hop[1] / 360.0) * 1440))
                        yValues[arr].append(int((0.5 - hop[2] / 180.0) * 720))
                        if not minFound and index < len(bottleneck[1]) and bottleneck[1][index] == minimum:
                            arr += 1
                            hopN = norm_path[index]
                            hopN1 = norm_path[index + 1] if index + 1 < length else norm_path[index]
                            xValues[arr].append(int((0.5 + hopN[1] / 360.0) * 1440))
                            yValues[arr].append(int((0.5 - hopN[2] / 180.0) * 720))
                            xValues[arr].append(int((0.5 + hopN1[1] / 360.0) * 1440))
                            yValues[arr].append(int((0.5 - hopN1[2] / 180.0) * 720))
                            arr += 1
                            minFound = True
                        index += 1

                    plt.plot(xValues[0], yValues[0], 'b')
                    plt.plot(xValues[1], yValues[1], 'r')
                    plt.plot(xValues[2], yValues[2], 'b')
                else:
                    # Using normal plotting mode
                    # PATH PULITO E PROFESSIONALE - usa coordinate pixel come l'immagine satellitare
                    xValues = []
                    yValues = []
                    for hop in norm_path:
                        lon = hop[1] if len(hop) > 1 else 0
                        lat = hop[2] if len(hop) > 2 else 0

                        # Conversione sicura delle coordinate
                        px = int((0.5 + lon / 360.0) * 1440)
                        py = int((0.5 - lat / 180.0) * 720)

                        # Assicurati che i pixel siano nel range valido
                        px = max(0, min(1439, px))
                        py = max(0, min(719, py))

                        xValues.append(px)
                        yValues.append(py)

                    # Debug: stampa le prime 5 coordinate per capire il problema
                    print(f"[DEBUG] First 5 coordinates:")
                    for i, hop in enumerate(norm_path[:5]):
                        lon = hop[1] if len(hop) > 1 else 0
                        lat = hop[2] if len(hop) > 2 else 0
                        px = int((0.5 + lon / 360.0) * 1440)
                        py = int((0.5 - lat / 180.0) * 720)
                        print(f"  Hop {i}: {hop[0]} -> geo({lon:.2f}, {lat:.2f}) -> pixel({px}, {py})")
                        print(f"    Lon calc: (0.5 + {lon:.2f}/360) * 1440 = {0.5 + lon / 360.0:.4f} * 1440 = {px}")
                        print(f"    Lat calc: (0.5 - {lat:.2f}/180) * 720 = {0.5 - lat / 180.0:.4f} * 720 = {py}")

                    # print(f"[plotMap DEBUG] Plotting path with {len(xValues)} points")
                    # print(f"[plotMap DEBUG] X values (pixel): {xValues[:5]}...")
                    # print(f"[plotMap DEBUG] Y values (pixel): {yValues[:5]}...")

                    # Determine if it's satellite or terrestrial path
                    satellite_ids = [str(sat.ID) for plane in self.LEO for sat in plane.sats]
                    terrestrial_node_names = []
                    if hasattr(self, 'terr_graph') and self.terr_graph:
                        terrestrial_node_names = list(self.terr_graph.nodes())

                    # Conta quanti nodi sono satelliti vs terrestri
                    satellite_count = sum(1 for hop in norm_path if hop[0] in satellite_ids)
                    terrestrial_count = sum(1 for hop in norm_path if hop[0] in terrestrial_node_names)

                    # Se plotSat=False, forza il path come terrestre
                    if not plotSat:
                        is_satellite_path = False
                        # Forced terrestrial path because plotSat=False
                    else:
                        # If it contains satellites, it's a hybrid path (not purely terrestrial)
                        is_satellite_path = satellite_count > 0
                        # Path analysis: {satellite_count} satellite nodes, {terrestrial_count} terrestrial nodes -> is_satellite_path={is_satellite_path}

                    if is_satellite_path:
                        # Path satellitare - linea blu come nell'originale
                        plt.plot(xValues, yValues, color='blue', linewidth=3, alpha=0.8, zorder=5,
                                 label='Satellite Path')
                    else:
                        # Path terrestre - linea rossa sottile e elegante
                        plt.plot(xValues, yValues, color='red', linewidth=2, alpha=0.8, zorder=15,
                                 label='Terrestrial Path')

                # Nodi del path vengono disegnati nel loop successivo

                for i, hop in enumerate(norm_path):
                    # Usa coordinate pixel come l'immagine satellitare
                    px = int((0.5 + hop[1] / 360.0) * 1440)
                    py = int((0.5 - hop[2] / 180.0) * 720)
                    name = hop[0]

                    # Determine if it's satellite or terrestrial path for nodes
                    satellite_ids = [str(sat.ID) for plane in self.LEO for sat in plane.sats]
                    terrestrial_node_names = []
                    if hasattr(self, 'terr_graph') and self.terr_graph:
                        terrestrial_node_names = list(self.terr_graph.nodes())

                    # Conta quanti nodi sono satelliti vs terrestri
                    satellite_count = sum(1 for hop in norm_path if hop[0] in satellite_ids)
                    terrestrial_count = sum(1 for hop in norm_path if hop[0] in terrestrial_node_names)
                    is_satellite_path = satellite_count > terrestrial_count

                    if i == 0:
                        # START - punto verde GRANDE e visibile
                        if is_satellite_path:
                            plt.scatter(px, py, marker='o', c='green', s=100, linewidth=3,
                                        edgecolors='black', zorder=20, label='Start')
                        else:
                            # Terrestrial - even smaller node
                            plt.scatter(px, py, marker='o', c='green', s=50, linewidth=1,
                                        edgecolors='black', zorder=25, label='Start')
                            # Aggiungi etichetta per il nodo di partenza
                            plt.annotate(f'START: {name}', (px, py), xytext=(10, 10),
                                         textcoords='offset points', fontsize=8, fontweight='bold',
                                         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                    elif i == len(norm_path) - 1:
                        # END - punto rosso GRANDE e visibile
                        if is_satellite_path:
                            plt.scatter(px, py, marker='o', c='red', s=100, linewidth=3,
                                        edgecolors='black', zorder=20, label='End')
                        else:
                            # Terrestrial - even smaller node
                            plt.scatter(px, py, marker='o', c='red', s=50, linewidth=1,
                                        edgecolors='black', zorder=25, label='End')
                            # Aggiungi etichetta per il nodo di arrivo
                            plt.annotate(f'END: {name}', (px, py), xytext=(10, 10),
                                         textcoords='offset points', fontsize=8, fontweight='bold',
                                         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                    else:
                        # HOP - cerchi blu GRANDI e visibili
                        if is_satellite_path:
                            plt.scatter(px, py, marker='o', c='blue', s=80, linewidth=2,
                                        edgecolors='white', zorder=15, label='Hop' if i == 1 else "")
                        else:
                            # Terrestrial - smaller and more elegant nodes
                            # Debug: print real geographic coordinates
                            lon_deg = hop[1] if len(hop) > 1 else 0
                            lat_deg = hop[2] if len(hop) > 2 else 0
                            plt.scatter(px, py, marker='o', c='yellow', s=40, linewidth=1,
                                        edgecolors='black', zorder=20, label='Hop' if i == 1 else "")
                            # Aggiungi etichetta per i nodi intermedi (solo per i primi 3)
                            if i <= 3:
                                plt.annotate(f'Hop {i}: {name}', (px, py), xytext=(5, 5),
                                             textcoords='offset points', fontsize=6,
                                             bbox=dict(boxstyle='round,pad=0.2', facecolor='lightblue', alpha=0.7))
            else:
                if norm_path:
                    pass

        # --- Congestion map (optional, if paths is not None) ---------------------
        if paths is not None and (hasattr(paths, '__len__') and len(paths) > 0) and not (
                isinstance(paths, np.ndarray) and paths.size == 0):
            # se usi Q-Learning potresti avere block.QPath, ma qui ci aspettiamo una lista di path
            link_usage = calculate_link_usage(paths)

            try:
                max_usage = max(info['count'] for info in link_usage.values())
                # min_usage non usato direttamente: lasciamo un floor al 10%
                # min_usage = max_usage * 0.1
            except ValueError:
                print("Error: No data available for plotting congestion map.")
                print('Link usage values:\n', list(link_usage.values()))
                return -1

            most_used_link = max(link_usage.items(), key=lambda x: x[1]['count'])
            print(f"Most used link: {most_used_link[0]}, Packets: {most_used_link[1]['count']}")

            norm = Normalize(vmin=usage_threshold, vmax=100)
            cmap = cm.get_cmap('cool')

            for link_str, info in link_usage.items():
                usage = info['count']
                usage_percentage = max(usage_threshold, (usage / max_usage) * 100.0)
                # Reduce arrow width to be more reasonable
                width = 0.3 + (usage_percentage / 100.0) * 1.2  # Max width 1.5 instead of 2.5
                color = cmap(norm(usage_percentage))
                coordinates = info['coordinates']

                orig_start_x = (0.5 + coordinates[0][0] / 360.0) * 1440
                orig_start_y = (0.5 - coordinates[0][1] / 180.0) * 720
                orig_end_x = (0.5 + coordinates[1][0] / 360.0) * 1440
                orig_end_y = (0.5 - coordinates[1][1] / 180.0) * 720

                (start_x, start_y), (end_x, end_y) = adjust_arrow_points(
                    (orig_start_x, orig_start_y), (orig_end_x, orig_end_y), arrow_gap
                )

                # Bezier per una curva leggera
                mid_x, mid_y = (start_x + end_x) / 2.0, (start_y + end_y) / 2.0
                ctrl_x, ctrl_y = mid_x + (end_y - start_y) / 10.0, mid_y - (end_x - start_x) / 5.0
                verts = [(start_x, start_y), (ctrl_x, ctrl_y), (end_x, end_y)]
                codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
                pth = Path(verts, codes)
                patch = FancyArrowPatch(path=pth, arrowstyle='-|>', color=color,
                                        linewidth=width, mutation_scale=2,
                                        zorder=0.5)  # Reduce mutation_scale from 5 to 2
                plt.gca().add_patch(patch)

            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            ticks = [10] + list(np.linspace(10, 100, num=5))
            plt.colorbar(sm, orientation='vertical', label='Relative Traffic Load (%)',
                         fraction=0.02, pad=0.04, ticks=[int(t) for t in ticks])

            plt.xticks([]);
            plt.yticks([])

        # --- LEGENDA PULITA E PROFESSIONALE -----------------------------------------
        if path and len(norm_path) >= 2:
            from matplotlib.patches import Patch
            from matplotlib.lines import Line2D

            # Determine if it's satellite or terrestrial path
            is_satellite_path = any(
                hop[0] in [str(sat.ID) for plane in self.LEO for sat in plane.sats] for hop in norm_path)

            if is_satellite_path:
                # Legend for satellite path - smaller and bottom left
                legend_elements = [
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=6,
                           markeredgecolor='black', markeredgewidth=1),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=6,
                           markeredgecolor='black', markeredgewidth=1),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=4,
                           markeredgecolor='white', markeredgewidth=1),
                    Line2D([0], [0], color='blue', linewidth=2)
                ]
                legend_labels = ['Start', 'End', 'Hop', 'Path']
            else:
                # Legenda per path terrestre
                legend_elements = [
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=6,
                           markeredgecolor='black', markeredgewidth=1),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=6,
                           markeredgecolor='black', markeredgewidth=1),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=4,
                           markeredgecolor='white', markeredgewidth=1),
                    Line2D([0], [0], color='red', linewidth=2)
                ]
                legend_labels = ['Start', 'End', 'Hop', 'Path']

            plt.legend(legend_elements, legend_labels, loc='lower left',
                       prop={'size': 8}, markerscale=0.8,
                       framealpha=0.8, fancybox=True, shadow=False)
        else:
            # Legenda standard per mappe generali
            if plotSat and plotGT and scat1 is not None and scat2 is not None:
                plt.legend([scat1, scat2], ['Gateways', 'Satellites'],
                           loc=3, prop=legend_properties, markerscale=markerscale)
            elif plotSat and scat2 is not None:
                plt.legend([scat2], ['Satellites'], loc=3, prop=legend_properties, markerscale=markerscale)
            elif plotGT and scat1 is not None:
                plt.legend([scat1], ['Gateways'], loc=3, prop=legend_properties, markerscale=markerscale)

        plt.xticks([]);
        plt.yticks([])

        # Sfondo mappa del mondo per path terrestri
        if path and len(norm_path) >= 2:
            # Se plotSat=False, forza il path come terrestre
            if not plotSat:
                is_satellite_path = False
            else:
                # Determine if it's terrestrial or satellite path
                is_satellite_path = any(
                    hop[0] in [str(sat.ID) for plane in self.LEO for sat in plane.sats] for hop in norm_path)

            if not is_satellite_path:
                # Per path terrestri: crea un background simile all'immagine satellitare
                # print(f"[plotMap DEBUG] Creating world map background similar to satellite image")
                try:
                    # Crea una mappa del mondo con colori simili all'immagine satellitare
                    world_map = np.ones((720, 1440, 3)) * 0.1  # Sfondo scuro

                    # Aggiungi continenti con colori simili all'immagine satellitare
                    for lat in range(720):
                        for lon in range(1440):
                            # Converti coordinate pixel in lat/lon
                            lat_deg = 90 - (lat / 720) * 180
                            lon_deg = (lon / 1440) * 360 - 180

                            # Continenti con colori simili all'immagine satellitare
                            is_land = False

                            # Nord America
                            if (lat_deg > 15 and lat_deg < 70 and lon_deg > -170 and lon_deg < -50):
                                is_land = True
                            # Sud America
                            elif (lat_deg > -60 and lat_deg < 15 and lon_deg > -85 and lon_deg < -35):
                                is_land = True
                            # Europa/Africa
                            elif (lat_deg > -35 and lat_deg < 70 and lon_deg > -25 and lon_deg < 40):
                                is_land = True
                            # Asia
                            elif (lat_deg > 10 and lat_deg < 70 and lon_deg > 40 and lon_deg < 180):
                                is_land = True
                            # Australia
                            elif (lat_deg > -50 and lat_deg < -10 and lon_deg > 110 and lon_deg < 180):
                                is_land = True

                            if is_land:
                                # Colori simili all'immagine satellitare: verde/giallo per terra
                                world_map[lat, lon] = [0.2, 0.6, 0.2]  # Verde scuro per terra
                            else:
                                # Colori simili all'immagine satellitare: blu/viola per oceano
                                world_map[lat, lon] = [0.1, 0.3, 0.6]  # Blu scuro per oceano

                    plt.imshow(world_map, extent=(-180, 180, -90, 90), alpha=0.8, zorder=0)
                    # print(f"[plotMap DEBUG] World map background added (similar to satellite)")
                except Exception as e:
                    pass

        # Heatmap utenti celle (se non stiamo tracciando 'paths')
        if paths is None or (hasattr(paths, '__len__') and len(paths) == 0) or (
                isinstance(paths, np.ndarray) and paths.size == 0):
            try:
                cell_users = np.array(self.getCellUsers()).transpose()
                plt.imshow(cell_users, norm=LogNorm(), cmap='viridis')
            except Exception:
                # fallback: niente heatmap
                pass
        else:
            plt.gca().invert_yaxis()

        # Improvement: more informative and clear titles (without emojis for compatibility)
        if time is not None and ID is not None:
            try:
                if path and len(norm_path) >= 2:
                    plt.title(f"Communication Path: {norm_path[0][0]} → {norm_path[-1][0]}\n"
                              f"Block ID: {ID} | Time: {time * 1000:.0f}ms | Hops: {len(norm_path)}",
                              fontsize=12, weight='bold', pad=20)
                else:
                    plt.title(f"Communication Path (ID: {ID}, Time: {time * 1000:.0f}ms)",
                              fontsize=12, weight='bold', pad=20)
            except Exception:
                plt.title(f"Communication Path (ID: {ID}, Time: {time * 1000:.0f}ms)",
                          fontsize=12, weight='bold', pad=20)
        elif path and len(norm_path) >= 2:
            try:
                # Se plotSat=False, forza il path come terrestre
                if not plotSat:
                    is_satellite_path = False
                else:
                    # Determine if it's satellite or terrestrial path
                    is_satellite_path = any(
                        hop[0] in [str(sat.ID) for plane in self.LEO for sat in plane.sats] for hop in norm_path)

                if is_satellite_path:
                    plt.title(f"Satellite Communication Path: {norm_path[0][0]} → {norm_path[-1][0]}\n"
                              f"Hops: {len(norm_path)} | Distance: {len(norm_path) - 1} satellite links",
                              fontsize=12, weight='bold', pad=20)
                else:
                    plt.title(f"Terrestrial Communication Path: {norm_path[0][0]} → {norm_path[-1][0]}\n"
                              f"Hops: {len(norm_path)} | Distance: {len(norm_path) - 1} terrestrial links",
                              fontsize=12, weight='bold', pad=20)
            except Exception:
                plt.title("Communication Path", fontsize=12, weight='bold', pad=20)
        else:
            plt.title("Satellite Network Map", fontsize=12, weight='bold', pad=20)

        # NO ZOOM - usa la vista completa come l'immagine satellitare

        # Always save the plot as PNG
        if not save:
            # If save is not specified, create a default filename
            if not fileName or fileName == "map.png":
                fileName = f"plot_{int(time.time())}.png"

        os.makedirs(os.path.dirname(fileName) if os.path.dirname(fileName) else '.', exist_ok=True)
        plt.tight_layout()
        plt.savefig(fileName, dpi=300, bbox_inches='tight', pad_inches=0.1)
        print(f"[plotMap] Plot saved as: {fileName}")

        # Non mostrare durante la simulazione, solo salvare
        # plt.show()

    def initializeQTables(self, NGT, hyperparams, g):
        '''
        QTables initialization at each satellite
        '''
        print('----------------------------------')

        # path = './Results/Q-Learning/qTablesImport/qTablesExport/' + str(NGT) + 'GTs/'
        path = tablesPath

        if importQVals:
            print('Importing Q-Tables from: ' + path)
        else:
            print('Initializing Q-tables...')

        i = 0
        for plane in self.LEO:
            for sat in plane.sats:
                i += 1
                if importQVals:
                    with open(path + sat.ID + '.npy', 'rb') as f:
                        qTable = np.load(f)
                    sat.QLearning = QLearning(NGT, hyperparams, self, g, sat, qTable=qTable)
                else:
                    sat.QLearning = QLearning(NGT, hyperparams, self, g, sat)

        if importQVals:
            print(str(i) + ' Q-Tables imported!')
        else:
            print(str(i) + ' Q-Tables created!')
        print('----------------------------------')

    def plot3D(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        xs = []
        ys = []
        zs = []
        xG = []
        yG = []
        zG = []
        for con in self.LEO:
            for sat in con.sats:
                xs.append(sat.x)
                ys.append(sat.y)
                zs.append(sat.z)
        ax.scatter(xs, ys, zs, marker='o')
        for GT in self.gateways:
            xG.append(GT.x)
            yG.append(GT.y)
            zG.append(GT.z)
        ax.scatter(xG, yG, zG, marker='^')
        plt.show()

    def __repr__(self):
        return 'total divisions in x = {}\n total divisions in y = {}\n total cells = {}\n window of operation ' \
               '(longitudes) = {}\n window of operation (latitudes) = {}'.format(
            self.total_x,
            self.total_y,
            self.total_cells,
            self.windowx,
            self.windowy)

    def build_lookups(self):
        """
        Crea dizionari di lookup per accedere ai nodi in modo uniforme.
        - self.node_by_name: nome ➜ oggetto (gateway o terrestrial node)
        - self.sat_by_id: id intero ➜ satellite
        """
        # gateway e terrestrial nodes
        self.node_by_name = {}
        for tn in self.terrestrial_nodes:
            self.node_by_name[tn.name] = tn
        for gw in self.gateways:
            self.node_by_name[gw.name] = gw

        # satelliti
        self.sat_by_id = {}
        for plane in self.LEO:
            for sat in plane.sats:
                self.sat_by_id[sat.ID] = sat

    def getNode(self, key):
        """
        Restituisce l'oggetto nodo a partire da:
        - int (id satellite) -> Satellite
        - str (nome gateway o terrestrial node) -> oggetto corrispondente
        - oggetto nodo già valido -> lo restituisce direttamente
        """
        if isinstance(key, int):
            return self.sat_by_id.get(key, None)
        elif isinstance(key, str):
            return self.node_by_name.get(key, None)
        else:
            return key


class hyperparam:
    def __init__(self, pathing):
        '''
        Hyperparameters of the Q-Learning model
        '''
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.ArriveR = ArriveReward
        self.w1 = w1
        self.w2 = w2
        self.w4 = w4
        self.again = againPenalty
        self.unav = unavPenalty
        self.pathing = pathing
        self.tau = tau
        self.updateF = updateF
        self.batchSize = batchSize
        self.bufferSize = bufferSize
        self.hardUpdate = hardUpdate == 1
        self.importQ = importQVals
        self.MAX_EPSILON = MAX_EPSILON
        self.MIN_EPSILON = MIN_EPSILON
        self.LAMBDA = LAMBDA
        self.plotPath = plotPath
        self.coordGran = coordGran
        self.ddqn = ddqn
        self.latBias = latBias
        self.lonBias = lonBias
        self.diff = diff
        self.explore = explore
        self.reducedState = reducedState
        self.online = onlinePhase

    def __repr__(self):
        return 'Hyperparameters:\nalpha: {}\ngamma: {}\nepsilon: {}\nw1: {}\nw2: {}\n'.format(
            self.alpha,
            self.gamma,
            self.epsilon,
            self.w1,
            self.w2)


# @profile
class QLearning:
    def __init__(self, NGT, hyperparams, earth, g, sat, qTable=None):
        '''
        Create a 6D numpy array to hold the current Q-values for each state and action pair: Q(s, a)
        The array contains 5 dimensions with the shape of the environment, as well as a 6th "action" dimension.
        The "action" dimension consists of 4 layers that will allow us to keep track of the Q-values for each possible action in each state
        The value of each (state, action) pair is initialized ranomly.
        '''
        satUp, satDown, satRight, satLeft = 3, 3, 3, 3
        linkedSats = getLinkedSats(sat, g, earth)
        self.linkedSats = {'U': linkedSats['U'],
                           'D': linkedSats['D'],
                           'R': linkedSats['R'],
                           'L': linkedSats['L']}

        self.actions = ('U', 'D', 'R', 'L')  # Up, Down, Left, Right
        self.Destinations = NGT

        self.nStates = satUp * satDown * satRight * satLeft * NGT
        self.nActions = len(self.actions)

        if qTable is None:  # initialize it randomly if we are not going to import it
            self.qTable = np.random.rand(satUp, satDown, satRight, satLeft, NGT,
                                         self.nActions)  # first 5 fields are states while 6th field is the action. 4050 values with 10 GTs

        else:
            self.qTable = qTable

        self.alpha = hyperparams.alpha
        self.gamma = hyperparams.gamma
        # self.epsilon= hyperparams.epsilon
        self.epsilon = []
        self.maxEps = hyperparams.MAX_EPSILON
        self.minEps = hyperparams.MIN_EPSILON
        self.w1 = hyperparams.w1
        self.w2 = hyperparams.w2

        self.oldState = (0, 0, 0, 0, 0)
        self.oldAction = 0

    def makeAction(self, block, sat, g, earth, prevSat=None):
        '''
        This function will:
        1. Check if the destination is the linked gateway. In that case it will just return 0 and the block will be sent there.
        2. Observation of the environment in order to determine state space and get the linked satellites.
        3. Chooses an action. Random one (Exploration) or the most valuable one (Exploitation). If the direction of that action has no linked satellite, the QValue will be -inf
        4. Receive reward/penalty
            Penalties: If the block visits again the same satellite. Reward = -1
                       Another one directly proportional to the length of the destination queue.
            Reward: So far, it will be higher if it gets physically closer to the satellite
        5. Updates Q-Table of the previous hop (Agent) with the following information:
            1. Reward      : Time waited at satB Queue && slant range reduction.
            2. maxNewQValue: Max Q Value of all possible actions at the new agent.
            3. Old state-action taken at satA in order to know where to update the Q-Table.
            Everytime satB receives a dataBlock from satA satB will send the information required to update satA QTable.
        '''

        # There is no 'Done' state, it will simply continue until the time stops
        # simplemente se va a recibir una recompensa positiva si el satelite al que envias el paquete es el linkado al destino de este

        # 1. check if the destination is the linked gateway. The value of this action becomes 10. # ANCHOR plots route of delivered package Q-Learning
        if sat.linkedGT and block.destination.name == sat.linkedGT.name:
            prevSat.QLearning.qTable[block.oldState][block.oldAction] = ArriveReward
            earth.rewards.append([ArriveReward, sat.env.now])
            if plotDeliver:
                if int(block.ID[len(block.ID) - 1]) == 0:  # Draws 1/10 arrivals
                    os.makedirs(earth.outputPath + '/pictures/', exist_ok=True)  # drawing delivered
                    outputPath = earth.outputPath + '/pictures/' + block.ID + '_' + str(len(block.QPath)) + '_'
                    plotShortestPath(earth, block.QPath, outputPath, ID=block.ID, time=block.creationTime)

            return 0

        # 2. Observation of the environment
        newState = tuple(getState(block, sat, g, earth))

        # 3. Choose an action (the direction of the next hop)
        # randomly
        if explore and random.uniform(0, 1) < self.alignEpsilon(earth, sat):
            action = self.actions[random.randrange(len(self.actions))]
            while (self.linkedSats[action] == None):
                action = self.actions[random.randrange(len(self.actions))]  # if that direction has no linked satellite

        # highest value
        else:
            qValues = self.qTable[newState]
            action = self.actions[np.argmax(qValues)]  # Most valuable action (The one that will give more reward)
            while self.linkedSats[action] == None:
                self.qTable[newState][
                    self.actions.index(action)] = -np.inf  # change qTable if that action is not available
                action = self.actions[np.argmax(qValues)]

        destination = self.linkedSats[
            action]  # Action is the keyword of the chosen linked satellite, linkedSats is a dictionary with each satellite associated to its corresponding keyword

        # ACT -> [it is done outside, the next hop is added at sat.receiveBlock method to block.QPath]
        nextHop = [destination.ID, math.degrees(destination.longitude), math.degrees(destination.latitude)]

        # 4. Receive reward/penalty for the previous action
        if prevSat is not None:
            hop = [sat.ID, math.degrees(sat.longitude), math.degrees(sat.latitude)]
            # if the next hop was already visited before the reward will be againPenalty
            if hop in block.QPath[:len(block.QPath) - 2]:
                reward = againPenalty
            else:
                distanceReward = getDistanceReward(prevSat, sat, block.destination, self.w2)
                try:
                    queueReward = getQueueReward(block.queueTime[len(block.queueTime) - 1], self.w1)
                except IndexError:
                    queueReward = 0  # FIXME
                reward = distanceReward + queueReward

            earth.rewards.append([reward, sat.env.now])

            # 5. Updates Q-Table
            # Update QTable of previous Node (Agent, satellite) if it was not a gateway
            nextMax = np.max(self.qTable[newState])  # max value of next state given oldAction
            oldQValue = prevSat.QLearning.qTable[block.oldState][block.oldAction]
            newQvalue = (1 - self.alpha) * oldQValue + self.alpha * (reward + self.gamma * nextMax)
            prevSat.QLearning.qTable[block.oldState][block.oldAction] = newQvalue

        else:
            # prev node was a gateway, no need to compute the reward
            reward = 0

        # this will be saved always, except when the next hop is the destination, where the process will have already returned
        block.oldState = newState
        block.oldAction = self.actions.index(action)

        earth.step += 1

        return nextHop

    def alignEpsilon(self, earth, sat):
        global CurrentGTnumber
        epsilon = self.minEps + (self.maxEps - self.minEps) * math.exp(
            -LAMBDA * earth.step / (decayRate * (CurrentGTnumber ** 2)))
        earth.epsilon.append([epsilon, sat.env.now])
        return epsilon

    def __repr__(self):
        return '\n Nº of destinations = {}\n Action Space = {}\n Nº of states = {}\n qTable: {}'.format(
            self.Destinations,
            self.actions,
            self.nStates,
            self.qTable)


# @profile
class DDQNAgent:
    def __init__(self, NGT, hyperparams, earth, sat_ID=None):
        self.actions = ('U', 'D', 'R', 'L')
        if not reducedState:
            self.states = ['UpLinked Up', 'UpLinked Down', 'UpLinked Right', 'UpLinked Left',  # Up Link
                           'Up Latitude', 'Up Longitude',  # Up positions
                           'DownLinked Up', 'DownLinked Down', 'DownLinked Right', 'DownLinked Left',  # Down Link
                           'Down Latitude', 'Down Longitude',  # Down positions
                           'RightLinked Up', 'RightLinked Down', 'RightLinked Right', 'RightLinked Left',  # Right Link
                           'Right Latitude', 'Right Longitude',  # Right positions
                           'LeftLinked Up', 'LeftLinked Down', 'LeftLinked Right', 'LeftLinked Left',  # Left Link
                           'Left Latitude', 'Left Longitude',  # Left positions

                           'Actual latitude', 'Actual longitude',  # Actual Position
                           'Destination latitude', 'Destination longitude']  # Destination Position
        elif reducedState:
            self.states = ('Up Latitude', 'Up Longitude',  # Up Link
                           'Down Latitude', 'Down Longitude',  # Down Link
                           'Right Latitude', 'Right Longitude',  # Right Link
                           'Left Latitude', 'Left Longitude',  # Left Link
                           'Actual latitude', 'Actual longitude',  # Current pos
                           'Destination latitude', 'Destination longitude')  # Destination pos
        if diff_lastHop:
            self.states.insert(0, 'Last Hop')

        self.actionSize = len(self.actions)
        self.stateSize = len(self.states)
        self.destinations = NGT
        self.earth = earth

        if sat_ID is None:
            print(f'State Space:\n {self.states}\nState size: {self.stateSize} states')
            print(f'Action Space:\n {self.actions}')

        self.alpha = hyperparams.alpha
        self.gamma = hyperparams.gamma
        self.epsilon = []
        self.maxEps = hyperparams.MAX_EPSILON
        self.minEps = hyperparams.MIN_EPSILON
        self.w1 = hyperparams.w1
        self.w2 = hyperparams.w2
        self.w4 = hyperparams.w4
        self.tau = hyperparams.tau
        self.updateF = hyperparams.updateF
        self.batchS = hyperparams.batchSize
        self.bufferS = hyperparams.bufferSize
        self.hardUpd = hyperparams.hardUpdate
        self.importQ = hyperparams.importQ
        self.online = hyperparams.online

        self.step = 0
        self.i = 0

        self.replayBuffer = []
        self.experienceReplay = ExperienceReplay(self.bufferS)
        # self.optimizer        = Adam(learning_rate=self.alpha, clipnorm=Clipnorm)
        self.loss_function = losses.Huber()

        if not self.importQ:
            '''
            The compile method is used to configure the learning process of qNetwork and it sets the optimizer and loss function that the model will use to learn during training.
            It only is done in the q network because in the DDQN algorithm, we train the qNetwork with the data from the environment and update qTarget periodically.

            In DDQN the qNetwork is updated with the learning process defined by the loss and optimizer, but the qTarget network used for evaluation and stability purpose is
            a frozen version of qNetwork, which is updated periodically and not during the learning process.
            '''
            # The first model makes the predictions for Q-values which are used to make a action
            self.qNetwork = self.createModel()
            if sat_ID is None:
                print('----------------------------------')
                print(f"Q-NETWORK created:")
                print('----------------------------------')
                self.qNetwork.summary()
            else:
                print(f'Satellite {sat_ID} Q-Network initialized')
            if ddqn:
                self.qTarget = self.createModel()
                if sat_ID is None:
                    print('----------------------------------')
                    print("DDQN enabled, TARGET NETWORK created:")
                    print('----------------------------------')
                    self.qTarget.summary()
                else:
                    print(f'Satellite {sat_ID} Q-Target initialized')
        else:
            # if import models, it will import a trained model
            try:
                global nnpath
                self.qNetwork = keras.models.load_model(nnpath)
                if sat_ID is None:
                    print('----------------------------------')
                    print(f"Q-Network imported!!!")
                    print('----------------------------------')
                    self.qNetwork.summary()
                else:
                    print(f'Satellite {sat_ID} Q-Network imported!')

                if ddqn:
                    global nnpathTarget
                    # self.qTarget = self.qNetwork
                    self.qTarget = keras.models.load_model(nnpathTarget)
                    if sat_ID is None:
                        print('----------------------------------')
                        # print("DDQN enabled, TARGET NETWORK copied from Q-NETWORK:")
                        print(f"Q-Target imported!!!")
                        print('----------------------------------')
                    else:
                        # print(f'Satellite {sat_ID} Q-Target copied from Q-Network')
                        print(f'Satellite {sat_ID} Q-Target imported!')

            except FileNotFoundError:
                print('----------------------------------')
                print(f"Wrong Neural Network path")
                print('----------------------------------')

    def getNextHop(self, newState, linkedSats, sat, block):
        '''
        Given a new observed state and the linkied satellites, it will return the next hop
        '''
        # randomly (Exploration)
        if explore and random.uniform(0, 1) < self.alignEpsilon(self.step, sat):
            actIndex = random.randrange(self.actionSize)
            action = self.actions[actIndex]
            while (linkedSats[action] == None):  # if that direction has no linked satellite
                self.experienceReplay.store(newState, actIndex, unavPenalty, newState,
                                            False)  # stores experience, repeats randomly
                self.earth.rewards.append([unavPenalty, sat.env.now])
                action = self.actions[random.randrange(len(self.actions))]

        # highest value (Exploitation)
        else:
            if noPingPong:  # No PING PONG: if one of the neighbours is the connected satellite then choose that one
                actIndex = -1
                if sat.upper == block.destination.linkedSat[1]:
                    actIndex = 0
                elif sat.lower == block.destination.linkedSat[1]:
                    actIndex = 1
                elif sat.right == block.destination.linkedSat[1]:
                    actIndex = 2
                elif sat.left == block.destination.linkedSat[1]:
                    actIndex = 3

                if actIndex > -1:
                    action = self.actions[actIndex]
                    destination = linkedSats[action]
                    return [destination.ID, math.degrees(destination.longitude),
                            math.degrees(destination.latitude)], actIndex

                # # Mapping from state indices to direction decisions
                # decision_map = {
                #     (4, 5): 0,    # Up
                #     (10, 11): 1,  # Down
                #     (16, 17): 2,  # Right
                #     (22, 23): 3   # Left
                # }
                #     # Current satellite's destination position
                # dest_lat = newState[0, 26]
                # dest_lon = newState[0, 27]

                # # Iterate through the decision map and compare
                # for (lat_idx, lon_idx), actIndex in decision_map.items():
                #     if np.isclose(dest_lat, newState[0, lat_idx]) and np.isclose(dest_lon, newState[0, lon_idx]):
                #         action      = self.actions[actIndex]
                #         destination = linkedSats[action]
                #         return [destination.ID, math.degrees(destination.longitude), math.degrees(destination.latitude)], actIndex

            # Predict
            qValues = self.qNetwork(newState).numpy()  # NOTE NN. Gets next hop. state structure in debugging
            actIndex = np.argmax(qValues)
            action = self.actions[actIndex]
            while (linkedSats[
                       action] == None):  # the chosen action has no linked satellite. NEGATIVE REWARD and store it, motherfucker.

                # while (linkedSats[action] is None or        # the chosen action has no linked satellite or the chosen satellite has been visited twice.
                # sum(linkedSats[action].ID == path[0] for path in block.QPath[:-1]) > 1):

                self.experienceReplay.store(newState, actIndex, unavPenalty, newState,
                                            False)  # from state to the same state, reward -1, not terminated
                self.earth.rewards.append([unavPenalty, sat.env.now])
                qValues[0][
                    actIndex] = -np.inf  # it will not be chosen again (as the model has still not trained with that)

                #     if np.all(qValues == -np.inf):              # all the neighbors have been visited twice
                #         print(f'WARNING: All neighbors have been visited at least twice. A loop is going on in {sat.ID} with block: {block.ID}')
                #         while (linkedSats[action] is None): # if all options were either not available or visited twice, then choose randomly an action that is available
                #             np.random.randint(4)
                #             actIndex = np.argmax(qValues)               # find again for the highest value
                #             action   = self.actions[actIndex]
                #         break
                actIndex = np.argmax(qValues)  # find again for the highest value
                action = self.actions[actIndex]

        destination = linkedSats[
            action]  # Action is the keyword of the chosen linked satellite, linkedSats is a dictionary with
        # each satellite associated to its corresponding keyword

        # ACT -> [it is done outside, the next hop is added at sat.receiveBlock method to block.QPath]
        try:
            return [destination.ID, math.degrees(destination.longitude), math.degrees(destination.latitude)], actIndex
        except:
            return -1

    def makeDeepAction(self, block, sat, g, earth, prevSat=None):
        '''
        There is no 'Done' state, it will simply continue until the time stops.
        This function will:
        1. Observation of the environment in order to determine state space and get the linked satellites to the one making the action.
        2. Check if the destination is the linked gateway.
            If the satellite sent the block to the satellite linked to the destination GW, it will receive a reward of 10.
            The previous satellite will match the destination of the block to the linked gateway of the next state (I hope and I guess)
            In that case it will just return 0 and the block will be sent there.
        3. Chooses an action.
            Random one (Exploration)
            The most valuable one (Exploitation).
            If the direction of that action has no linked satellite, that action will not be available.
       4. Receive reward/penalty
            Penalties: If the block visits again the same satellite. Reward = -1
                       If it tries to send the block to a direction where there is no linked satellite.
                       Another one directly proportional to the length of the destination queue.
            Reward: One proportional to the slant range reduction, meaning that it will be higher if it gets physically closer to the satellite.
                    Another one when it reaches the destination
        5. Store experience from the previous hop (Agent) with the following information:
            1. Reward      : Time waited at satB Queue && slant range reduction.
            2. maxNewQValue: Max Q Value of all possible actions at the new agent.
            3. Old state-action taken at satA in order to know where to update the NNs.
            Everytime satB receives a dataBlock from satA satB will send the information required to update the NNs.
            Unlike in regular Q-Learning, in this step we just have to store the experience into the experience replay buffer.
            It will be updated automatically taking a random batch from the buffer every n iterations.
            We will store the old state of the block, the action index taken there, the reward received and the new state it moved into.
        6. Update the qTarget every n iterations.
        '''
        # 1. Observe the state and search for the satellites linked to the one making the action
        linkedSats = getDeepLinkedSats(sat, g, earth)
        if reducedState:
            newState = getDeepStateReduced(block, sat, linkedSats)
        elif diff and not diff_lastHop:
            newState = getDeepStateDiff(block, sat, linkedSats)  # This is the one being used by default
        elif diff_lastHop:
            newState = getDeepStateDiffLastHop(block, sat, linkedSats)
        else:
            newState = getDeepState(block, sat, linkedSats)

        if newState is None:
            earth.lostBlocks += 1
            return 0
        self.step += 1

        # 2. Check if the destination is the linked gateway. The reward is ArriveReward here and goes to the previous satellite. # ANCHOR plot delivered deep NN
        if sat.linkedGT and (block.destination.ID == sat.linkedGT.ID):  # Compare IDs
            if distanceRew == 4:
                satDest = block.destination.linkedSat[1]
                distanceReward = getDistanceRewardV4(prevSat, sat, satDest, self.w2, self.w4)
                # distanceReward  = getDistanceRewardV4(prevSat, sat, block.destination, self.w2, self.w4)
                queueReward = getQueueReward(block.queueTime[len(block.queueTime) - 1], self.w1)
                reward = distanceReward + queueReward + ArriveReward
                self.experienceReplay.store(block.oldState, block.oldAction, reward, newState, True)
                self.earth.rewards.append([reward, sat.env.now])
                # self.experienceReplay.store(block.oldState, block.oldAction, ArriveReward, newState, True)
            elif distanceRew == 5:
                distanceReward = getDistanceRewardV5(prevSat, sat, self.w2)
                reward = distanceReward + ArriveReward
                self.experienceReplay.store(block.oldState, block.oldAction, reward, newState, True)
                self.earth.rewards.append([reward, sat.env.now])
            else:
                self.experienceReplay.store(block.oldState, block.oldAction, ArriveReward, newState, True)
                self.earth.rewards.append([ArriveReward, sat.env.now])

            if TrainThis: self.train(sat,
                                     earth)  # FIXME why here a train?? should not be here. Make a test without this when the model is stable
            if plotDeliver:
                if int(block.ID[len(block.ID) - 1]) == 0:  # Draws 1/10 arrivals
                    os.makedirs(earth.outputPath + '/pictures/', exist_ok=True)  # drawing delivered
                    outputPath = earth.outputPath + '/pictures/' + block.ID + '_' + str(len(block.QPath)) + '_'
                    plotShortestPath(earth, block.QPath, outputPath, ID=block.ID, time=block.creationTime)
            return 0

        # 3. Choose an action (the direction of the next hop)
        nextHop, actIndex = self.getNextHop(newState, linkedSats, sat, block)

        # 4. Computes reward/penalty for the previous action
        if prevSat is not None:
            hop = [sat.ID, math.degrees(sat.longitude), math.degrees(sat.latitude)]
            # if the next hop was already visited before the reward will be -1
            if hop in block.QPath[:len(block.QPath) - 2]:
                again = againPenalty
            else:
                again = 0

            if distanceRew == 1:
                distanceReward = getDistanceReward(prevSat, sat, block.destination, self.w2)
            elif distanceRew == 2:
                prevLinkedSats = getDeepLinkedSats(prevSat, g, earth)
                distanceReward = getDistanceRewardV2(prevSat, sat, prevLinkedSats['U'], prevLinkedSats['D'],
                                                     prevLinkedSats['R'], prevLinkedSats['L'], block.destination,
                                                     self.w2)
            elif distanceRew == 3:
                prevLinkedSats = getDeepLinkedSats(prevSat, g, earth)
                distanceReward = getDistanceRewardV3(prevSat, sat, prevLinkedSats['U'], prevLinkedSats['D'],
                                                     prevLinkedSats['R'], prevLinkedSats['L'], block.destination,
                                                     self.w2)
            elif distanceRew == 4:
                satDest = block.destination.linkedSat[1]
                distanceReward = getDistanceRewardV4(prevSat, sat, satDest, self.w2, self.w4)
                # distanceReward  = getDistanceRewardV4(prevSat, sat, block.destination, self.w2, self.w4)
            elif distanceRew == 5:
                distanceReward = getDistanceRewardV5(prevSat, sat, self.w2)

            try:
                queueReward = getQueueReward(block.queueTime[len(block.queueTime) - 1], self.w1)
            except IndexError:
                queueReward = 0  # FIXME In some hop the queue time was not appended to block.queueTime, line 620
            reward = distanceReward + again + queueReward

            # 5. Store the experience of previous Node (Agent, satellite) if it was not a gateway
            self.experienceReplay.store(block.oldState, block.oldAction, reward, newState, False)  # action index
            self.earth.rewards.append([reward, sat.env.now])

            # 6. Learning, train the Q-Network at every time we store experience
            if TrainThis and self.step % nTrain == 0:
                self.train(sat, earth)

        else:
            # prev node was a gateway, no need to compute the reward
            reward = 0

        # 7. Align the Q-Target
        if ddqn:
            self.alignQTarget(hardUpdate)

        # this will be saved always, except when the next hop is the destination, where the process will have already returned
        block.oldState = newState
        block.oldAction = actIndex

        return nextHop

    def alignEpsilon(self, step, sat):  # the epsilon is reduced with time
        '''
        Updates epsilon value at each step
        0.01+0.99*e^(-0.0005*10000):
        0     -> 1
        1000  -> 0.61
        5000  -> 0.091
        10000 -> 0.01667
        '''
        global CurrentGTnumber
        epsilon = self.minEps + (self.maxEps - self.minEps) * math.exp(
            -LAMBDA * step / (decayRate * (CurrentGTnumber ** 2)))
        self.epsilon.append([epsilon, sat.env.now])
        return epsilon

    def alignQTarget(self, hardUpdate=True):  # Soft one is done every step
        '''
        This function is not used now since the q target only exists in double deep q learning and it is not implemented.
        Updates the qTarget NN with the weights of the qNetwork.

        The choice between using hard updates or soft updates for the target network depends on the specific requirements of your problem and the properties of your data.

        Hard updates, where the target network is updated with the latest weights of the Q-network, could be more beneficial when the data changes frequently and quickly.
        However, if the data is relatively stable and consistent, then hard updates may cause the target network to oscillate too much, destabilizing the training of the Q-network.

        Soft updates, where the target network's parameters are updated with a moving average of the Q-network's parameters, are more stable than hard updates and can help the
        Q-network converge more smoothly. This is because soft updates gradually propagate the changes in the Q-network's parameters to the target network, rather than suddenly
        switching to the latest weights. This can be a better choice when the data is relatively stable and consistent, or when you're worried about potential stability issues in
        the training process.

        Ultimately, the best way to determine which method is more convenient is through experimentation with your specific problem and dataset.
        '''
        if hardUpdate:
            self.i += 1
            if self.i == self.updateF:
                self.qTarget.set_weights(self.qNetwork.get_weights())  # NOTE qTarget gets qNetrowk values
                # print(f"Q-Target network hard updated!!!")
                self.i = 0

        else:
            for t, e in zip(self.qTarget.trainable_variables,
                            self.qNetwork.trainable_variables): t.assign(t * (1 - self.tau) + e * self.tau)

    def createModel(self):
        model = Sequential()
        model.add(Dense(32, activation='relu', input_shape=(self.stateSize,), kernel_initializer='random_uniform'))
        model.add(Dense(32, activation='relu', kernel_initializer='random_uniform'))
        model.add(Dense(self.actionSize, activation='linear'))
        # optimizer = Adam(learning_rate=alpha_dnn)
        # model.compile(loss='mse', optimizer=optimizer)
        model.compile(loss='mse', optimizer='adam')
        return model

    def train(self, sat, earth):
        if self.experienceReplay.buffeSize < self.batchS * 3:
            return -1

        # 1. Get a random batch from the experience
        miniBatch = self.experienceReplay.getBatch(self.batchS)
        states, actions, rewards, nextStates, Dones = self.experienceReplay.getArraysFromBatch(miniBatch)
        states = states.reshape((self.batchS, self.stateSize))
        nextStates = nextStates.reshape((self.batchS, self.stateSize))

        # 2. Compute expected reward
        if ddqn:
            futureRewards = self.qTarget(nextStates)  # NOTE NN. Gets future rewards
        else:
            futureRewards = self.qNetwork(nextStates)  # NOTE NN. Gets future rewards
        expectedRewards = rewards + self.gamma * np.max(futureRewards, axis=1)

        # 3. Mask for the actions
        acts = np.eye(self.actionSize)[actions]

        # 4. Stop Loss
        if stopLoss and len(sat.orbPlane.earth.loss) > nLosses:
            savedLoss = sat.orbPlane.earth.loss
            last_n_losses = [sample[0] for sample in savedLoss[-nLosses:]]
            average = sum(last_n_losses) / nLosses
            sat.orbPlane.earth.lossAv.append(average)
            if average < lThreshold:
                global TrainThis
                TrainThis = False
                print('----------------------------------')
                print(f"STOP LOSS ACTIVATED")
                print(f'Last {nLosses} losses: {last_n_losses}')
                print(f'Simulation time: {sat.env.now}')
                print('----------------------------------')
                return 0

        # 5. fit the model and save the loss
        loss = self.qNetwork.fit(states, acts * expectedRewards[:, None], batch_size=self.batchS, epochs=1,
                                 verbose=0)  # NOTE qNetwork fit
        sat.orbPlane.earth.loss.append([loss.history['loss'][0], sat.env.now])
        earth.trains.append([sat.env.now])  # counts the number of trainings


# @profile
class ExperienceReplay:
    def __init__(self, maxlen=100):
        '''
        This is a buffer that holds information that are used during training process.

        Deque (Doubly Ended Queue). Deque is preferred over a list in the cases where we need quicker append and pop operations
        from both the ends of the container, as deque provides an O(1) time complexity for append and pop operations as compared
        to a list that provides O(n) time complexity
        '''
        self.buffer = deque(maxlen=maxlen)

    def store(self, state, action, reward, nextState, terminated):
        '''
        appends a set of (state, action, reward, next state, terminated) to the experience replay buffer
        '''
        # if the buffer is full, it behave as a FIFO
        self.buffer.append((state, action, reward, nextState, terminated))

    def getBatch(self, batchSize):
        '''
        gets a random batch of samples from all the samples
        '''
        return random.sample(self.buffer, batchSize)

    def getArraysFromBatch(self, batch):
        '''
        gets the batch data divided into fields
        '''
        states = np.array([x[0] for x in batch])
        actions = np.array([x[1] for x in batch])
        rewards = np.array([x[2] for x in batch])
        next_st = np.array([x[3] for x in batch])
        dones = np.array([x[4] for x in batch])

        return states, actions, rewards, next_st, dones

    @property
    def buffeSize(self):
        '''
        a pythonic way to use getters and setters in object-oriented programming
        this decorator is a built-in function that allows us to define methods that can be accessed like an attribute
        '''
        return len(self.buffer)


###############################################################################
############################   Functions    ###################################
###############################################################################


# @profile
def initialize(env, popMapLocation, GTLocation, distance, inputParams, movementTime, totalLocations, outputPath,
               matching='Greedy', TerrestrialNodesLocation=None):
    print = builtins.print

    # Build terrestrial backbone
    print("Building terrestrial graph...")
    G = build_terrestrial_graph(json_path="world_named.json", gateways_csv="Gateways.csv", k_nearest=3)
    print("Graph built.")

    # draw_terrestrial_graph()

    # Extract simulation parameters
    constellationType = inputParams['Constellation'][0]
    # Use a default fraction of 1.0 if not specified or if it's NaN/empty
    try:
        fraction = float(inputParams['Fraction'][0])
        if pd.isna(fraction) or fraction <= 0:
            fraction = 1.0
    except (ValueError, KeyError, IndexError):
        fraction = 1.0
    testType = inputParams['Test type'][0]
    print(f'Fraction of traffic generated: {fraction}, test type: {testType}')
    getRates = (testType == "Rates")

    # Create Earth object (satellites + gateways)
    earth = Earth(
        env=env,
        img_path=popMapLocation,
        gt_path=GTLocation,
        constellation=constellationType,
        inputParams=inputParams,
        deltaT=movementTime,
        totalLocations=totalLocations,
        getRates=getRates,
        outputPath=outputPath,
        terrestrial_nodes_path=TerrestrialNodesLocation,
        enable_gateway_traffic=False
    )
    print(earth);
    print()

    # Build constellation and space graph
    # earth.linkCells2GTs(distance)  # Disabled: not using gateways as traffic generators
    earth.linkSats2GTs("Optimize")
    spaceG = createGraph(earth, matching=matching)

    earth.terr_graph = G
    earth.space_graph = spaceG

    # Map gateway keys between terrestrial and space graphs
    gw_ok = 0
    gw_alias = 0
    gw_fail = 0
    for gt in earth.gateways:
        terr_key, why = resolve_gateway_terr_key(earth.terr_graph, gt.name)
        if terr_key:
            gt.terr_key = terr_key
            gt.space_key = gw_space_key(terr_key, earth.space_graph, gt.name)
            gw_ok += 1 if gt.space_key == terr_key else 0
            gw_alias += 1 if gt.space_key != terr_key else 0
        else:
            gt.terr_key = None
            gt.space_key = gt.name
            gw_fail += 1

    attached = attach_gateways_to_space_graph(earth)

    earth.node_by_key = {}
    earth.node_by_name = {}
    earth.sat_by_id = {}

    # Create TerrestrialNode for all TopoHub nodes
    terrestrial_nodes = []
    totalX = getattr(earth, 'totalX', getattr(earth, 'total_x', 1440))
    totalY = getattr(earth, 'totalY', getattr(earth, 'total_y', 720))

    terrestrial_nodes.extend(getattr(earth, 'terrestrial_nodes', []))
    already = {getattr(n, 'name', None) for n in terrestrial_nodes}

    for node_key, nd in G.nodes(data=True):
        if node_key in already:
            n = next(n for n in terrestrial_nodes if n.name == node_key)
            earth.node_by_key[node_key] = n
            earth.node_by_name[str(node_key)] = n
            continue

        pos = nd.get('pos') or (nd.get('lon'), nd.get('lat'))
        if not pos or pos[0] is None or pos[1] is None:
            continue
        lon, lat = float(pos[0]), float(pos[1])

        tn = TerrestrialNode(
            name=node_key,
            ID=node_key,
            latitude=lat,
            longitude=lon,
            totalX=totalX,
            totalY=totalY,
            totalNodes=len(G),
            env=env,
            totalLocations=[],
            earth=earth,
            graph=G
        )
        terrestrial_nodes.append(tn)
        earth.node_by_key[node_key] = tn
        earth.node_by_name[str(node_key)] = tn
        # Normalize Unicode keys for better matching
        try:
            _ud = unicodedata
            for k in {_ud.normalize('NFC', str(node_key)), _ud.normalize('NFKC', str(node_key))}:
                earth.node_by_name[k] = tn
        except Exception:
            pass

    earth.terrestrial_nodes = terrestrial_nodes

    # Link nodes to graphs
    for n in earth.terrestrial_nodes:
        n.graph = earth.terr_graph
        n.earth = earth

    # Add gateways to node indices
    for gt in earth.gateways:
        gt.graph = earth.space_graph
        gt.space_graph = earth.space_graph
        gt.terr_graph = earth.terr_graph
        gt.earth = earth
        # Normalize Unicode keys for better matching
        _ud = unicodedata
        for k in {str(gt.name), _ud.normalize('NFC', str(gt.name)), _ud.normalize('NFKC', str(gt.name))}:
            earth.node_by_name[k] = gt
        earth.node_by_key[gt.name] = gt

    # Add satellites by ID
    for plane in earth.LEO:
        for sat in plane.sats:
            earth.sat_by_id[sat.ID] = sat

    def _getNodeByName(key):
        if isinstance(key, int) and key in earth.sat_by_id:
            return earth.sat_by_id[key]
        return earth.node_by_name.get(str(key))

    earth.getNodeByName = _getNodeByName

    # Load active terrestrial nodes from inputRL.csv
    topo_types = nx.get_node_attributes(G, 'type')
    name2node = {str(n.name): n for n in earth.terrestrial_nodes}

    try:
        rl_df = pd.read_csv('inputRL.csv')
        col = 'Locations' if 'Locations' in rl_df.columns else rl_df.columns[0]
        csv_nodes = [str(x).strip() for x in rl_df[col].dropna().tolist()]
    except Exception:
        csv_nodes = []

    def pick_city(nm):
        n = name2node.get(nm)
        if n is None or topo_types.get(n.name) != 'City':
            pass
        return n if (n and topo_types.get(n.name) == 'City') else None

    # Initialize active terrestrial nodes
    active_terrestrial = []
    for nm in csv_nodes:
        n = pick_city(nm)
        if n is None:
            continue
        if n not in active_terrestrial:
            active_terrestrial.append(n)

    if not active_terrestrial:
        # If no active nodes from CSV, select random cities
        city_nodes = [n for n in earth.terrestrial_nodes if topo_types.get(n.name) == 'City']
        if city_nodes:
            random.seed(42)
            active_terrestrial = random.sample(city_nodes, k=min(10, len(city_nodes)))
        else:
            print("[WARN] No city nodes found in terrestrial graph")

    earth.active_terrestrial_nodes = active_terrestrial

    earth.src_node = active_terrestrial[0] if len(active_terrestrial) >= 1 else None
    earth.dst_node = active_terrestrial[1] if len(active_terrestrial) >= 2 else None

    print("Selected active terrestrial nodes:", [str(n.name) for n in active_terrestrial])
    if earth.src_node and earth.dst_node:
        print(f"Source/Destination pair: {earth.src_node.name} → {earth.dst_node.name}")
    else:
        print("[WARN] Meno di due City attive dal CSV: coppia S/D non impostata per il monitoraggio.")

    active_set = set(active_terrestrial)

    for n in earth.terrestrial_nodes:
        n.totalLocations = [m for m in active_terrestrial if m is not n] if (n in active_set) else []

    # Traffic configuration
    MAX_COVERAGE_KM = 25
    DIST_FUNC = "Step"
    FRACTION = float(fraction)
    AVG_FLOW_PER_USER = getattr(globals(), 'avUserLoad', 20e6)

    print("Traffic generated per Active Terrestrial Node (totalAvgFlow per Milliard):")
    print('----------------------------------')
    for n in earth.active_terrestrial_nodes:
        n.setup_coverage_and_flow(
            earth=earth,
            maxDistance_km=MAX_COVERAGE_KM,
            distanceFunc=DIST_FUNC,
            capacity=None,
            fraction=FRACTION,
            avgFlowPerUser=AVG_FLOW_PER_USER,
            clear=True
        )
    print('----------------------------------')

    # Precompute paths between active nodes (terrestrial vs hybrid choice)
    from collections import defaultdict

    path_stats = {
        "terr": 0,
        "hyb": 0,
        "total": 0,
        "hyb_available": 0,
        "samples": [],
        "sample_wins": []
    }

    for src in earth.active_terrestrial_nodes:
        src.paths = getattr(src, "paths", {})
        src.path_types = {}

        for dst in earth.active_terrestrial_nodes:
            if src is dst:
                continue

            p_terr = getShortestPathTerrestrial(src.name, dst.name, G)
            p_hyb = compute_hybrid_path(src.name, dst.name, earth)

            terr_cost = estimate_path_cost(earth, p_terr) if p_terr else float("inf")
            hyb_cost = estimate_path_cost(earth, p_hyb) if p_hyb else float("inf")

            if p_hyb:
                path_stats["hyb_available"] += 1

            # Choose best path (prefer terrestrial if costs are equal)
            if hyb_cost is not None and hyb_cost < terr_cost:
                chosen_raw, chosen_type, other_cost = p_hyb, "hyb", terr_cost
                path_stats["hyb"] += 1
            else:
                chosen_raw, chosen_type, other_cost = p_terr, "terr", hyb_cost
                if p_terr:
                    path_stats["terr"] += 1

            chosen_path = coerce_path_only(chosen_raw)
            src.paths[dst.name] = chosen_path
            src.path_types[dst.name] = chosen_type if chosen_path else "none"

            chosen_cost = estimate_path_cost(earth, chosen_path) if chosen_path else float("inf")
            srec = {
                "src": src.name,
                "dst": dst.name,
                "choice": chosen_type,
                "cost_chosen": chosen_cost,
                "cost_other": other_cost
            }
            path_stats["samples"].append(srec)
            if chosen_path and len(path_stats["sample_wins"]) < 10:
                path_stats["sample_wins"].append(srec)

            path_stats["total"] += 1

    print(
        f"[PATH CHOICES] terrestrial: {path_stats['terr']} ({100.0 * path_stats['terr'] / max(1, path_stats['total']):.1f}%)"
        f"  |  hybrid: {path_stats['hyb']} ({100.0 * path_stats['hyb'] / max(1, path_stats['total']):.1f}%)"
        f"  |  hybrid available: {path_stats['hyb_available']} ({100.0 * path_stats['hyb_available'] / max(1, path_stats['total']):.1f}%)"
        f"  |  total pairs: {path_stats['total']}"
    )

    if path_stats["sample_wins"]:
        print("[PATH CHOICES] esempi:")
        for s in path_stats["sample_wins"]:
            other_str = "inf"
            if (s.get("cost_other") is not None) and (s["cost_other"] != float("inf")):
                other_str = f"{s['cost_other']:.4f}"
            print(f"  {s['src']} -> {s['dst']}: chose {s['choice']} | "
                  f"chosen={s['cost_chosen']:.4f}  other={other_str}")

    # Debug: show cost breakdown for first path
    for s in earth.active_terrestrial_nodes[:1]:
        for d in earth.active_terrestrial_nodes[:1]:
            if s is d: continue
            p = s.paths.get(d.name)
            if p:
                print(f"\n[cost-verbose] {s.name} -> {d.name}")
                estimate_path_cost(earth, p, verbose=True, label=f"{s.name} -> {d.name}")

    # 9) Start block generation from active nodes
    for n in earth.active_terrestrial_nodes:
        if n.totalLocations:
            n.makeFillBlockProcesses(target_nodes=n.totalLocations)

    # Setup gateway-to-gateway paths
    paths = []
    for GT in earth.gateways:
        for destination in earth.gateways:
            if GT != destination and destination.linkedSat[0] is not None and GT.linkedSat[0] is not None:
                path = getShortestPath(GT.name, destination.name, earth.pathParam, GT.graph)
                GT.paths[destination.name] = path
                paths.append(path)

    # Setup satellite ISL references and sending processes
    sats = []
    for plane in earth.LEO:
        for sat in plane.sats:
            sats.append(sat)
            sat.findInterNeighbours(earth)

    for plane in earth.LEO:
        for sat in plane.sats:
            if sat.linkedGT is not None:
                sat.adjustDownRate()
                sat.sendBlocksGT.append(sat.env.process(sat.sendBlock((sat.GTDist, sat.linkedGT), False)))
            neighbors = list(nx.neighbors(spaceG, sat.ID))
            itt = 0
            for sat2 in sats:
                if sat2.ID in neighbors:
                    dataRate = nx.path_weight(spaceG, [sat2.ID, sat.ID], "dataRateOG")
                    distance = nx.path_weight(spaceG, [sat2.ID, sat.ID], "slant_range")
                    if sat2.in_plane == sat.in_plane:
                        sat.intraSats.append((distance, sat2, dataRate))
                        sat.sendBufferSatsIntra.append(([sat.env.event()], [], sat2.ID))
                        sat.sendBlocksSatsIntra.append(
                            sat.env.process(sat.sendBlock((distance, sat2, dataRate), True, True)))
                    else:
                        sat.interSats.append((distance, sat2, dataRate))
                        sat.sendBufferSatsInter.append(([sat.env.event()], [], sat2.ID))
                        sat.sendBlocksSatsInter.append(
                            sat.env.process(sat.sendBlock((distance, sat2, dataRate), True, False)))
                    itt += 1
                    if itt == len(neighbors):
                        break

    # Calculate space network bottlenecks
    if len(paths) >= 2:
        bottleneck2, minimum2 = findBottleneck(paths[1], earth, False)
        bottleneck1, minimum1 = findBottleneck(paths[0], earth, False, minimum2)
    else:
        bottleneck1 = bottleneck2 = minimum1 = minimum2 = None

    # Setup Q-learning if enabled
    if pathing == 'Q-Learning' or pathing == 'Deep Q-Learning':
        hyperparams = hyperparam(pathing)
    if pathing == 'Deep Q-Learning':
        if not onlinePhase:
            earth.DDQNA = DDQNAgent(len(earth.gateways), hyperparams, earth)
        else:
            print('----------------------------------')
            print('Creating satellites agents...')
            if importQVals:
                print: (f'Importing the Neural networks from: \n{nnpath}\n{nnpathTarget}')
            for plane in earth.LEO:
                for sat in plane.sats:
                    sat.DDQNA = DDQNAgent(len(earth.gateways), hyperparams, earth, sat.ID)
            print('----------------------------------')

    if pathing == 'Q-Learning' or pathing == "Deep Q-Learning":
        saveHyperparams(earth.outputPath, inputParams, hyperparams)
    if pathing == 'Q-Learning':
        earth.initializeQTables(len(earth.gateways), hyperparams, spaceG)

    return earth, spaceG, bottleneck1, bottleneck2


def _terr_edge_cost(u, v, edata):
    prop = edata.get('propDelay') or edata.get('prop_delay')
    if prop is None:
        dist_km = edata.get('distance_km')
        prop = (dist_km * 1000.0) / C_TER if dist_km is not None else 0.0
    dr = edata.get('dataRate') or edata.get('data_rate') or edata.get('capacity')
    ser = (BLOCK_SIZE / float(dr)) if (dr and dr > 0) else 0.0
    return float(prop) + float(ser)


# @profile
def findBottleneck(path, earth, plot=False, minimum=None):
    # Find the bottleneck of a route.
    bottleneck = [[], [], [], []]
    for GT in earth.gateways:
        if GT.name == path[0][0]:
            bottleneck[0].append(str(path[0][0].split(",")[0]) + "," + str(path[1][0]))
            bottleneck[1].append(GT.dataRate)
            bottleneck[2].append(GT.latitude)
            if minimum:
                bottleneck[3].append(minimum / GT.dataRate)

    for i, step in enumerate(path[1:], 1):
        for orbit in earth.LEO:
            for satellite in orbit.sats:
                if satellite.ID == step[0]:

                    for sat in satellite.interSats:
                        if sat[1].ID == path[i + 1][0]:
                            bottleneck[0].append(str(path[i][0]) + "," + str(path[i + 1][0]))
                            bottleneck[1].append(sat[2])
                            bottleneck[2].append(satellite.latitude)
                            if minimum:
                                bottleneck[3].append(minimum / sat[2])
                    for sat in satellite.intraSats:
                        if sat[1].ID == path[i + 1][0]:
                            bottleneck[0].append(str(path[i][0]) + "," + str(path[i + 1][0]))
                            bottleneck[1].append(sat[2])
                            bottleneck[2].append(satellite.latitude)
                            if minimum:
                                bottleneck[3].append(minimum / sat[2])
    for GT in earth.gateways:
        if GT.name == path[-1][0]:
            bottleneck[0].append(str(path[-2][0]) + "," + str(path[-1][0].split(",")[0]))
            bottleneck[1].append(GT.linkedSat[1].downRate)
            bottleneck[2].append(GT.latitude)
            if minimum:
                bottleneck[3].append(minimum / GT.dataRate)

    if plot:
        earth.plotMap(True, True, path, bottleneck)
        plt.show()
        plt.close()

    minimum = np.amin(bottleneck[1])
    return bottleneck, minimum


# @profile
def create_Constellation(specific_constellation, env, earth):
    if specific_constellation == "small":  # Small Walker star constellation for tests.
        print("Using small walker Star constellation")
        P = 4  # Number of orbital planes
        N_p = 8  # Number of satellites per orbital plane
        N = N_p * P  # Total number of satellites
        height = 1000e3  # Altitude of deployment for each orbital plane (set to the same altitude here)
        inclination_angle = 53  # Inclination angle for the orbital planes, set to 90 for Polar
        Walker_star = True  # Set to True for Walker star and False for Walker Delta
        min_elevation_angle = 30

    elif specific_constellation == "Kepler":
        print("Using Kepler constellation design")
        P = 7
        N_p = 20
        N = N_p * P
        height = 600e3
        inclination_angle = 98.6
        Walker_star = True
        min_elevation_angle = 30

    elif specific_constellation == "Iridium_NEXT":
        print("Using Iridium NEXT constellation design")
        P = 6
        N_p = 11
        N = N_p * P
        height = 780e3
        inclination_angle = 86.4
        Walker_star = True
        min_elevation_angle = 30

    elif specific_constellation == "OneWeb":
        print("Using OneWeb constellation design")
        P = 18
        N = 648
        N_p = int(N / P)
        height = 1200e3
        inclination_angle = 86.4
        Walker_star = True
        min_elevation_angle = 30

    elif specific_constellation == "Starlink":  # Phase 1 550 km altitude orbit shell
        print("Using Starlink constellation design")
        P = 72
        N = 1584
        N_p = int(N / P)
        height = 550e3
        inclination_angle = 53
        Walker_star = False
        min_elevation_angle = 25

    elif specific_constellation == "Test":
        print("Using a test constellation design")
        P = 30  # Number of orbital planes
        N = 1200  # Total number of satellites
        N_p = int(N / P)  # Number of satellites per orbital plane
        height = 600e3  # Altitude of deployment for each orbital plane (set to the same altitude here)
        inclination_angle = 86.4  # Inclination angle for the orbital planes, set to 90 for Polar
        Walker_star = True  # Set to True for Walker star and False for Walker Delta
        min_elevation_angle = 30
    else:
        print("Not valid Constellation Name")
        P = np.NaN
        N_p = np.NaN
        N = np.NaN
        height = np.NaN
        inclination_angle = np.NaN
        Walker_star = False
        exit()

    distribution_angle = 2 * math.pi  # Angle in which the orbital planes are distributed in

    if Walker_star:
        distribution_angle /= 2
    orbital_planes = []

    # Add orbital planes and satellites
    # Orbital_planes.append(orbital_plane(0, height, 0, math.radians(inclination_angle), N_p, min_elevation_angle, 0))
    for i in range(0, P):
        orbital_planes.append(
            OrbitalPlane(str(i), height, i * distribution_angle / P, math.radians(inclination_angle), N_p,
                         min_elevation_angle, str(i) + '_', env, earth))

    return orbital_planes


###############################################################################
###############################  Create Graph   ###############################
###############################################################################


def get_direction(Satellites):
    '''
    Gets the direction of the satellites so each transceiver antenna can be set to one direction.
    '''
    N = len(Satellites)
    direction = np.zeros((N, N), dtype=np.int8)
    for i in range(N):
        epsilon = -Satellites[i].inclination  # orbital plane inclination
        for j in range(N):
            direction[i, j] = np.sign(Satellites[i].y * math.sin(epsilon) +
                                      Satellites[i].z * math.cos(epsilon) - Satellites[j].y * math.sin(epsilon) -
                                      Satellites[j].z * math.cos(epsilon))
    return direction


def get_pos_vectors_omni(Satellites):
    '''
    Given a list of satellites returns a list with x, y, z coordinates and the plane where they are (meta)
    '''
    N = len(Satellites)
    Positions = np.zeros((N, 3))
    meta = np.zeros(N, dtype=np.int_)
    for n in range(N):
        Positions[n, :] = [Satellites[n].x, Satellites[n].y, Satellites[n].z]
        meta[n] = Satellites[n].in_plane

    return Positions, meta


def get_slant_range(edge):
    return (edge.slant_range)


# @numba.jit  # Using this decorator you can mark a function for optimization by Numba's JIT compiler
def get_slant_range_optimized(Positions, N):
    '''
    returns a matrix with the all the distances between the satellites (optimized)
    '''
    slant_range = np.zeros((N, N))
    for i in range(N):
        slant_range[i, i] = math.inf
        for j in range(i + 1, N):
            slant_range[i, j] = np.linalg.norm(Positions[i, :] - Positions[j, :])
    slant_range += np.transpose(slant_range)
    return slant_range


@numba.jit  # Using this decorator you can mark a function for optimization by Numba's JIT compiler
def los_slant_range(_slant_range, _meta, _max, _Positions):
    '''
    line of sight slant range
    '''
    _slant_range_new = np.copy(_slant_range)
    _N = len(_slant_range)
    for i in range(_N):
        for j in range(_N):
            if _slant_range_new[i, j] > _max[_meta[i], _meta[j]]:
                _slant_range_new[i, j] = math.inf
    return _slant_range_new


def get_data_rate(_slant_range_los, interISL):
    """
    Given a matrix of slant ranges returns a matrix with all the shannon dataRates possibles between all the satellites.
    """
    speff_thresholds = np.array(
        [0, 0.434841, 0.490243, 0.567805, 0.656448, 0.789412, 0.889135, 0.988858, 1.088581, 1.188304, 1.322253,
         1.487473, 1.587196, 1.647211, 1.713601, 1.779991, 1.972253, 2.10485, 2.193247, 2.370043, 2.458441,
         2.524739, 2.635236, 2.637201, 2.745734, 2.856231, 2.966728, 3.077225, 3.165623, 3.289502, 3.300184,
         3.510192, 3.620536, 3.703295, 3.841226, 3.951571, 4.206428, 4.338659, 4.603122, 4.735354, 4.933701,
         5.06569, 5.241514, 5.417338, 5.593162, 5.768987, 5.900855])
    lin_thresholds = np.array(
        [1e-10, 0.5188000389, 0.5821032178, 0.6266138647, 0.751622894, 0.9332543008, 1.051961874, 1.258925412,
         1.396368361, 1.671090614, 2.041737945, 2.529297996, 2.937649652, 2.971666032, 3.25836701, 3.548133892,
         3.953666201, 4.518559444, 4.83058802, 5.508076964, 6.45654229, 6.886522963, 6.966265141, 7.888601176,
         8.452788452, 9.354056741, 10.49542429, 11.61448614, 12.67651866, 12.88249552, 14.48771854, 14.96235656,
         16.48162392, 18.74994508, 20.18366364, 23.1206479, 25.00345362, 30.26913428, 35.2370871, 38.63669771,
         45.18559444, 49.88844875, 52.96634439, 64.5654229, 72.27698036, 76.55966069, 90.57326009])

    pathLoss = 10 * np.log10((4 * math.pi * _slant_range_los * interISL.f / Vc) ** 2)  # Free-space pathloss in dB
    snr = 10 ** ((interISL.maxPtx_db + interISL.G - pathLoss - interISL.No) / 10)  # SNR in times
    shannonRate = interISL.B * np.log2(1 + snr)  # data rates matrix in bits per second

    speffs = np.zeros((len(_slant_range_los), len(_slant_range_los)))

    for n in range(len(_slant_range_los)):
        for m in range(len(_slant_range_los)):
            feasible_speffs = speff_thresholds[np.nonzero(lin_thresholds <= snr[n, m])]
            if feasible_speffs.size == 0:
                speffs[n, m] = 0
            else:
                speffs[n, m] = interISL.B * feasible_speffs[-1]

    return speffs


def markovianMatchingTwo(earth):
    '''
    Returns a list of edge class elements. Each edge stands for a connection between two satellites. On that class
    the slant range and the data rate between both satellites are stored as attributes.
    This function is for satellites with two transceivers antennas that will enable two inter-plane ISL each one
    in a different direction.
    Intra-plane ISL are also computed and returned in _A_Markovian list

    It is not the optimal solution, but it is from 10 to 1000x faster.
    Minimizes the total cost of the constellation matching problem.
    '''

    _A_Markovian = []  # list with all the
    Satellites = []  # list with all the satellites
    W_M = []  # list with the distances of every possible link between sats
    covered = set()  # Set with the connections already covered

    for plane in earth.LEO:
        for sat in plane.sats:
            Satellites.append(sat)

    N = len(Satellites)

    interISL = RFlink(
        frequency=26e9,
        bandwidth=500e6,
        maxPtx=10,
        aDiameterTx=0.26,
        aDiameterRx=0.26,
        pointingLoss=0.3,
        noiseFigure=2,
        noiseTemperature=290,
        min_rate=10e3
    )

    # max slant range for each orbit
    ###########################################################
    M = len(earth.LEO)  # Number of planes in LEO
    Max_slnt_rng = np.zeros((M, M))  # All ISL slant ranges must me lowe than 'Max_slnt_rng[i, j]'

    Orb_heights = []
    for plane in earth.LEO:
        Orb_heights.append(plane.h)
        maxSlantRange = plane.sats[0].maxSlantRange

    for _i in range(M):
        for _j in range(M):
            Max_slnt_rng[_i, _j] = (np.sqrt((Orb_heights[_i] + Re) ** 2 - Re ** 2) +
                                    np.sqrt((Orb_heights[_j] + Re) ** 2 - Re ** 2))

    # Get data rate old method
    ###########################################################
    direction = get_direction(Satellites)  # get both directions of the satellites to use the two transceivers
    Positions, meta = get_pos_vectors_omni(Satellites)  # position and plane of all the satellites
    slant_range = get_slant_range_optimized(Positions, N)  # matrix with all the distances between satellties
    slant_range_los = los_slant_range(slant_range, meta, Max_slnt_rng,
                                      Positions)  # distance matrix but if d>dMax, d=infinite
    shannonRate = get_data_rate(slant_range_los, interISL)  # max dataRate

    '''
    Compute all possible edges between different plane satellites whose transceiver antennas are free.
    if slant range > max slant range then that edge is not added
    '''
    ###########################################################
    for i in range(N):
        for j in range(i + 1, N):
            if Satellites[i].in_plane != Satellites[j].in_plane and ((i, direction[i, j]) not in covered) and (
                    (j, direction[j, i]) not in covered):
                if slant_range_los[i, j] < 6000e3:  # math.inf:
                    W_M.append(edge(Satellites[i].ID, Satellites[j].ID, slant_range_los[i, j], direction[i, j],
                                    direction[j, i], shannonRate[i, j]))

    W_sorted = sorted(W_M, key=get_slant_range)  # NOTE we could choose shannonRate instead

    # from all the possible links adds only the uncovered with the best weight possible
    ###########################################################
    while W_sorted:
        if ((W_sorted[0].i, W_sorted[0].dji) not in covered) and ((W_sorted[0].j, W_sorted[0].dij) not in covered):
            _A_Markovian.append(W_sorted[0])
            covered.add((W_sorted[0].i, W_sorted[0].dji))
            covered.add((W_sorted[0].j, W_sorted[0].dij))
        W_sorted.pop(0)

    # add intra-ISL edges
    ###########################################################
    for plane in earth.LEO:
        nPerPlane = len(plane.sats)
        for sat in plane.sats:
            sat.findIntraNeighbours(earth)

            # upper neighbour
            i = sat.in_plane * nPerPlane + sat.i_in_plane
            j = sat.upper.in_plane * nPerPlane + sat.upper.i_in_plane

            _A_Markovian.append(edge(sat.ID, sat.upper.ID,  # satellites IDs
                                     slant_range_los[i, j],  # distance between satellites
                                     direction[i, j], direction[j, i],  # directions
                                     shannonRate[i, j]))  # Max dataRate

            # lower neighbour
            j = sat.lower.in_plane * nPerPlane + sat.lower.i_in_plane

            _A_Markovian.append(edge(sat.ID, sat.lower.ID,  # satellites IDs
                                     slant_range_los[i, j],  # distance between satellites
                                     direction[i, j], direction[j, i],  # directions
                                     shannonRate[i, j]))  # Max dataRate

    return _A_Markovian


def greedyMatching(earth):
    '''
    Returns a list of edge class elements based on a greedy algorithm.
    Each satellite is connected to its upper and lower satellite in the same orbital plane (intra-plane),
    and the nearest satellites to the east and west in different planes (inter-plane).
    The slant range and the data rate between satellites are stored as attributes in the edge class.
    '''

    _A_Greedy = []  # list to store edges
    Satellites = []  # list of all satellites

    # Collect all satellites from each plane
    for plane in earth.LEO:
        for sat in plane.sats:
            Satellites.append(sat)

    N = len(Satellites)

    # inter-plane ISL
    ##############################################################
    # link parameters
    interISL = RFlink(
        frequency=f,
        bandwidth=B,
        maxPtx=maxPtx,
        aDiameterTx=Adtx,
        aDiameterRx=Adrx,
        pointingLoss=pL,
        noiseFigure=Nf,
        noiseTemperature=Tn,
        min_rate=min_rate
    )

    # max slant range for each orbit
    ###########################################################
    M = len(earth.LEO)  # Number of planes in LEO
    Max_slnt_rng = np.zeros((M, M))  # All ISL slant ranges must be lowe than 'Max_slnt_rng[i, j]'

    Orb_heights = []
    for plane in earth.LEO:
        Orb_heights.append(plane.h)
        maxSlantRange = plane.sats[0].maxSlantRange

    for _i in range(M):
        for _j in range(M):
            Max_slnt_rng[_i, _j] = (np.sqrt((Orb_heights[_i] + Re) ** 2 - Re ** 2) +
                                    np.sqrt((Orb_heights[_j] + Re) ** 2 - Re ** 2))

    # Compute positions and slant ranges
    ##############################################################
    direction = get_direction(Satellites)  # get both directions of the satellites to use the two transceivers
    Positions, meta = get_pos_vectors_omni(Satellites)  # position and plane of all the satellites
    slant_range = get_slant_range_optimized(Positions, N)  # matrix with all the distances between satellties
    slant_range_los = los_slant_range(slant_range, meta, Max_slnt_rng,
                                      Positions)  # distance matrix but if d>dMax, d=infinite
    shannonRate = get_data_rate(slant_range_los, interISL)  # max dataRate

    # Create edges for inter-plane links (closest east and west satellites)
    for i, sat in enumerate(Satellites):
        closest_east, closest_west = None, None
        min_east_distance, min_west_distance = float('inf'), float('inf')

        for j, other_sat in enumerate(Satellites):
            if sat.in_plane != other_sat.in_plane:
                if slant_range_los[i, j] < min_east_distance and Positions[j, 0] > Positions[i, 0]:  # East satellite
                    closest_east, min_east_distance = other_sat, slant_range_los[i, j]
                elif slant_range_los[i, j] < min_west_distance and Positions[j, 0] < Positions[i, 0]:  # West satellite
                    closest_west, min_west_distance = other_sat, slant_range_los[i, j]

        # Add edges for closest east and west satellites
        if closest_east:
            _A_Greedy.append(edge(sat.ID, closest_east.ID, min_east_distance, None, None,
                                  shannonRate[i, Satellites.index(closest_east)]))
        if closest_west:
            _A_Greedy.append(edge(sat.ID, closest_west.ID, min_west_distance, None, None,
                                  shannonRate[i, Satellites.index(closest_west)]))

    # intra-plane ISL links (upper and lower neighbors)
    ##############################################################
    for plane in earth.LEO:
        nPerPlane = len(plane.sats)
        for sat in plane.sats:
            sat.findIntraNeighbours(earth)

            # upper neighbour
            i = sat.in_plane * nPerPlane + sat.i_in_plane
            j = sat.upper.in_plane * nPerPlane + sat.upper.i_in_plane

            _A_Greedy.append(edge(sat.ID, sat.upper.ID,  # satellites IDs
                                  slant_range_los[i, j],  # distance between satellites
                                  None, None,  # directions
                                  shannonRate[i, j]))  # Max dataRate

            # lower neighbour
            j = sat.lower.in_plane * nPerPlane + sat.lower.i_in_plane

            _A_Greedy.append(edge(sat.ID, sat.lower.ID,  # satellites IDs
                                  slant_range_los[i, j],  # distance between satellites
                                  None, None,  # directions
                                  shannonRate[i, j]))  # Max dataRate

    return _A_Greedy


def deleteDuplicatedLinks(satA, g, earth):
    '''
    Given a satellite, searches for its east and west neighbour. If the east or west link is duplicated,
    it will remove the link with a higher latitude difference, keeping the horizontal links
    '''

    def getMostHorizontal(currentSat, satA, satB):
        '''
        Chooses the dat with the closest latitude to currentSat
        '''
        return (satA, satB) if abs(satA.latitude - currentSat.latitude) < abs(
            satB.latitude - currentSat.latitude) else (satB, satA)

    linkedSats = {'U': None, 'D': None, 'R': None, 'L': None}
    for edge in list(g.edges(satA.ID)):
        if edge[1][0].isdigit():
            satB = findByID(earth, edge[1])
            dir = getDirection(satA, satB)

            if (dir == 3):  # Found Satellite at East
                if linkedSats['R'] is not None:
                    # print(f"{satA.ID} east satellite duplicated: {linkedSats['R'].ID}, {satB.ID}")
                    most_horizontal, less_horizontal = getMostHorizontal(satA, linkedSats['R'], satB)
                    # print(f'Keeping most horizontal link: {most_horizontal.ID}')
                    linkedSats['R'] = most_horizontal
                    # remove pair from G
                    g.remove_edge(satA.ID, less_horizontal.ID)
                else:
                    linkedSats['R'] = satB

            elif (dir == 4):  # Found Satellite at West
                if linkedSats['L'] is not None:
                    # print(f"{satA.ID} West satellite duplicated: {linkedSats['L'].ID}, {satB.ID}")
                    most_horizontal, less_horizontal = getMostHorizontal(satA, linkedSats['L'], satB)
                    # print(f'Keeping most horizontal link: {most_horizontal.ID}')
                    linkedSats['L'] = most_horizontal
                    # remove pair from G
                    g.remove_edge(satA.ID, less_horizontal.ID)
                else:
                    linkedSats['L'] = satB


def establishRemainingISLs(earth, g):
    Satellites = []

    # Collect all satellites from each plane
    for plane in earth.LEO:
        for sat in plane.sats:
            Satellites.append(sat)

    # Gather positions and other parameters
    Positions, meta = get_pos_vectors_omni(Satellites)
    direction = get_direction(Satellites)
    slant_range = get_slant_range_optimized(Positions, len(Satellites))

    # Prepare link parameters
    interISL = RFlink(
        frequency=f,
        bandwidth=B,
        maxPtx=maxPtx,
        aDiameterTx=Adtx,
        aDiameterRx=Adrx,
        pointingLoss=pL,
        noiseFigure=Nf,
        noiseTemperature=Tn,
        min_rate=min_rate
    )

    # Calculate maximum slant range
    Max_slnt_rng = np.zeros((len(earth.LEO), len(earth.LEO)))
    Orb_heights = [plane.h for plane in earth.LEO]
    for i in range(len(earth.LEO)):
        for j in range(len(earth.LEO)):
            Max_slnt_rng[i, j] = (np.sqrt((Orb_heights[i] + Re) ** 2 - Re ** 2) +
                                  np.sqrt((Orb_heights[j] + Re) ** 2 - Re ** 2))

    # Define slant range and data rate matrices
    slant_range_los = los_slant_range(slant_range, meta, Max_slnt_rng, Positions)
    shannonRate = get_data_rate(slant_range_los, interISL)

    # Identify satellites with specific missing neighbors
    satellites_with_no_right = {sat: Positions[idx] for idx, sat in enumerate(Satellites) if sat.right is None}
    satellites_with_no_left = {sat: Positions[idx] for idx, sat in enumerate(Satellites) if sat.left is None}

    # Calculate potential matches sorted by horizontal alignment
    potential_links = []
    for sat_r in satellites_with_no_right:
        for sat_l in satellites_with_no_left:
            if sat_r.in_plane != sat_l.in_plane:
                idx_r = Satellites.index(sat_r)
                idx_l = Satellites.index(sat_l)
                if slant_range_los[idx_r, idx_l] < math.inf:
                    # Handle longitude wrapping correctly
                    longitude_difference = (satellites_with_no_left[sat_l][0] - satellites_with_no_right[sat_r][
                        0] + 360) % 360
                    if longitude_difference > 0 and longitude_difference < 180:
                        # lat_diff = abs(satellites_with_no_right[sat_r][1] - satellites_with_no_left[sat_l][1])
                        lat_diff = abs(sat_r.latitude - sat_l.latitude)
                        potential_links.append((lat_diff, sat_r, sat_l, slant_range_los[idx_r, idx_l]))

    # Sort by latitude difference to prioritize horizontal links
    # potential_links.sort()
    potential_links.sort(key=lambda x: x[0])  # Uses latitude difference as sort key

    # Establish links from closest to farthest in terms of horizontal alignment
    for lat_diff, sat_r, sat_l, distance in potential_links:
        if sat_r.right is None and sat_l.left is None:
            # ``distance`` is given in meters → store kilometres on the edge
            g.add_edge(
                sat_r.ID,
                sat_l.ID,
                slant_range=distance / 1000.0,
                dataRate=1 / shannonRate[Satellites.index(sat_r), Satellites.index(sat_l)],
                dataRateOG=shannonRate[Satellites.index(sat_r), Satellites.index(sat_l)],
                hop=1,
                type='ISL',
            )
            sat_r.right = sat_l
            sat_l.left = sat_r
            # print(f"Established horizontal link between {sat_r.ID} (right) and {sat_l.ID} (left) with latitude difference {lat_diff:.2f} deg and distance: {distance/1000:.2F} km.")

    return g


def createGraph(earth, matching='Greedy'):
    '''
    Each satellite has two transceiver antennas that are connected to the closest satellite in east and west direction to a satellite
    from another plane (inter-ISL). Each satellite also has anoteher two transceiver antennas connected to the previous and to the
    following satellite at their orbital plane (intra-ISL).
    A graph is created where each satellite is a node and each connection is an edge with a specific weight based either on the
    inverse of the maximum data rate achievable, total distance or number of hops.
    '''
    g = nx.Graph()

    # add LEO constellation
    ###############################
    for plane in earth.LEO:
        for sat in plane.sats:
            g.add_node(sat.ID, sat=sat)

    # add gateways and GSL edges
    ###############################
    for GT in earth.gateways:
        if GT.linkedSat[1]:
            g.add_node(GT.name, GT=GT)  # add GT as node
            # ``GT.linkedSat[0]`` is provided in meters → store kilometres
            g.add_edge(
                GT.name,
                GT.linkedSat[1].ID,  # add GT linked sat as edge
                slant_range=GT.linkedSat[0] / 1000.0,  # slant range in km
                invDataRate=1 / GT.dataRate,  # Inverse of dataRate
                dataRateOG=GT.dataRate,  # original shannon dataRate
                hop=1,
                type='GSL',  # in case we just want to count hops
            )

    # add inter-ISL and intra-ISL edges
    ###############################
    if matching == 'Markovian':
        markovEdges = markovianMatchingTwo(earth)
    elif matching == 'Greedy':
        markovEdges = greedyMatching(earth)
    print(f'Matching: {matching}')
    # print('----------------------------------')

    global biggestDist
    global firstMove
    # biggestDist = -1
    for markovEdge in markovEdges:
        g.add_edge(
            markovEdge.i,
            markovEdge.j,  # source and destination IDs
            slant_range=markovEdge.slant_range / 1000.0,  # store km
            dataRate=1 / markovEdge.shannonRate if markovEdge.shannonRate > 0 else float('inf'),
            # Inverse of dataRate # Fixed: handle zero shannonRate
            dataRateOG=markovEdge.shannonRate,  # Original shannon datRate
            hop=1,  # in case we just want to count hops
            dij=markovEdge.dij,
            dji=markovEdge.dji,
            type='ISL',
        )
        if firstMove and markovEdge.slant_range > biggestDist:  # keep the biggest possible distance for the normalization of the rewards
            biggestDist = markovEdge.slant_range

    # remove duplicated links and keep the most horizontal ones
    print('Removing duplicated links...')
    for plane in earth.LEO:
        for sat in plane.sats:
            deleteDuplicatedLinks(sat, g, earth)

    earth.graph = g

    # update the neighbors
    for plane in earth.LEO:
        for sat in plane.sats:
            sat.findIntraNeighbours(earth)
            sat.findInterNeighbours(earth)

    print('Establishing remaining edges...')
    g = establishRemainingISLs(earth, g)

    if firstMove:
        print(f'Biggest slant range between satellites: {biggestDist / 1000:.2f} km')
        firstMove = False
    print('----------------------------------')

    return g


def build_terrestrial_graph(json_path: str, gateways_csv: str, k_nearest: int = 3, rate_normal: float = 5e9,
                            rate_seacable: float = 10e9, rate_gateway: float = 5e9):
    """
    Costruisce il grafo terrestre a partire dal JSON TopoHub (node-link),
    rinomina i nodi usando 'name' come chiave (stringa),
    assegna 'distance_km', 'dataRate'/'data_rate', 'propDelay'/'prop_delay' e 'weight' agli archi,
    e collega i gateway del CSV al backbone (edge type='gateway_backhaul').
    """

    C_FIBER = 2e8  # m/s (≈ effective speed in fiber)

    # --- 1) carica TopoHub JSON e costruisci grafo di base ---
    with open(json_path, "r", encoding="utf-8") as f:
        topo = json.load(f)

    rawG = nx.node_link_graph(topo)

    # --- 1b) rinomina i nodi usando 'name' come chiave stringa ---
    id2name = {}
    name_counts = {}  # Track duplicate names
    for n, nd in rawG.nodes(data=True):
        nm = nd.get("name")
        if not nm:
            # fallback robusto: sintetizza un nome leggibile
            t = nd.get("type", "Node")
            nm = f"{t} {n}"
            nd["name"] = nm

        # Handle duplicate names by adding geographic suffix
        if nm in name_counts:
            name_counts[nm] += 1
            # Add geographic suffix based on coordinates
            pos = nd.get("pos", [0, 0])
            if len(pos) >= 2:
                lon, lat = pos[0], pos[1]
                if lat > 0:
                    suffix = " (North)"
                else:
                    suffix = " (South)"
                if lon > 0:
                    suffix += " East"
                else:
                    suffix += " West"
                nm = f"{nm}{suffix}"
            else:
                nm = f"{nm} {name_counts[nm]}"
        else:
            name_counts[nm] = 1

        id2name[n] = str(nm)

    rawG = nx.relabel_nodes(rawG, id2name, copy=True)

    # --- 1c) normalizza in Graph (no multiarco) mergiando gli attributi ---
    G = nx.Graph()
    G.add_nodes_from(rawG.nodes(data=True))
    for u, v, d in rawG.edges(data=True):
        if G.has_edge(u, v):
            G[u][v].update(d)
        else:
            G.add_edge(u, v, **d)

    # --- 2) pos -> lon/lat sui nodi ---
    for n, nd in G.nodes(data=True):
        p = nd.get("pos")
        if isinstance(p, (list, tuple)) and len(p) == 2:
            try:
                nd["lon"] = float(p[0])
                nd["lat"] = float(p[1])
            except Exception:
                pass

    TYPE2RATE = {
        "normal": rate_normal,
        "seacable": rate_seacable,
    }

    for u, v, ed in G.edges(data=True):
        lu, la = G.nodes[u].get("lon"), G.nodes[u].get("lat")
        lv, lb = G.nodes[v].get("lon"), G.nodes[v].get("lat")
        dist_km = haversine(lu, la, lv, lb) if None not in (lu, la, lv, lb) else 1.0
        ed["distance_km"] = float(dist_km)

        etype = str(ed.get("type", "normal")).lower()
        rate = float(TYPE2RATE.get(etype, rate_normal))
        ed["dataRate"] = rate
        ed["data_rate"] = rate

        # ritardo di propagazione coerente (anche alias Camel/snake)
        prop = (dist_km * 1000.0) / C_FIBER
        ed["propDelay"] = float(prop)
        ed["prop_delay"] = float(prop)

        # compat: alcuni pezzi di codice leggono 'slant_range'
        ed.setdefault("slant_range", float(dist_km))

        # peso per shortest-path
        ed["weight"] = float(dist_km)

    # --- 4) add/connect CSV gateways to nearest City ---
    gdf = pd.read_csv(gateways_csv)  # columns: Location, Longitude, Latitude
    for _, row in gdf.iterrows():
        g_name = str(row["Location"])
        g_lon = float(row["Longitude"])
        g_lat = float(row["Latitude"])

        if g_name not in G:
            G.add_node(g_name, type="Gateway", lon=g_lon, lat=g_lat)

        # find the nearest City
        best_node, best_d = None, float("inf")
        for n, nd in G.nodes(data=True):
            if nd.get("type") != "City":
                continue
            lon, lat = nd.get("lon"), nd.get("lat")
            if lon is None or lat is None:
                continue
            d = haversine(g_lon, g_lat, lon, lat)
            if d < best_d:
                best_d, best_node = d, n
        if best_node is None:
            continue

        prop = (best_d * 1000.0) / C_FIBER
        G.add_edge(
            g_name,
            best_node,
            type="gateway_backhaul",
            dataRate=float(rate_gateway),
            data_rate=float(rate_gateway),
            distance_km=float(best_d),
            propDelay=float(prop),
            prop_delay=float(prop),
            slant_range=float(best_d),
            weight=float(best_d),
        )

    # --- 5) connect each gateway to k nearest Cities (if k>1) ---
    if k_nearest and k_nearest > 1:
        for _, row in gdf.iterrows():
            g_name = str(row["Location"])
            g_lon = float(row["Longitude"])
            g_lat = float(row["Latitude"])

            dists = []
            for n, nd in G.nodes(data=True):
                if n == g_name or nd.get("type") != "City":
                    continue
                lon, lat = nd.get("lon"), nd.get("lat")
                if lon is None or lat is None:
                    continue
                dists.append((haversine(g_lon, g_lat, lon, lat), n))
            dists.sort(key=lambda x: x[0])

            for d, n2 in dists[:k_nearest]:
                prop = (d * 1000.0) / C_FIBER
                if G.has_edge(g_name, n2):
                    ed = G[g_name][n2]
                    ed["type"] = "gateway_backhaul"
                    ed["dataRate"] = float(rate_gateway)
                    ed["data_rate"] = float(rate_gateway)
                    ed["distance_km"] = float(d)
                    ed["propDelay"] = float(prop)
                    ed["prop_delay"] = float(prop)
                    ed.setdefault("slant_range", float(d))
                    ed["weight"] = float(d)
                else:
                    G.add_edge(
                        g_name,
                        n2,
                        type="gateway_backhaul",
                        dataRate=float(rate_gateway),
                        data_rate=float(rate_gateway),
                        distance_km=float(d),
                        propDelay=float(prop),
                        prop_delay=float(prop),
                        slant_range=float(d),
                        weight=float(d),
                    )

    return G


def draw_terrestrial_graph(G, gateways_csv="Gateways.csv", highlight_path=None):
    # Costruisci dizionario posizioni
    pos = {}
    for node, data in G.nodes(data=True):
        if "pos" in data and isinstance(data["pos"], (list, tuple)) and len(data["pos"]) == 2:
            lon, lat = data["pos"]
            pos[node] = (lon, lat)

    # Aggiungi i gateway dal CSV
    gateways_df = pd.read_csv(gateways_csv)
    for _, row in gateways_df.iterrows():
        name = row["Location"]
        lon = row["Longitude"]
        lat = row["Latitude"]
        pos[name] = (lon, lat)

    # Classi di nodi
    node_types = nx.get_node_attributes(G, 'type')
    cities = [n for n in G.nodes if node_types.get(n) == 'City']
    landings = [n for n in G.nodes if node_types.get(n) == 'Seacable Landing Point']
    waypoints = [n for n in G.nodes if node_types.get(n) == 'Seacable Waypoint']
    gateways = gateways_df["Location"].tolist()

    plt.figure(figsize=(14, 7))

    # Disegna archi
    nx.draw_networkx_edges(G, pos, alpha=0.2)

    # Disegna nodi
    nx.draw_networkx_nodes(G, pos, nodelist=cities, node_color="green", node_size=15, label="City")
    nx.draw_networkx_nodes(G, pos, nodelist=landings, node_color="blue", node_size=15, label="Landing Point")
    nx.draw_networkx_nodes(G, pos, nodelist=waypoints, node_color="gray", node_size=10, label="Waypoint")

    # Disegna i gateway come croci rosse
    nx.draw_networkx_nodes(G, pos, nodelist=gateways, node_color="red", node_shape="X", node_size=100, label="Gateway")

    # Evidenzia percorso se presente
    if highlight_path and len(highlight_path) > 1:
        edge_path = list(zip(highlight_path[:-1], highlight_path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=edge_path, edge_color="orange", width=2, label="Path")

    plt.title("Terrestrial Backbone con Gateway evidenziati")
    plt.legend()
    plt.axis("off")
    plt.show()


def attach_gateways_to_space_graph(earth):
    """
    Aggiunge (o aggiorna) gli archi GSL (Gateway<->Satellite) nel grafo spaziale,
    assicurandosi che abbiano dataRateOG e slant_range per il calcolo dei costi.
    Ritorna il numero di gateway connessi al grafo spaziale.
    """
    spaceG = getattr(earth, "space_graph", None)
    if spaceG is None:
        return 0

    attached = 0
    for gt in getattr(earth, "gateways", []):
        # GT.linkedSat is a tuple: (satellite, distance_km?) in the original
        sat = None
        try:
            sat = gt.linkedSat[1] if isinstance(gt.linkedSat, (list, tuple)) and len(gt.linkedSat) > 1 else \
            gt.linkedSat[0]
        except Exception:
            sat = gt.linkedSat[0] if (hasattr(gt, "linkedSat") and gt.linkedSat) else None

        if sat is None:
            continue

        # assicura che i nodi esistano nel grafo spaziale
        if gt.name not in spaceG:
            spaceG.add_node(gt.name)
        if sat.ID not in spaceG:
            spaceG.add_node(sat.ID)

        # distance/prop: try to derive a realistic slant range
        # attempts in order: existing attributes, sat.GTDist, fallback 0
        slant_km = None
        try:
            # if edge already exists, try to reuse the value
            if spaceG.has_edge(gt.name, sat.ID):
                ed0 = spaceG.get_edge_data(gt.name, sat.ID)
                slant_km = ed0.get("slant_range") or ed0.get("distance_km")
        except Exception:
            pass
        if slant_km is None:
            slant_km = getattr(gt, "GTDist", None) or getattr(sat, "GTDist", None)
        if slant_km is None:
            slant_km = 0.0

        # rate: usa il downRate del sat (dopo adjustDownRate) oppure un fallback
        down_bps = None
        try:
            # if edge already exists, try to reuse
            if spaceG.has_edge(gt.name, sat.ID):
                ed0 = spaceG.get_edge_data(gt.name, sat.ID)
                down_bps = ed0.get("dataRateOG") or ed0.get("dataRate")
        except Exception:
            pass
        if down_bps is None:
            down_bps = getattr(sat, "downRate", None)
        if down_bps in (None, 0):
            # fallback prudente: 100 Mbps
            down_bps = 1e8

        # create/update edge in both directions (not always needed, but convenient)
        spaceG.add_edge(gt.name, sat.ID,
                        type="GSL",
                        slant_range=float(slant_km),
                        distance_km=float(slant_km),
                        dataRateOG=float(down_bps))
        spaceG.add_edge(sat.ID, gt.name,
                        type="GSL",
                        slant_range=float(slant_km),
                        distance_km=float(slant_km),
                        dataRateOG=float(down_bps))
        attached += 1

    return attached


def getShortestPath(source, destination, weight, g):
    '''
    Gives you the shortest path between a source and a destination and plots it if desired.
    Uses the 'dijkstra' algorithm to compute the sortest path, where the total weight of the path can be either the sum of inverse
    of the maximumm dataRate achevable, the total slant range or the number of hops taken between source and destination.

    returns a list where each element is a sublist with the name of the node, its longitude and its latitude.
    '''

    path = []
    try:
        # Check if source and destination exist in the graph
        if source not in g:
            return -1
        if destination not in g:
            return -1

        shortest = nx.shortest_path(g, source, destination,
                                    weight=weight)  # computes the shortest path [dataRate, slant_range, hops]
        for hop in shortest:  # pre process the data so it can be used in the future
            key = list(g.nodes[hop])[0]
            if shortest.index(hop) == 0 or shortest.index(hop) == len(shortest) - 1:
                path.append([hop, g.nodes[hop][key].longitude, g.nodes[hop][key].latitude])
            else:
                path.append([hop, math.degrees(g.nodes[hop][key].longitude), math.degrees(g.nodes[hop][key].latitude)])
    except Exception as e:
        print(f"getShortestPath Caught an exception: {e}")
        print('No path between ' + source + ' and ' + destination + ', check the graph to see more details.')
        return -1
    return path


def getShortestPathTerrestrial(src_name, dst_name, G):
    if src_name not in G or dst_name not in G:
        return []
    try:
        return nx.shortest_path(G, src_name, dst_name, weight=lambda u, v, ed: _terr_edge_cost(u, v, ed))
    except Exception:
        return []


def plotShortestPath(earth, path, outputPath, ID=None, time=None):
    output_file = outputPath + 'popMap_' + path[0][0] + '_to_' + path[len(path) - 1][0] + '.png'
    earth.plotMap(True, True, path=path, ID=ID, time=time, save=True, fileName=output_file)
    plt.close()


def plotPathClean(earth, path_nodes, src_name, dst_name, output_file):
    """
    Nuova funzione per plottare i path con mappa del mondo realistica.
    Gestisce correttamente i path ibridi e terrestri.
    """

    fig = plt.figure(figsize=(16, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Imposta i limiti della mappa
    ax.set_global()

    # Aggiungi le caratteristiche geografiche realistiche
    ax.add_feature(cfeature.OCEAN, color='#E6F3FF', alpha=0.8)
    ax.add_feature(cfeature.LAND, color='#90EE90', alpha=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, alpha=0.7)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, alpha=0.8)
    ax.add_feature(cfeature.RIVERS, linewidth=0.3, alpha=0.5)
    ax.add_feature(cfeature.LAKES, color='#E6F3FF', alpha=0.8)

    # Griglia
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
                 alpha=0.5, linestyle='--', linewidth=0.5)

    x_coords = []
    y_coords = []
    node_info = []

    satellite_ids = [str(sat.ID) for plane in earth.LEO for sat in plane.sats]

    gateway_nodes = set()
    if hasattr(earth, 'gateways') and earth.gateways:
        for gw in earth.gateways:
            if hasattr(gw, 'name'):
                gateway_nodes.add(gw.name)
            elif hasattr(gw, 'city'):
                gateway_nodes.add(gw.city)

    # Aggiungi anche i nodi che sono sia nel grafo terrestre che usati per accesso satelliti
    for node in path_nodes:
        if node in satellite_ids:
            # If it's a satellite, check if the previous or next node is terrestrial
            node_idx = path_nodes.index(node)
            if node_idx > 0 and path_nodes[node_idx - 1] not in satellite_ids:
                gateway_nodes.add(path_nodes[node_idx - 1])
            if node_idx < len(path_nodes) - 1 and path_nodes[node_idx + 1] not in satellite_ids:
                gateway_nodes.add(path_nodes[node_idx + 1])

    # Gateway nodes identified

    for i, node in enumerate(path_nodes):
        # Trova le coordinate del nodo
        coords = None

        # Cerca nei nodi terrestri
        if hasattr(earth, 'terr_graph') and earth.terr_graph and node in earth.terr_graph:
            node_data = earth.terr_graph.nodes[node]
            lon = node_data.get('lon')
            lat = node_data.get('lat')
            if lon is not None and lat is not None:
                coords = (float(lon), float(lat))  # Formato corretto: (lon, lat)
                # Debug: stampa i dati raw del nodo
                if 'Poznań' in node or 'Sydney' in node or 'Tokyo' in node:
                    pass  # Debug block removed

        # Cerca nei satelliti
        if coords is None and node in satellite_ids:
            for plane in earth.LEO:
                for sat in plane.sats:
                    if str(sat.ID) == node:
                        coords = (math.degrees(sat.longitude), math.degrees(sat.latitude))
                        break
                if coords:
                    break

        if coords:
            lon, lat = coords
            # Usa direttamente le coordinate geografiche
            x_coords.append(lon)
            y_coords.append(lat)
            node_info.append((node, i, lon, lat, lon, lat))

            # Node {i}: {node} -> geo({lon:.2f}, {lat:.2f})

            # Debug: verify if coordinates seem correct for some known cities
            if 'Poznań' in node or 'Sydney' in node or 'Tokyo' in node:
                # Known city {node}: expected vs actual coordinates
                if 'Poznań' in node:
                    # Poznań expected: ~(16.93, 52.41), actual: ({lon:.2f}, {lat:.2f})
                    pass
                elif 'Sydney' in node:
                    # Sydney expected: ~(151.21, -33.87), actual: ({lon:.2f}, {lat:.2f})
                    pass
                elif 'Tokyo' in node:
                    # Tokyo expected: ~(139.69, 35.68), actual: ({lon:.2f}, {lat:.2f})
                    pass

    if len(x_coords) < 2:
        # Not enough valid coordinates, skipping plot
        plt.close()
        return

    # Determine if it's a hybrid path
    has_satellites = any(node in satellite_ids for node in path_nodes)

    if has_satellites:
        ax.plot(x_coords, y_coords, color='blue', linewidth=3, alpha=0.8, zorder=10,
                transform=ccrs.PlateCarree(), label='Hybrid Path')
        path_type = "Hybrid Communication Path"
    else:
        ax.plot(x_coords, y_coords, color='red', linewidth=2, alpha=0.8, zorder=10,
                transform=ccrs.PlateCarree(), label='Terrestrial Path')
        path_type = "Terrestrial Communication Path"

    # Disegna i nodi
    for i, (node, idx, lon, lat, px, py) in enumerate(node_info):
        if i == 0:  # Start
            ax.scatter(lon, lat, marker='o', c='green', s=100, linewidth=3,
                       edgecolors='black', zorder=20, transform=ccrs.PlateCarree(), label='Start')
        elif i == len(node_info) - 1:  # End
            ax.scatter(lon, lat, marker='o', c='red', s=100, linewidth=3,
                       edgecolors='black', zorder=20, transform=ccrs.PlateCarree(), label='End')
        elif node in gateway_nodes:  # Gateway
            ax.scatter(lon, lat, marker='x', c='red', s=80, linewidth=3,
                       zorder=18, alpha=0.9, transform=ccrs.PlateCarree(), label='Gateway' if i == 2 else "")
        else:  # Hops normali
            if has_satellites and node in satellite_ids:
                # Nodo satellitare
                ax.scatter(lon, lat, marker='o', c='purple', s=30, linewidth=1,
                           edgecolors='black', zorder=15, alpha=0.8, transform=ccrs.PlateCarree())
            else:
                # Nodo terrestre
                ax.scatter(lon, lat, marker='o', c='cyan', s=40, linewidth=1,
                           edgecolors='black', zorder=15, alpha=0.8, transform=ccrs.PlateCarree())

    # Titolo
    plt.title(f"{path_type}: {src_name} → {dst_name}\n"
              f"Hops: {len(path_nodes)} | Distance: {len(path_nodes) - 1} links",
              fontsize=14, weight='bold', pad=20)

    # Legenda
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, markeredgecolor='black',
                   markeredgewidth=2),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, markeredgecolor='black',
                   markeredgewidth=2),
        plt.Line2D([0], [0], marker='x', color='red', markersize=8, markeredgewidth=3),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='cyan', markersize=6, markeredgecolor='black',
                   markeredgewidth=1),
        plt.Line2D([0], [0], color='red' if not has_satellites else 'blue', linewidth=3)
    ]
    legend_labels = ['Start', 'End', 'Gateway', 'Terrestrial Hop', 'Path']

    if has_satellites:
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=6, markeredgecolor='black',
                       markeredgewidth=1))
        legend_labels.append('Satellite Hop')

    plt.legend(legend_elements, legend_labels, loc='lower left', fontsize=10)

    # Salva
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    # Plot saved as: {output_file}
    plt.close()


###############################################################################
#########################    Q-Tables - StateSpace    #########################
###############################################################################

def findByID(earth, satID):
    '''
    given the ID of a satellite, this function will return the corresponding satellite object
    '''
    for plane in earth.LEO:
        for sat in plane.sats:
            if (sat.ID == satID):
                return sat


def computeOutliers(g):
    '''
    Given a graph, will return the throughput and slant range thresholds that will be used to find the outliers
    (Devices with bad conditions)
    '''
    # define outliers
    slantRanges = []
    dataRates = []

    for edge in list(g.edges()):
        slantRanges.append(g.edges[edge]['slant_range'])
        dataRates.append(g.edges[edge]['dataRateOG'])

    # Slant Range Outliers
    slantRanges = pd.Series(slantRanges)
    Q3 = slantRanges.describe()['75%']
    Q1 = slantRanges.describe()['25%']
    IQR = Q3 - Q1
    upperFence = Q3 + (1.5 * IQR)

    # Data Rate Outliers
    dataRates = pd.Series(dataRates)
    Q3 = dataRates.describe()['75%']
    Q1 = dataRates.describe()['25%']
    IQR = Q3 - Q1
    lowerFence = Q1 - (1.5 * IQR)

    return lowerFence, upperFence


def getQueues(sat, threshold=None, DDQN=False):
    '''
    When !DDQN, this function will return True if one of the satellite queues has a length over a limit or they are
    missing one link

    Each satellite has a queue for each link which includes both ISL and GSL (sat 2 GT). The Queues are implemented as
    tuples that contain a list of simpy events, a list of the data blocks, and the ID of the satellite for the link
    (there is no ID for the GT queues). The structure is tuple[list[Simpy.event], list[DataBlock], ID].
    The list of events will always have at least one event present which will be non-triggered when there are no blocks
    in the queue. When blocks are present, there will be as many triggered events as there are blocks.

    On the GTs, there is one queue which has the same structure as the queues for the GSLs on the satellites:
    tuple[list[Simpy.event], list[DataBlock]]

    ISLs Queues: sendBufferSats where each entry is a separate queue.
    GSLs Queues: sendBufferGT. While there will never be more than one queue in this list.
    GTs  Queues: sendBuffer which is just the tuple itself

    In our case we will just choose the highest queue of all the ISLs and compare it to a threshold

    The try excepts are for those cases where the linked satellite does not have the 4 linked satllites queues.
    IF THE SATELLITE DOES NOT HAVE 4 LINEKD SATELLITES IT WILL BE CONSIDERED AS HIGH QUEUE
    '''
    queuesLen = []
    infQueue = False
    queuesDic = {'U': np.inf,
                 'D': np.inf,
                 'R': np.inf,
                 'L': np.inf}
    try:
        queuesLen.append(len(sat.sendBufferSatsIntra[0][1]))
        queuesDic['U'] = len(sat.sendBufferSatsIntra[0][1])
    except (IndexError, AttributeError):
        infQueue = True
    try:
        queuesLen.append(len(sat.sendBufferSatsIntra[1][1]))
        queuesDic['D'] = len(sat.sendBufferSatsIntra[1][1])

    except (IndexError, AttributeError):
        infQueue = True
    try:
        queuesLen.append(len(sat.sendBufferSatsInter[0][1]))
        queuesDic['R'] = len(sat.sendBufferSatsInter[0][1])
    except (IndexError, AttributeError):
        infQueue = True
    try:
        queuesLen.append(len(sat.sendBufferSatsInter[1][1]))
        queuesDic['L'] = len(sat.sendBufferSatsInter[1][1])
    except (IndexError, AttributeError):
        infQueue = True

    if not DDQN:
        return max(queuesLen) > threshold or infQueue
    else:
        return queuesDic


def hasBadConnection(satA, satB, thresholdSL, thresholdTHR, g):
    '''
    This function will return true if the satellites distance between them > trheshold or if their throughpuyt < trheshold
    They are far away or the link is weak
    '''
    slantRange = g.edges[satA.ID, satB.ID]['slant_range']
    throughputSats = g.edges[satA.ID, satB.ID]['dataRateOG']

    return (slantRange > thresholdSL or throughputSats < thresholdTHR)


def getSatScore(satA, satB, g):
    '''
    This function will compute the score of sending the package from satA to satB
    0: (Low  slant range || high throughput) && low queue
    1:  High slant range && low  throughput  && low queue
    2:  High queue

    Queue threshold:
    As high queue threshold we have set 125 packets, which is the 92 percentile of all the queues when we have 13 GTs
    (The moment when we start having congestion with slant range policy). The waiting time of a queue with 125 blocks
    is 9 msg (Each packet in the queue lasts ~0.072ms)
    '''
    thresholdQueue = 125
    thresholdTHR, thresholdSL = computeOutliers(g)

    if satB is None or getQueues(satB, thresholdQueue):
        return 2
    elif hasBadConnection(satA, satB, thresholdSL, thresholdTHR, g):
        return 1
    else:
        return 0


# @profile
def getDeepSatScore(queueLength):
    # return 1 if queueLength > infQueue else (int(np.floor(queueVals*np.log10(queueLength + 1)/np.log10(infQueue))))/queueVals
    return queueVals if queueLength > infQueue else int(
        np.floor(queueVals * np.log10(queueLength + 1) / np.log10(infQueue)))


def getDirection(satA, satB):
    '''
    Returns the direction from satA to satB, considering the Earth's wrap-around for longitude.
    '''

    def normalize_longitude(lon):
        # Normalize longitude to the range [-math.pi, math.pi]
        return ((lon + math.pi) % (2 * math.pi)) - math.pi

    planei = int(satA.in_plane)
    planej = int(satB.in_plane)

    if planei == planej:
        if satA.latitude < satB.latitude:
            return 1  # Go Upper
        else:
            return 2  # Go Lower

    # Normalize the longitudes
    norm_lonA = normalize_longitude(satA.longitude)
    norm_lonB = normalize_longitude(satB.longitude)

    # Calculate the normalized longitude difference
    lon_diff = normalize_longitude(norm_lonB - norm_lonA)

    # Decide direction based on normalized difference
    if lon_diff > 0:
        return 3  # Go Right
    else:
        return 4  # Go Left


def linkedSatsList(g):
    '''
    This funtion retunrs a dictionary (Gateway: linekdSatellite)
    '''
    linkedSats = []
    for node in g.nodes:
        if not node[0].isdigit():
            linkedSats.append(list(g.edges(node))[0])
    return pd.DataFrame(linkedSats)


def getDestination(Block, g, sat=None):
    '''
    Returns:
    blockDestination: Position of the satellite linked to the block destination Gateway among a list of all the
                      satellites linked to Gateways
    linkedGateway:    If the satellite provided is linked to a gateway, it will return the position of the satellite in
                      the mentioned list. Otherwise it will return -1.
    '''
    destination = list(g.edges(Block.destination.name))[0][1]  # ID of the Satellite linked to the block destination GT
    blockDestination = (linkedSatsList(g)[1] == destination).argmax()

    if sat is None:
        return blockDestination
    else:
        pass
        # satDest = Block.destination.linkedSat[1]
        # return getGridPosition(GridSize, [tuple([math.degrees(satDest.latitude), math.degrees(satDest.longitude), satDest.ID])], False, False)[0]


def getLinkedSats(satA, g, earth):
    '''
    Given a satellite the function will return a list with the linked satellite at each direction.
    If that direction has no linked satellite, it will be None
    At the graph each edge is a satA, satB pair with properties like dirij or dirji, i will always
    be the satellite of the lowest plane and 1 will be righ direction (East).

    SAT UP:      northest linked satellite
    SAT DOWN:    southest linked satellite
    SAT LEFT:    linked satellite with lower  plane ID
    SAT RIGHT:   linked satellite with higher plane ID
    '''
    linkedSats = {'U': None, 'D': None, 'R': None, 'L': None}
    for edge in list(g.edges(satA.ID)):
        if edge[1][0].isdigit():
            satB = findByID(earth, edge[1])
            dir = getDirection(satA, satB)

            if (dir == 1 and linkedSats['U'] is None):  # Found a satellite at north
                linkedSats['U'] = satB
            elif (dir == 1):  # Found second North, this sat is on South Pole
                if satB.latitude > linkedSats['U'].latitude:
                    # the satellite seen is more at north than Up one, so is set as new Up
                    linkedSats['D'] = linkedSats['U']
                    linkedSats['U'] = satB
                else:
                    # the satellite seen is less at north than Up one, so is set as Down
                    linkedSats['D'] = satB

            elif (dir == 2 and linkedSats['D'] is None):  # Found satellite at South
                linkedSats['D'] = satB
            elif (dir == 2):  # Found second Down, this sat is on North Pole
                if satB.latitude < linkedSats['D'].latitude:
                    linkedSats['U'] = linkedSats['D']
                    linkedSats['D'] = satB
                else:
                    linkedSats['U'] = satB

            elif (dir == 3):  # Found Satellite at East
                # if linkedSats['R'] is not None:
                #     print(f"{satA.ID} east satellite duplicated! Replacing {linkedSats['R'].ID} with {satB.ID}")
                linkedSats['R'] = satB

            elif (dir == 4):  # Found Satellite at West
                # if linkedSats['L'] is not None:
                #     print(f"{satA.ID} west satellite duplicated! Replacing {linkedSats['L'].ID} with {satB.ID}")
                linkedSats['L'] = satB

        else:
            pass
    return linkedSats


def getDeepLinkedSats(satA, g, earth):
    '''
    Given a satellite, this function will return a dictionary with the linked satellite
    at each direction based on the new definition of upper and lower satellites.
    Satellite at the right and left are determined based on inter-plane links.
    '''
    linkedSats = {'U': None, 'D': None, 'R': None, 'L': None}

    # Use the provided logic to find intra-plane neighbours (upper and lower)
    # satA.findIntraNeighbours(earth)
    linkedSats['U'] = satA.upper
    linkedSats['D'] = satA.lower
    linkedSats['R'] = satA.right
    linkedSats['L'] = satA.left

    # # Find inter-plane neighbours (right and left)
    # for edge in list(g.edges(satA.ID)):
    #     if edge[1][0].isdigit():
    #         satB = findByID(earth, edge[1])
    #         dir = getDirection(satA, satB)
    #         if(dir == 3):                                         # Found Satellite at East
    #             if linkedSats['R'] is not None:
    #                 print(f"{satA.ID} east satellite duplicated! Replacing {linkedSats['R'].ID} with {satB.ID}")
    #             linkedSats['R']  = satB

    #         elif(dir == 4):                                       # Found Satellite at West
    #             if linkedSats['L'] is not None:
    #                 print(f"{satA.ID} west satellite duplicated! Replacing {linkedSats['L'].ID} with {satB.ID}")
    #             linkedSats['L']  = satB
    #     else:
    #         pass

    return linkedSats


def getState(Block, satA, g, earth):
    '''
    Given a dataBlock and the current satellite this function will return a list with the
    values of the 5 fields of the state space.
    Destination: linked satellite to the destination gateway index.

    we initialize the score of the satellites in 2 (worst case) because we do not know if they
    will actually have a linked satellite in that direction.
    If they have it the satellite score will replace the initialization score (2) but if they dont
    have it, as we need a score in order to set the state space we will give the worst score and
    send a None in the destinations dict. That action will be initialized with -infinite in the QTable
    '''
    destination = getDestination(Block, g)
    state = [2, 2, 2, 2, destination]

    state[0] = getSatScore(satA, satA.QLearning.linkedSats['U'], g)
    state[1] = getSatScore(satA, satA.QLearning.linkedSats['D'], g)
    state[2] = getSatScore(satA, satA.QLearning.linkedSats['R'], g)
    state[3] = getSatScore(satA, satA.QLearning.linkedSats['L'], g)

    return state


def getBiasedLatitude(sat):
    try:
        return (int(math.degrees(sat.latitude)) + latBias) / coordGran
    except AttributeError as e:
        # print(f"getBiasedLatitude Caught an exception: {e}")
        return notAvail


def getBiasedLongitude(sat):
    try:
        return (int(math.degrees(sat.longitude)) + lonBias) / coordGran
    except AttributeError as e:
        # print(f"getBiasedLongitude Caught an exception: {e}")
        return notAvail


def getDeepStateReduced(block, sat, linkedSats):
    satDest = block.destination.linkedSat[1]
    if satDest is None:
        print(f'{block.destination} has no linked satellite :(')
        return None
    return np.array([getBiasedLatitude(linkedSats['U']),  # Up link Positions
                     getBiasedLongitude(linkedSats['U']),
                     getBiasedLatitude(linkedSats['D']),  # Down link Positions
                     getBiasedLongitude(linkedSats['D']),
                     getBiasedLatitude(linkedSats['R']),  # Right link Positions
                     getBiasedLongitude(linkedSats['R']),
                     getBiasedLatitude(linkedSats['L']),  # Left link Positions
                     getBiasedLongitude(linkedSats['L']),
                     getBiasedLatitude(sat),  # Actual Latitude
                     getBiasedLongitude(sat),  # Actual Longitude
                     getBiasedLatitude(satDest),  # Destination Latitude
                     getBiasedLongitude(satDest)]).reshape(1, -1)  # Destination Longitude


def getDeepStateDiff(block, sat, linkedSats):
    def normalize_angle_diff(angle_diff):
        # Ensure the angle difference is within [-180, 180]
        return (angle_diff + 180) % 360 - 180

    def get_relative_position(neighbor_sat, current_coord, is_lat=True):
        # Convert and calculate relative position, considering the 180-degree discontinuity
        try:
            neighbor_coord = math.degrees(neighbor_sat.latitude if is_lat else neighbor_sat.longitude)
            current_coord = math.degrees(current_coord)
            diff = normalize_angle_diff(neighbor_coord - current_coord)
            return diff / coordGran
        except AttributeError:
            return notAvail

    def get_absolute_position(coord, bias, gran):
        # Convert absolute position to a normalized value within the specified range
        return (math.degrees(coord) + bias) / gran

    satDest = block.destination.linkedSat[1]
    if satDest is None:
        print(f'{block.destination} has no linked satellite :(')
        return None

    # Current coordinates
    current_lat = sat.latitude
    current_lon = sat.longitude

    # Queues
    queuesU = getQueues(linkedSats['U'], DDQN=True)
    queuesD = getQueues(linkedSats['D'], DDQN=True)
    queuesR = getQueues(linkedSats['R'], DDQN=True)
    queuesL = getQueues(linkedSats['L'], DDQN=True)

    state = [
        # Up link scores and positions
        getDeepSatScore(queuesU['U']),
        getDeepSatScore(queuesU['D']),
        getDeepSatScore(queuesU['R']),
        getDeepSatScore(queuesU['L']),
        get_relative_position(linkedSats['U'], current_lat, is_lat=True),
        get_relative_position(linkedSats['U'], current_lon, is_lat=False),

        # Down link scores and positions
        getDeepSatScore(queuesD['U']),
        getDeepSatScore(queuesD['D']),
        getDeepSatScore(queuesD['R']),
        getDeepSatScore(queuesD['L']),
        get_relative_position(linkedSats['D'], current_lat, is_lat=True),
        get_relative_position(linkedSats['D'], current_lon, is_lat=False),

        # Right link scores and positions
        getDeepSatScore(queuesR['U']),
        getDeepSatScore(queuesR['D']),
        getDeepSatScore(queuesR['R']),
        getDeepSatScore(queuesR['L']),
        get_relative_position(linkedSats['R'], current_lat, is_lat=True),
        get_relative_position(linkedSats['R'], current_lon, is_lat=False),

        # Left link scores and positions
        getDeepSatScore(queuesL['U']),
        getDeepSatScore(queuesL['D']),
        getDeepSatScore(queuesL['R']),
        getDeepSatScore(queuesL['L']),
        get_relative_position(linkedSats['L'], current_lat, is_lat=True),
        get_relative_position(linkedSats['L'], current_lon, is_lat=False),

        # Absolute current satellite's coordinates
        get_absolute_position(current_lat, latBias, coordGran),
        get_absolute_position(current_lon, lonBias, coordGran),

        # Destination's differential coordinates
        get_relative_position(satDest, current_lat, is_lat=True),
        get_relative_position(satDest, current_lon, is_lat=False)
    ]

    return np.array(state).reshape(1, -1)


def getDeepStateDiffLastHop(block, sat, linkedSats):
    def normalize_angle_diff(angle_diff):
        # Ensure the angle difference is within [-180, 180]
        return (angle_diff + 180) % 360 - 180

    def get_relative_position(neighbor_sat, current_coord, is_lat=True):
        # Convert and calculate relative position, considering the 180-degree discontinuity
        try:
            neighbor_coord = math.degrees(neighbor_sat.latitude if is_lat else neighbor_sat.longitude)
            current_coord = math.degrees(current_coord)
            diff = normalize_angle_diff(neighbor_coord - current_coord)
            return diff / coordGran
        except AttributeError:
            return notAvail

    def get_absolute_position(coord, bias, gran):
        # Convert absolute position to a normalized value within the specified range
        return (math.degrees(coord) + bias) / gran

    def get_last_satellite(block, sat):  # REVIEW if index here are the same as decision index
        '''This will return information about the last block hop in relation to the current satellite:
        -1: Constellation moved and the last block's satellite is not connected to current satellite
        0: Upper neighbour
        1: Lower neighbour
        2: Right Neighbour
        3: Left  Neighbour'''
        actIndex = -1
        try:
            if len(block.QPath) > 2:
                if sat.upper and sat.upper.ID == block.QPath[-2][0]:
                    actIndex = 0
                elif sat.lower and sat.lower.ID == block.QPath[-2][0]:
                    actIndex = 1
                elif sat.right and sat.right.ID == block.QPath[-2][0]:
                    actIndex = 2
                elif sat.left and sat.left.ID == block.QPath[-2][0]:
                    actIndex = 3
            return actIndex
        except AttributeError as e:
            print(f'An error occurred when checking if {block.QPath[-2][0]} is a neighbour satellite of {sat.ID}')
            return actIndex

    satDest = block.destination.linkedSat[1]
    if satDest is None:
        print(f'{block.destination} has no linked satellite :(')
        return None

    # Current coordinates
    current_lat = sat.latitude
    current_lon = sat.longitude

    # Queues
    queuesU = getQueues(linkedSats['U'], DDQN=True)
    queuesD = getQueues(linkedSats['D'], DDQN=True)
    queuesR = getQueues(linkedSats['R'], DDQN=True)
    queuesL = getQueues(linkedSats['L'], DDQN=True)

    state = [
        # Previous satellite information
        get_last_satellite(block, sat),
        # Up link scores and positions
        getDeepSatScore(queuesU['U']),
        getDeepSatScore(queuesU['D']),
        getDeepSatScore(queuesU['R']),
        getDeepSatScore(queuesU['L']),
        get_relative_position(linkedSats['U'], current_lat, is_lat=True),
        get_relative_position(linkedSats['U'], current_lon, is_lat=False),

        # Down link scores and positions
        getDeepSatScore(queuesD['U']),
        getDeepSatScore(queuesD['D']),
        getDeepSatScore(queuesD['R']),
        getDeepSatScore(queuesD['L']),
        get_relative_position(linkedSats['D'], current_lat, is_lat=True),
        get_relative_position(linkedSats['D'], current_lon, is_lat=False),

        # Right link scores and positions
        getDeepSatScore(queuesR['U']),
        getDeepSatScore(queuesR['D']),
        getDeepSatScore(queuesR['R']),
        getDeepSatScore(queuesR['L']),
        get_relative_position(linkedSats['R'], current_lat, is_lat=True),
        get_relative_position(linkedSats['R'], current_lon, is_lat=False),

        # Left link scores and positions
        getDeepSatScore(queuesL['U']),
        getDeepSatScore(queuesL['D']),
        getDeepSatScore(queuesL['R']),
        getDeepSatScore(queuesL['L']),
        get_relative_position(linkedSats['L'], current_lat, is_lat=True),
        get_relative_position(linkedSats['L'], current_lon, is_lat=False),

        # Absolute current satellite's coordinates
        get_absolute_position(current_lat, latBias, coordGran),
        get_absolute_position(current_lon, lonBias, coordGran),

        # Destination's differential coordinates
        get_relative_position(satDest, current_lat, is_lat=True),
        get_relative_position(satDest, current_lon, is_lat=False)
    ]

    return np.array(state).reshape(1, -1)


def getDeepState(block, sat, linkedSats):
    satDest = block.destination.linkedSat[1]
    if satDest is None:
        print(f'{block.destination} has no linked satellite :(')
        return None

    queuesU = getQueues(linkedSats['U'], DDQN=True)
    queuesD = getQueues(linkedSats['D'], DDQN=True)
    queuesR = getQueues(linkedSats['R'], DDQN=True)
    queuesL = getQueues(linkedSats['L'], DDQN=True)
    return np.array([getDeepSatScore(queuesU['U']),  # Up link scores
                     getDeepSatScore(queuesU['D']),
                     getDeepSatScore(queuesU['R']),
                     getDeepSatScore(queuesU['L']),
                     getBiasedLatitude(linkedSats['U']),  # Up link Positions
                     getBiasedLongitude(linkedSats['U']),
                     getDeepSatScore(queuesD['U']),  # Down link scores
                     getDeepSatScore(queuesD['D']),
                     getDeepSatScore(queuesD['R']),
                     getDeepSatScore(queuesD['L']),
                     getBiasedLatitude(linkedSats['D']),  # Down link Positions
                     getBiasedLongitude(linkedSats['D']),
                     getDeepSatScore(queuesR['U']),  # Right link scores
                     getDeepSatScore(queuesR['D']),
                     getDeepSatScore(queuesR['R']),
                     getDeepSatScore(queuesR['L']),
                     getBiasedLatitude(linkedSats['R']),  # Right link Positions
                     getBiasedLongitude(linkedSats['R']),
                     getDeepSatScore(queuesL['U']),  # Left link scores
                     getDeepSatScore(queuesL['D']),
                     getDeepSatScore(queuesL['R']),
                     getDeepSatScore(queuesL['L']),
                     getBiasedLatitude(linkedSats['L']),  # Left link Positions
                     getBiasedLongitude(linkedSats['L']),

                     # int(math.degrees(sat.latitude))+latBias,                    # Actual Latitude
                     # int(math.degrees(sat.longitude))+lonBias,                   # Actual Longitude
                     # int(math.degrees(satDest.latitude))+latBias,                # Destination Latitude
                     # int(math.degrees(satDest.longitude))+lonBias]).reshape(1,-1)# Destination Longitude

                     getBiasedLatitude(sat),  # Actual Latitude
                     getBiasedLongitude(sat),  # Actual Longitude
                     getBiasedLatitude(satDest),  # Destination Latitude
                     getBiasedLongitude(satDest)]).reshape(1, -1)  # Destination Longitude


def createQTable(NGT):
    '''
    Create a 6D numpy array to hold the current Q-values for each state and action pair: Q(s, a)
    The array contains 5 dimensions with the shape of the environment, as well as a 6th "action" dimension.
    The "action" dimension consists of 4 layers that will allow us to keep track of the Q-values for each possible action in each state
    The value of each (state, action) pair is initialized to 0.
    '''

    actions = ('N', 'S', 'E', 'W')
    satUp, satDown, satRight, satLeft = 3, 3, 3, 3
    Destination = NGT

    qValues = np.zeros((satUp, satDown, satRight, satLeft, Destination,
                        len(actions)))  # first 5 fields are states while 6th field is the action. 4050 values with 10 GTs

    return qValues


###############################################################################
##########################   Q-Learning - Rewards    ##########################
###############################################################################


# @profile
def getSlantRange(satA, satB):
    '''
    given 2 satellites, it will return the slant range between them (With the method used at 'get_slant_range_optimized')
    '''
    return np.linalg.norm(np.array((satA.x, satA.y, satA.z)) - np.array((satB.x, satB.y, satB.z)))  # posA - posB


# @profile
def getQueueReward(queueTime, w1):
    '''
    Given the queue time in seconds, this function will return the queue reward.
    With 125 packets, 9ms Queue (The thershold that we take to consider a queue as high) the reward will be -0.04 (with w1 = 2)
    '''
    return w1 * (1 - 10 ** queueTime)


# @profile
def getDistanceReward(satA, satB, destination, w2):
    '''
    This function will return the instant reward regarding to the slant range reduction from actual node to destination
    just after the agent takes an action (destination is the satellite linked to the destination Gateway)

    TSLa: Total slant range from sat A to destination
    TSLb: Total slant range from sat B to destination
    SLR : Slant Range reduction after taking the action (Going from satA to satB)

    Formula: w*(SLR + TSLa)/TSLa = w*(TSLa - TSLb + TSLa)/TSLa = w*(2*TSLa - TSLb)/TSLa
    '''
    balance = -1  # centralizes the result in 0

    TSLa = getSlantRange(satA, destination)
    TSLb = getSlantRange(satB, destination)
    return w2 * ((2 * TSLa - TSLb) / TSLa + balance)


def getDistanceRewardV2(sat, nextSat, satU, satD, satR, satL, destination, w2):
    '''
    Computes the reward by comparing how closer you get to the destination in terms of KM (SLr, Slant Range Reduction) with the
    average distance with all your neighbours (SLav, Slant Range average)
    If any of the linked satellites is not available, it is handled
    SLr/SLav + balance
    '''

    SLr = getSlantRange(sat, destination) - getSlantRange(nextSat, destination)
    SLU = SLD = SLR = SLL = 0
    count = 0

    # Calculate slant range for each satellite, if it is not None
    if satU is not None:
        SLU = getSlantRange(satU, sat)
        count += 1
    if satD is not None:
        SLD = getSlantRange(satD, sat)
        count += 1
    if satR is not None:
        SLR = getSlantRange(satR, sat)
        count += 1
    if satL is not None:
        SLL = getSlantRange(satL, sat)
        count += 1

    SLav = (SLU + SLD + SLR + SLL) / count if count > 0 else 0

    return w2 * (SLr / SLav) if SLav != 0 else 0


def getDistanceRewardV3(sat, nextSat, satU, satD, satR, satL, destination, w2):
    '''
    Returns the distance reward computed by comparing how closer you get to the destination in terms of KM (SLr, Slant Range Reduction) with
    how close you could get as maximum taking the other options going to any of the other neighbours (max(SLrs), max(Slant range reductions from all the neighbours))
    reward = SLr/max(SLs)
    '''
    SLr = getSlantRange(sat, destination) - getSlantRange(nextSat, destination)
    SLrs = []

    if satU is not None:
        SLrs.append(getSlantRange(sat, destination) - getSlantRange(satU, destination))
    if satD is not None:
        SLrs.append(getSlantRange(sat, destination) - getSlantRange(satD, destination))
    if satR is not None:
        SLrs.append(getSlantRange(sat, destination) - getSlantRange(satR, destination))
    if satL is not None:
        SLrs.append(getSlantRange(sat, destination) - getSlantRange(satL, destination))

    return w2 * SLr / max(SLrs)


def getDistanceRewardV4(sat, nextSat, satDest, w2, w4):
    global biggestDist
    SLr = getSlantRange(sat, satDest) - getSlantRange(nextSat, satDest)
    TravelDistance = getSlantRange(sat, nextSat)
    if TravelDistance > biggestDist:
        # print(f'Very big distance: {sat.ID}, {nextSat.ID}')
        pass
    return w2 * (SLr - TravelDistance / w4) / biggestDist
    # return w2*(SLr/biggestDist)
    # return w2*SLr/1000000


def getDistanceRewardV5(sat, nextSat, w2):
    SLr = getSlantRange(sat, nextSat)
    return w2 * SLr / 1000000


def saveHyperparams(outputPath, inputParams, hyperparams):
    print('Saving hyperparams at: ' + str(outputPath))
    hyperparams = ['Constellation: ' + str(inputParams['Constellation'][0]),
                   'Import QTables: ' + str(hyperparams.importQ),
                   'plotPath: ' + str(hyperparams.plotPath),
                   'Test length: ' + str(inputParams['Test length'][0]),
                   'Alphas: ' + str(hyperparams.alpha) + ', ' + str(alpha_dnn),
                   'Gamma: ' + str(hyperparams.gamma),
                   'Epsilon: ' + str(hyperparams.epsilon),
                   'Max epsilon: ' + str(hyperparams.MAX_EPSILON),
                   'Min epsilon: ' + str(hyperparams.MIN_EPSILON),
                   'Arrive Reward: ' + str(hyperparams.ArriveR),
                   'w1: ' + str(hyperparams.w1),
                   'w2: ' + str(hyperparams.w2),
                   'w4: ' + str(hyperparams.w4),
                   'againPenalty: ' + str(hyperparams.again),
                   'unavPenalty: ' + str(hyperparams.unav),
                   'Coords granularity: ' + str(hyperparams.coordGran),
                   'Update freq: ' + str(hyperparams.updateF),
                   'Batch Size: ' + str(hyperparams.batchSize),
                   'Buffer Size: ' + str(hyperparams.bufferSize),
                   'Hard Update: ' + str(hyperparams.hardUpdate),
                   'Exploration: ' + str(hyperparams.explore),
                   'DDQN: ' + str(hyperparams.ddqn),
                   'Latitude bias: ' + str(hyperparams.latBias),
                   'Longitude bias: ' + str(hyperparams.lonBias),
                   'Diff: ' + str(hyperparams.diff),
                   'Reduced State: ' + str(hyperparams.reducedState),
                   'Online phase: ' + str(hyperparams.online)]

    # save hyperparams
    with open(outputPath + 'hyperparams.txt', 'w') as f:
        for param in hyperparams:
            f.write(param + '\n')


def saveQTables(outputPath, earth):
    print('Saving Q-Tables at: ' + outputPath)
    # create output path if it does not exist
    path = outputPath + 'qTablesExport_' + str(len(earth.gateways)) + 'GTs/'
    os.makedirs(path, exist_ok=True)

    # save Q-Tables
    for plane in earth.LEO:
        for sat in plane.sats:
            qTable = sat.QLearning.qTable
            with open(path + sat.ID + '.npy', 'wb') as f:
                np.save(f, qTable)


def saveDeepNetworks(outputPath, earth):
    print('Saving Deep Neural networks at: ' + outputPath)
    os.makedirs(outputPath, exist_ok=True)
    if not onlinePhase:
        earth.DDQNA.qNetwork.save(outputPath + 'qNetwork_' + str(len(earth.gateways)) + 'GTs' + '.h5')
        if ddqn:
            earth.DDQNA.qTarget.save(outputPath + 'qTarget_' + str(len(earth.gateways)) + 'GTs' + '.h5')
    else:
        for plane in earth.LEO:
            for sat in plane.sats:
                sat.DDQNA.qNetwork.save(outputPath + sat.ID + 'qNetwork_' + str(len(earth.gateways)) + 'GTs' + '.h5')
                if ddqn:
                    sat.DDQNA.qTarget.save(outputPath + sat.ID + 'qTarget_' + str(len(earth.gateways)) + 'GTs' + '.h5')


###############################################################################
#########################    Simulation && Results    #########################
###############################################################################


def plotLatenciesBars(percentages, outputPath):
    '''
    Bar plot where each bar is a scenario with a different nº of gateways and where each color represents one of the three latencies.
    '''
    # plot percent stacked barplot
    barWidth = 0.85
    r = percentages['GTnumber']
    numbers = percentages['GTnumber']
    GTnumber = len(r)

    plt.bar(r, percentages['Propagation time'], color='#b5ffb9', edgecolor='white', width=barWidth,
            label="Propagation time")  # Propagation time
    plt.bar(r, percentages['Queue time'], bottom=percentages['Propagation time'], color='#f9bc86',  # Queue time
            edgecolor='white', width=barWidth, label="Queue time")
    plt.bar(r, percentages['Transmission time'],
            bottom=[i + j for i, j in zip(percentages['Propagation time'],  # Tx time
                                          percentages['Queue time'])], color='#a3acff', edgecolor='white',
            width=barWidth, label="Transmission time")

    # Custom x axis
    plt.xticks(numbers)
    plt.xlabel("Nº of gateways")
    plt.ylabel('Latency')

    # Add a legend
    plt.legend(loc='lower left')

    # Show and save graphic
    plt.savefig(outputPath + 'Percentages_{}_Gateways.png'.format(GTnumber + 1))
    # plt.show()
    plt.close()


def plotQueues(queue_fractions, outputPath, n_active):
    """
    Generate CDF plot showing what fraction of total latency is due to queuing.

    queue_fractions: list of values in [0,1] representing queue_time/total_latency for each block.

    Interpretation:
    - Values near 0: Latency dominated by propagation/transmission (good)
    - Values near 1: Latency dominated by queuing (network congestion)
    """
    try:
        if not queue_fractions:
            print("[plotQueues] No queue data available: skipping.")
            return

        xs = np.sort(np.asarray(queue_fractions, dtype=float))
        xs = xs[np.isfinite(xs)]
        if xs.size == 0:
            print("[plotQueues] No finite values found: skipping.")
            return

        ys = np.linspace(0, 1, xs.size, endpoint=True)

        # Calculate statistics
        mean_queue_fraction = np.mean(xs)
        median_queue_fraction = np.median(xs)
        p90_queue_fraction = np.percentile(xs, 90)

        plt.figure(figsize=(10, 6))
        plt.step(xs, ys, where='post', linewidth=2.5, alpha=0.9, label="CDF", color='blue')

        # Add vertical lines for key statistics
        plt.axvline(mean_queue_fraction, color='red', linestyle='--', alpha=0.7,
                    label=f'Mean: {mean_queue_fraction:.3f}')
        plt.axvline(median_queue_fraction, color='green', linestyle='--', alpha=0.7,
                    label=f'Median: {median_queue_fraction:.3f}')
        plt.axvline(p90_queue_fraction, color='orange', linestyle='--', alpha=0.7,
                    label=f'90th percentile: {p90_queue_fraction:.3f}')

        plt.xlabel("Queue Fraction of Total Latency", fontsize=12)
        plt.ylabel("Cumulative Distribution Function (CDF)", fontsize=12)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.grid(alpha=0.3)
        plt.legend(fontsize=10)

        plt.title(f"Queue Latency Distribution - {n_active} Active Nodes\n" +
                  f"({len(queue_fractions)} data blocks analyzed)", fontsize=14)

        # Add interpretation text
        if mean_queue_fraction < 0.1:
            interpretation = "Low queuing (network not congested)"
        elif mean_queue_fraction < 0.3:
            interpretation = "Moderate queuing"
        else:
            interpretation = "High queuing (network congested)"

        plt.text(0.98, 0.02, f"Network Status: {interpretation}",
                 transform=plt.gca().transAxes, ha='right', va='bottom',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))

        os.makedirs(outputPath, exist_ok=True)
        out = os.path.join(outputPath, f"queues_{n_active}.png")
        plt.savefig(out, bbox_inches='tight', dpi=140)
        plt.close()

        print(f"[plotQueues] Queue analysis saved: {out}")
        print(f"[plotQueues] Mean queue fraction: {mean_queue_fraction:.3f} ({interpretation})")

    except Exception as e:
        print(f"[plotQueues] Error: {e}")


def extract_block_index(block_id):
    return int(block_id.split('_')[-1])


def save_plot_rewards(outputPath, reward, GTnumber, window_size=200):
    rewards = [x[0] for x in reward]
    times = [x[1] for x in reward]
    data = pd.DataFrame({'Rewards': rewards, 'Time': times})

    # Smoothed Rewards
    data['Smoothed Rewards'] = data['Rewards'].rolling(window=window_size).mean()

    # Top 10% and Bottom 10% Rewards
    data['Top 10% Avg Rewards'] = data['Rewards'].rolling(window=window_size).apply(
        lambda x: np.mean(np.partition(x, -int(len(x) * 0.1))[-int(len(x) * 0.1):]), raw=True)
    data['Bottom 10% Avg Rewards'] = data['Rewards'].rolling(window=window_size).apply(
        lambda x: np.mean(np.partition(x, int(len(x) * 0.1))[:int(len(x) * 0.1)]), raw=True)

    # Plotting
    plt.figure(figsize=(8, 4))
    line1, = plt.plot(data['Time'], data['Top 10% Avg Rewards'], color='skyblue', linewidth=2, label='Top 10% reward')
    line2, = plt.plot(data['Time'], data['Smoothed Rewards'], color='blue', linewidth=2, label='Average reward')
    line3, = plt.plot(data['Time'], data['Bottom 10% Avg Rewards'], color='navy', linewidth=2,
                      label='Bottom 10% reward')

    plt.legend(handles=[line1, line2, line3], fontsize=15, loc='upper right')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel("Time [ms]", fontsize=15)
    plt.ylabel("Average rewards", fontsize=15)
    plt.grid(True)
    # plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.15)

    # Save plot
    rewards_dir = os.path.join(outputPath, 'Rewards')
    plt.tight_layout()
    os.makedirs(rewards_dir, exist_ok=True)  # create output path
    plt.savefig(os.path.join(rewards_dir, "rewards_{}_gateways.png".format(GTnumber)))  # , bbox_inches='tight')
    plt.close()

    # Save CSV
    csv_dir = os.path.join(outputPath, 'csv')
    os.makedirs(csv_dir, exist_ok=True)  # create output path
    data.to_csv(os.path.join(csv_dir, "rewards_{}_gateways.csv".format(GTnumber)), index=False)

    return data


def save_epsilons(outputPath, eps, GTnumber):
    """
    Save epsilon decay plot and data to CSV.
    """
    if not eps or len(eps) == 0:
        print("Warning: No epsilon data to plot")
        return None

    epsilons = [x[0] for x in eps]
    times = [x[1] for x in eps]

    # Create epsilon decay plot
    plt.figure(figsize=(10, 6))
    plt.plot(times, epsilons, 'b-', linewidth=2, label='Epsilon')
    plt.title("Epsilon over Time", fontsize=16)
    plt.xlabel("Time (s)", fontsize=14)
    plt.ylabel("Epsilon", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)

    # Set reasonable axis ranges
    if len(times) > 0:
        plt.xlim(0, max(times) * 1.05)
    if len(epsilons) > 0:
        plt.ylim(0, max(epsilons) * 1.05)

    plt.tight_layout()

    # Create output directories
    os.makedirs(outputPath + '/epsilons/', exist_ok=True)
    os.makedirs(outputPath + '/csv/', exist_ok=True)

    # Save plot
    plt.savefig(outputPath + '/epsilons/' + "epsilon_{}_gateways.png".format(GTnumber),
                dpi=300, bbox_inches='tight')
    plt.close()

    # Save epsilon data to CSV
    data = {'epsilon': [e for e in epsilons], 'time': [t for t in times]}
    df = pd.DataFrame(data)
    df.to_csv(outputPath + '/csv/' + "epsilons_{}_gateways.csv".format(GTnumber), index=False)

    return df


def save_training_counts(outputPath, train_times, GTnumber):
    # Extract times
    times = [x[0] * 1000 for x in train_times]

    # Calculate cumulative trainings over time
    training_counts = list(range(1, len(times) + 1))

    # Plotting the cumulative number of trainings
    plt.plot(times, training_counts)
    plt.title("Cumulative trainings over time")
    plt.xlabel("Time (ms)")
    plt.ylabel("Number of Trainings")

    # Create output path and save the figure
    os.makedirs(outputPath + '/trainings/', exist_ok=True)
    plt.savefig(outputPath + '/trainings/' + "trainings_{}_gateways.png".format(GTnumber))
    plt.close()

    # Prepare data for saving
    data = {'time': times, 'trainings': training_counts}
    df = pd.DataFrame(data)

    # Create CSV output path and save data
    os.makedirs(outputPath + '/csv/', exist_ok=True)
    df.to_csv(outputPath + '/csv/' + "trainings_{}_gateways.csv".format(GTnumber), index=False)

    # return df


def save_losses(outputPath, earth1, GTnumber):
    losses = [x[0] for x in earth1.loss]
    times = [x[1] for x in earth1.loss]
    plt.plot(times, losses)
    plt.xlabel("Time (s)")
    plt.ylabel("Loss")
    plt.title("Loss over Time")
    os.makedirs(outputPath + '/loss/', exist_ok=True)  # create output path
    plt.savefig(outputPath + '/loss/' + "loss_{}_gatewaysTime.png".format(GTnumber))
    plt.close()

    data = {'loss': [l for l in losses], 'time': [t for t in times]}
    df = pd.DataFrame(data)
    df.to_csv(outputPath + '/csv/' + "loss_{}_gateways.csv".format(GTnumber), index=False)
    os.makedirs(outputPath + '/loss/', exist_ok=True)  # create output path

    xs = [l for l in range(len(losses))]
    plt.plot(xs, losses)
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Loss over Steps")
    plt.savefig(outputPath + '/loss/' + "loss_{}_gatewaysSteps.png".format(GTnumber))
    plt.close()

    # save losses average
    plt.plot(range(len(earth1.lossAv)), earth1.lossAv)
    plt.xlabel("Steps")
    plt.ylabel("Loss average")
    plt.title("Loss average over Steps")
    os.makedirs(outputPath + '/loss/', exist_ok=True)  # create output path
    plt.savefig(outputPath + '/loss/' + "loss_{}_gatewaysAverage.png".format(GTnumber))
    plt.close()


def plotSavePathLatencies(outputPath, n_active_nodes, blocks):
    """
    Generate latency plots for individual paths.
    """
    if not blocks:
        print("[plotSavePathLatencies] No blocks to plot: skipping.")
        return

    # Extract latency and arrival time data from blocks
    latencies = []
    arrival_times = []

    for block in blocks:
        if hasattr(block, 'creationTime'):
            # Ensure totLatency is calculated
            if not hasattr(block, 'totLatency') or block.totLatency == 0:
                if hasattr(block, 'getTotalTransmissionTime'):
                    block.totLatency = block.getTotalTransmissionTime()
                else:
                    continue

            # Only include blocks with valid latency > 0
            if block.totLatency > 0:
                latencies.append(block.totLatency * 1000)  # Convert to ms
                arrival_times.append((block.creationTime + block.totLatency) * 1000)  # Convert to ms

    if not latencies:
        print("[plotSavePathLatencies] No valid latency data found: skipping.")
        return

    print(f"[plotSavePathLatencies] Found {len(latencies)} blocks with valid latency data")
    print(f"[plotSavePathLatencies] Latency range: {min(latencies):.2f} - {max(latencies):.2f} ms")

    # Create output directory
    os.makedirs(outputPath + '/pngLatencies/', exist_ok=True)

    # Plot 1: Latency vs Time
    plt.figure(figsize=(12, 8))
    plt.scatter(arrival_times, latencies, c='red', alpha=0.6, s=20)
    plt.xlabel("Time [ms]", fontsize=14)
    plt.ylabel("Latency [ms]", fontsize=14)
    plt.title(f"Latency vs Time (Active Nodes: {n_active_nodes})", fontsize=16)
    plt.grid(True, alpha=0.3)

    # Set reasonable axis ranges
    if latencies:
        plt.ylim(0, max(latencies) * 1.1)
    if arrival_times:
        plt.xlim(0, max(arrival_times) * 1.05)

    plt.tight_layout()
    plt.savefig(outputPath + '/pngLatencies/' + f'{n_active_nodes}_nodesTime.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: Latency vs Arrival Index
    plt.figure(figsize=(12, 8))
    arrival_indices = list(range(len(latencies)))
    plt.scatter(arrival_indices, latencies, c='red', alpha=0.6, s=20)
    plt.xlabel("Arrival Index", fontsize=14)
    plt.ylabel("Latency [ms]", fontsize=14)
    plt.title(f"Latency vs Arrival Index (Active Nodes: {n_active_nodes})", fontsize=16)
    plt.grid(True, alpha=0.3)

    # Set reasonable axis ranges
    if latencies:
        plt.ylim(0, max(latencies) * 1.1)
    plt.xlim(0, len(latencies) * 1.05)

    plt.tight_layout()
    plt.savefig(outputPath + '/pngLatencies/' + f'{n_active_nodes}_nodes.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Save latencies to CSV
    os.makedirs(outputPath + '/csv/', exist_ok=True)
    data = {'Latency': latencies, 'Arrival Time': arrival_times}
    df = pd.DataFrame(data)
    df.to_csv(outputPath + '/csv/' + f"pathLatencies_{n_active_nodes}_nodes.csv", index=False)

    print(f"[plotSavePathLatencies] Generated latency plots for {len(latencies)} blocks")


def plot_packet_latencies_and_uplink_downlink_throughput(data, outputPath, bins_num=30, save=False,
                                                         plot_separately=True):
    """
    Generate either separate scatter plots of packet latencies for each path (source-destination),
    or a single plot combining all paths. Overlay line plots of uplink and downlink throughput on
    a secondary y-axis, with a single legend for all items in the upper right.
    """

    save_dir = os.path.join(outputPath, 'Throughput')
    os.makedirs(save_dir, exist_ok=True)

    # Helper: normalize hop label (int/str/obj/tuple) -> string
    def _label(x):
        if isinstance(x, (int, str)):
            return str(x)
        if isinstance(x, (list, tuple)):
            return "" if not x else _label(x[0])
        # Objects (e.g. TerrestrialNode/Sat) -> use name or ID if present
        return str(getattr(x, "name", getattr(x, "ID", x)))

    # Group blocks by (source, destination) paths
    from collections import defaultdict
    paths_data = defaultdict(list)
    all_blocks = []  # Store all blocks for aggregated plotting

    print(f"[DEBUG] Processing {len(data)} blocks for throughput plotting...")

    for block in data:
        # Add to all blocks list regardless of path validity
        all_blocks.append(block)

        # Skip blocks without valid path for individual path analysis
        if not getattr(block, "path", None) or len(block.path) < 2:
            continue
        src = _label(block.path[0])  # Source
        dst = _label(block.path[-1])  # Destination
        paths_data[(src, dst)].append(block)

    print(f"[DEBUG] Found {len(paths_data)} unique paths:")
    for (src, dst), blocks in paths_data.items():
        print(f"  - {src} → {dst}: {len(blocks)} blocks")

    # Function to plot data for a single path or combined
    def plot_path_data(blocks, src=None, dst=None):
        if not blocks:
            return

        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Sort blocks by creation time
        blocks = sorted(blocks, key=lambda b: b.creationTime)

        # Extract times and latencies (converted to ms)
        creation_times = np.array([b.creationTime for b in blocks]) * 1000.0  # ms

        # Calculate latencies using the same logic as plotSavePathLatencies
        latencies = []
        arrival_times = []
        for b in blocks:
            if hasattr(b, 'creationTime'):
                # Ensure totLatency is calculated
                if not hasattr(b, 'totLatency') or b.totLatency == 0:
                    if hasattr(b, 'getTotalTransmissionTime'):
                        b.totLatency = b.getTotalTransmissionTime()
                    else:
                        continue

                # Only include blocks with valid latency > 0
                if b.totLatency > 0:
                    latencies.append(b.totLatency * 1000)  # Convert to ms
                    arrival_times.append((b.creationTime + b.totLatency) * 1000)  # Convert to ms

        if not latencies:
            print(f"Warning: No valid latencies found for path {src} → {dst}")
            plt.close()
            return

        # Convert to numpy arrays
        latencies = np.array(latencies)
        arrival_times = np.array(arrival_times)
        creation_times = creation_times[:len(latencies)]  # Adjust to match valid latencies

        # Scatter plot for packet arrival times vs latency
        arrival_scatter = ax1.scatter(arrival_times, latencies, color='#1E90FF', label='Packet Delivery', alpha=0.6,
                                      s=15)

        # Configure primary y-axis for latency (FIXED: no negative values)
        ax1.set_xlabel('Time [ms]', fontsize=14)
        ax1.set_ylabel('Average E2E Latency [ms]', fontsize=14)

        # Set reasonable latency range (FIXED: no negative values)
        latency_min = max(0, np.min(latencies) * 0.9)
        latency_max = np.max(latencies) * 1.1
        ax1.set_ylim(latency_min, latency_max)

        # Create secondary y-axis for throughput
        ax2 = ax1.twinx()

        # Build time bins robustly (avoid null range/division by zero)
        t_min = float(np.min(creation_times))
        t_max = float(np.max(arrival_times))
        if not np.isfinite(t_min) or not np.isfinite(t_max):
            plt.close()
            return
        if t_max <= t_min:
            t_max = t_min + 1.0  # 1 ms artificial span
        bins = max(int(bins_num), 2)
        time_bins = np.linspace(t_min, t_max, num=bins)

        # Calculate throughput (FIXED: proper Mbps calculation)
        bin_widths = np.diff(time_bins)  # ms
        uplink_counts, _ = np.histogram(creation_times, bins=time_bins)
        downlink_counts, _ = np.histogram(arrival_times, bins=time_bins)

        # Mbps calculation: (bits per block) / (time in seconds) / 1e6
        # BLOCK_SIZE is in bits, bin_widths in ms, so convert to seconds
        denom = bin_widths.copy()
        denom[denom == 0] = 1.0  # protection
        uplink_throughput = (uplink_counts * BLOCK_SIZE) / (denom / 1000.0) / 1e6  # Mbps
        downlink_throughput = (downlink_counts * BLOCK_SIZE) / (denom / 1000.0) / 1e6  # Mbps

        # Plot throughput on secondary y-axis (FIXED: proper axis alignment)
        uplink_line, = ax2.plot(time_bins[:-1], uplink_throughput, color='#00008B', lw=2, label='Uplink Throughput')
        downlink_line, = ax2.plot(time_bins[:-1], downlink_throughput, color='#1E90FF', lw=2,
                                  label='Downlink Throughput')

        # Configure secondary y-axis for throughput
        ax2.set_ylabel('Throughput [Mbps]', fontsize=14)

        # Set reasonable throughput range
        max_throughput = max(np.max(uplink_throughput), np.max(downlink_throughput))
        if max_throughput > 0:
            ax2.set_ylim(0, max_throughput * 1.1)

        # Combine legends
        handles = [arrival_scatter, uplink_line, downlink_line]
        labels = [h.get_label() for h in handles]
        ax1.legend(handles, labels, loc='upper right', fontsize=12)

        # Title useful if separated
        if src is not None and dst is not None:
            ax1.set_title(f'Path {src} → {dst}', fontsize=16)

        # Display grid and layout adjustments
        ax1.grid(True, alpha=0.3)
        ax2.grid(True, linestyle=':', linewidth=0.5, alpha=0.3)
        plt.tight_layout()

        # Save or show plot
        if save:
            filename = f'{src}_{dst}_path_latency_throughput.png' if (
                    src is not None and dst is not None) else 'combined_path_latency_throughput.png'
            plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

    # Plot all paths together or separately based on flag
    if plot_separately:
        for (src, dst), blocks in paths_data.items():
            plot_path_data(blocks, src, dst)
    else:
        all_blocks = [b for blocks in paths_data.values() for b in blocks]
        plot_path_data(all_blocks)


def plot_throughput_cdf(data, outputPath, bins_num=100, save=False, plot_separately=True):
    """
    Generate and save a CDF plot of the throughput. Either plot each route separately or
    combine all routes into a single plot based on the `plot_separately` flag.
    """

    save_dir = os.path.join(outputPath, 'Throughput')
    os.makedirs(save_dir, exist_ok=True)

    # Helper: normalize hop label (int/str/obj/tuple) -> string
    def _label(x):
        if isinstance(x, (int, str)):
            return str(x)
        if isinstance(x, (list, tuple)):
            return "" if not x else _label(x[0])
        # Objects (e.g. TerrestrialNode/Sat) -> use name or ID if present
        return str(getattr(x, "name", getattr(x, "ID", x)))

    # Group blocks by (source, destination) paths
    from collections import defaultdict
    paths_data = defaultdict(list)
    for block in data:
        # Skip blocks without valid path
        if not getattr(block, "path", None) or len(block.path) < 2:
            continue
        src = _label(block.path[0])  # Source
        dst = _label(block.path[-1])  # Destination
        paths_data[(src, dst)].append(block)

    # Helper function to plot CDF for a given set of blocks
    def plot_cdf_for_path(blocks, src=None, dst=None):
        if not blocks:
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        # Sort blocks by creation time
        blocks = sorted(blocks, key=lambda b: b.creationTime)

        # Extract creation times and arrival times
        creation_times = np.array([block.creationTime for block in blocks])
        arrival_times = np.array([block.creationTime + block.totLatency for block in blocks])

        # Define time bins and calculate throughput (FIXED: proper time conversion)
        time_bins = np.linspace(min(creation_times), max(arrival_times), num=bins_num)
        bin_widths = np.diff(time_bins)  # seconds

        uplink_counts, _ = np.histogram(creation_times, bins=time_bins)
        downlink_counts, _ = np.histogram(arrival_times, bins=time_bins)

        # FIXED: Proper Mbps calculation
        uplink_throughput = (uplink_counts * BLOCK_SIZE) / (bin_widths * 1e6)  # Mbps
        downlink_throughput = (downlink_counts * BLOCK_SIZE) / (bin_widths * 1e6)  # Mbps

        # Filter out zero throughput values for better CDF visualization
        uplink_nonzero = uplink_throughput[uplink_throughput > 0]
        downlink_nonzero = downlink_throughput[downlink_throughput > 0]

        if len(uplink_nonzero) == 0 and len(downlink_nonzero) == 0:
            print(f"Warning: No non-zero throughput found for path {src} → {dst}")
            plt.close()
            return

        # Sort and calculate CDF
        if len(uplink_nonzero) > 0:
            uplink_throughput_sorted = np.sort(uplink_nonzero)
        uplink_cdf = np.arange(1, len(uplink_throughput_sorted) + 1) / len(uplink_throughput_sorted)
        ax.plot(uplink_throughput_sorted, uplink_cdf, label='Uplink Throughput', color='#00008B', lw=2)

        if len(downlink_nonzero) > 0:
            downlink_throughput_sorted = np.sort(downlink_nonzero)
            downlink_cdf = np.arange(1, len(downlink_throughput_sorted) + 1) / len(downlink_throughput_sorted)
        ax.plot(downlink_throughput_sorted, downlink_cdf, label='Downlink Throughput', color='#1E90FF', lw=2)

        # Configure plot
        ax.set_xlabel('Throughput [Mbps]', fontsize=14)
        ax.set_ylabel('CDF', fontsize=14)
        ax.legend(loc='lower right', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=12)

        # Set reasonable x-axis range
        all_throughput = np.concatenate([uplink_nonzero, downlink_nonzero])
        if len(all_throughput) > 0:
            ax.set_xlim(0, np.max(all_throughput) * 1.05)

        # Title useful if separated
        if src is not None and dst is not None:
            ax.set_title(f'Throughput CDF: {src} → {dst}', fontsize=16)

        # Adjust layout, save plot, and close
        plt.tight_layout()
        if save:
            filename = f'Throughput_CDF_{src}_to_{dst}.png' if src and dst else 'Throughput_CDF_All_Paths.png'
            plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

    # Plot each path separately or all paths combined based on flag
    if plot_separately:
        for (src, dst), blocks in paths_data.items():
            plot_cdf_for_path(blocks, src, dst)
    else:
        all_blocks = [block for blocks in paths_data.values() for block in blocks]
        plot_cdf_for_path(all_blocks)


def plotAllLatencies(outputPath, GTnumber, blocks):
    """
    Creates the 4-panel AllLatencies plot showing latency trends and individual points
    over both arrival time and creation time.
    """
    if not blocks:
        print("No blocks available for AllLatencies plot")
        return

    print(f"Creating AllLatencies plot with {len(blocks)} blocks...")

    # Debug: check first few blocks structure
    if blocks:
        print(f"DEBUG: First block type: {type(blocks[0])}")
        print(f"DEBUG: First block content: {blocks[0] if len(blocks) > 0 else 'Empty'}")
        if len(blocks) > 0 and isinstance(blocks[0], (list, tuple)):
            print(f"DEBUG: First block length: {len(blocks[0])}")
            print(f"DEBUG: First block elements: {blocks[0][:5] if len(blocks[0]) >= 5 else blocks[0]}")

    # Extract data from blocks
    creation_times = []
    arrival_times = []
    latencies = []

    for block in blocks:
        try:
            if isinstance(block, (list, tuple)) and len(block) >= 3:
                # block is a list: [latency, creation_time, arrival_time, ...]
                latency = float(block[0]) * 1000  # Convert to ms
                creation_time = float(block[1]) * 1000  # Convert to ms
                arrival_time = float(block[2]) * 1000  # Convert to ms

                creation_times.append(creation_time)
                arrival_times.append(arrival_time)
                latencies.append(latency)
            elif hasattr(block, 'creationTime') and hasattr(block, 'totLatency'):
                # block is a DataBlock object
                creation_time = float(block.creationTime) * 1000  # Convert to ms
                arrival_time = creation_time + float(block.totLatency) * 1000  # Convert to ms
                latency = float(block.totLatency) * 1000  # Convert to ms

                creation_times.append(creation_time)
                arrival_times.append(arrival_time)
                latencies.append(latency)
        except Exception as e:
            print(f"DEBUG: Error processing block: {e}")
            continue

    if not latencies:
        print("No valid latency data found")
        return

    # Create the 4-panel plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Convert to numpy arrays
    creation_times = np.array(creation_times)
    arrival_times = np.array(arrival_times)
    latencies = np.array(latencies)

    # Sort by time for proper plotting
    sort_idx = np.argsort(creation_times)
    creation_times = creation_times[sort_idx]
    arrival_times = arrival_times[sort_idx]
    latencies = latencies[sort_idx]

    # Panel 1: Latency Trends Over Arrival Time
    ax1.set_title('Latency Trends Over Arrival Time (Window Size = 20)')
    ax1.set_xlabel('Arrival Time (ms)')
    ax1.set_ylabel('Average Latency (ms)')
    ax1.grid(True, alpha=0.3)

    # Smooth the data with a moving average
    window_size = 20
    if len(arrival_times) > window_size:
        # Create time bins
        time_bins = np.linspace(arrival_times.min(), arrival_times.max(), 50)
        bin_centers = (time_bins[:-1] + time_bins[1:]) / 2

        # Calculate average latency for each bin
        bin_latencies = []
        for i in range(len(time_bins) - 1):
            mask = (arrival_times >= time_bins[i]) & (arrival_times < time_bins[i + 1])
            if np.any(mask):
                bin_latencies.append(np.mean(latencies[mask]))
            else:
                bin_latencies.append(np.nan)

        # Plot smoothed line
        valid_mask = ~np.isnan(bin_latencies)
        ax1.plot(bin_centers[valid_mask], np.array(bin_latencies)[valid_mask],
                 linewidth=2, alpha=0.8)

    # Panel 2: Individual Latency Points Over Arrival Time
    ax2.set_title('Individual Latency Points Over Arrival Time')
    ax2.set_xlabel('Arrival Time')
    ax2.set_ylabel('Latency')
    ax2.grid(True, alpha=0.3)
    ax2.scatter(arrival_times, latencies, alpha=0.6, s=10)

    # Panel 3: Latency Trends Over Creation Time
    ax3.set_title('Latency Trends Over Creation Time (Window Size = 20)')
    ax3.set_xlabel('Creation Time (ms)')
    ax3.set_ylabel('Average Latency (ms)')
    ax3.grid(True, alpha=0.3)

    if len(creation_times) > window_size:
        # Create time bins for creation time
        time_bins = np.linspace(creation_times.min(), creation_times.max(), 50)
        bin_centers = (time_bins[:-1] + time_bins[1:]) / 2

        # Calculate average latency for each bin
        bin_latencies = []
        for i in range(len(time_bins) - 1):
            mask = (creation_times >= time_bins[i]) & (creation_times < time_bins[i + 1])
            if np.any(mask):
                bin_latencies.append(np.mean(latencies[mask]))
            else:
                bin_latencies.append(np.nan)

        # Plot smoothed line
        valid_mask = ~np.isnan(bin_latencies)
        ax3.plot(bin_centers[valid_mask], np.array(bin_latencies)[valid_mask],
                 linewidth=2, alpha=0.8)

    # Panel 4: Individual Latency Points Over Creation Time
    ax4.set_title('Individual Latency Points Over Creation Time')
    ax4.set_xlabel('Creation Time')
    ax4.set_ylabel('Latency')
    ax4.grid(True, alpha=0.3)
    ax4.scatter(creation_times, latencies, alpha=0.6, s=10)

    plt.tight_layout()
    plt.savefig(os.path.join(outputPath, 'pngAllLatencies.png'), dpi=200, bbox_inches='tight')
    plt.close()

    print(f"AllLatencies plot saved as: {os.path.join(outputPath, 'pngAllLatencies.png')}")


def plotSaveAllLatencies(outputPath, GTnumber, allLatencies, epsDF=None):
    """
    allLatencies può essere:
      - dict: { tuple(path) -> [latency_float, ...] }
      - list/array: [latency_float, ...]
    Produce: istogramma e CDF delle latenze totali; salva anche CSV riassuntivo per path.
    """
    print(
        f"DEBUG: plotSaveAllLatencies called with GTnumber={GTnumber}, allLatencies type={type(allLatencies)}, length={len(allLatencies) if hasattr(allLatencies, '__len__') else 'N/A'}")
    if allLatencies and hasattr(allLatencies, '__len__') and len(allLatencies) > 0:
        print(f"DEBUG: First element type: {type(allLatencies[0])}")
        if hasattr(allLatencies[0], 'totLatency'):
            print(f"DEBUG: First block totLatency: {getattr(allLatencies[0], 'totLatency', 'None')}")
        else:
            print(f"DEBUG: First element attributes: {dir(allLatencies[0])[:10]}...")

    os.makedirs(outputPath, exist_ok=True)

    records = []  # Initialize records list

    if isinstance(allLatencies, dict):
        for p, vals in allLatencies.items():
            if vals is None:
                continue
            try:
                path_list = list(p)
            except Exception:
                path_list = [str(p)]
            clean = []
            for h in path_list:
                clean.append(h[0] if isinstance(h, (tuple, list)) and h else h)
            pstr = " -> ".join(str(x) for x in clean)
            hops = len(clean) - 1 if len(clean) >= 2 else len(clean)

            for v in vals:
                try:
                    lv = float(v)
                except Exception:
                    continue
                records.append({'Path': pstr, 'Latency': lv, 'Hops': hops})
    else:
        # assume list/array of latencies already "flat" OR list of DataBlock objects
        try:
            if allLatencies and hasattr(allLatencies[0], 'totLatency'):
                # allLatencies is a list of DataBlock objects
                for block in allLatencies:
                    try:
                        latency = float(getattr(block, 'totLatency', 0.0) or 0.0)
                        path = getattr(block, 'path', [])
                        if isinstance(path, (list, tuple)) and len(path) > 0:
                            path_str = " -> ".join(str(x) for x in path)
                            hops = len(path) - 1 if len(path) >= 2 else len(path)
                        else:
                            path_str = '(all)'
                            hops = np.nan
                        records.append({'Path': path_str, 'Latency': latency, 'Hops': hops})
                    except Exception:
                        continue
            elif allLatencies and isinstance(allLatencies[0], (list, tuple)):
                # allLatencies is a list of lists (data rows)
                print(f"DEBUG: Processing {len(allLatencies)} data rows...")
                for i, row in enumerate(allLatencies):
                    try:
                        if len(row) >= 2:  # Assuming [latency, path, ...] format
                            latency = float(row[0])
                            path = row[1] if len(row) > 1 else []
                            if isinstance(path, (list, tuple)) and len(path) > 0:
                                path_str = " -> ".join(str(x) for x in path)
                                hops = len(path) - 1 if len(path) >= 2 else len(path)
                            else:
                                path_str = '(all)'
                                hops = np.nan
                            records.append({'Path': path_str, 'Latency': latency, 'Hops': hops})
                        else:
                            # Single latency value
                            latency = float(row[0])
                            records.append({'Path': '(all)', 'Latency': latency, 'Hops': np.nan})
                    except Exception as e:
                        if i < 5:  # Only print first few errors
                            print(f"DEBUG: Error processing row {i}: {e}")
                        continue
            else:
                # allLatencies is a list of latency values
                for v in allLatencies:
                    records.append({'Path': '(all)', 'Latency': float(v), 'Hops': np.nan})
        except Exception:
            records = []

    if not records:
        print("Creating empty AllLatencies plot - no data available")
        # Create empty plot to show that no data was available
        plt.figure(figsize=(8, 5))
        plt.text(0.5, 0.5, 'No latency data available', ha='center', va='center', transform=plt.gca().transAxes,
                 fontsize=16)
        plt.xlabel('Total latency (s)')
        plt.ylabel('Count')
        plt.title(f'Latency histogram (GTs={GTnumber}) - No Data')
        plt.tight_layout()
        plt.savefig(os.path.join(outputPath, f'latency_hist_GT{GTnumber}.png'), dpi=160)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.text(0.5, 0.5, 'No latency data available', ha='center', va='center', transform=plt.gca().transAxes,
                 fontsize=16)
        plt.xlabel('Total latency (s)')
        plt.ylabel('CDF')
        plt.title(f'Latency CDF (GTs={GTnumber}) - No Data')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(outputPath, f'latency_cdf_GT{GTnumber}.png'), dpi=160)
        plt.close()
        return

    df = pd.DataFrame.from_records(records)

    stats = (df.groupby('Path')['Latency']
             .agg(['count', 'mean', 'median', 'min', 'max', 'std'])
             .reset_index())
    stats_path = os.path.join(outputPath, f'latency_stats_GT{GTnumber}.csv')
    stats.to_csv(stats_path, index=False)

    plt.figure(figsize=(8, 5))
    plt.hist(df['Latency'].values, bins=40, alpha=0.8, edgecolor='black')
    plt.xlabel('Total latency (s)')
    plt.ylabel('Count')
    plt.title(f'Latency histogram (GTs={GTnumber})')
    plt.tight_layout()
    plt.savefig(os.path.join(outputPath, f'latency_hist_GT{GTnumber}.png'), dpi=160)
    plt.close()

    vals = np.sort(df['Latency'].values)
    y = np.linspace(0, 1, len(vals), endpoint=True)
    plt.figure(figsize=(8, 5))
    plt.step(vals, y, where='post')
    plt.xlabel('Total latency (s)')
    plt.ylabel('CDF')
    plt.title(f'Latency CDF (GTs={GTnumber})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outputPath, f'latency_cdf_GT{GTnumber}.png'), dpi=160)
    plt.close()

    if epsDF is not None and len(getattr(epsDF, 'index', [])) > 0:
        try:
            plt.figure(figsize=(8, 4))
            plt.plot(epsDF.index, epsDF.values)
            plt.xlabel('Step')
            plt.ylabel('Epsilon')
            plt.title(f'Epsilon schedule (GTs={GTnumber})')
            plt.tight_layout()
            plt.savefig(os.path.join(outputPath, f'epsilon_GT{GTnumber}.png'), dpi=160)
            plt.close()
        except Exception:
            pass

    m = df['Latency'].mean()
    p50 = np.percentile(df['Latency'], 50)
    p95 = np.percentile(df['Latency'], 95)


def plotRatesFigures():
    values = [upGSLRates, downGSLRates, interRates, intraRate]

    plt.figure()
    plt.hist(np.asarray(interRates) / 1e9, cumulative=1, histtype='step', density=True)
    plt.title('CDF - Inter plane ISL data rates')
    plt.ylabel('Empirical CDF')
    plt.xlabel('Data rate [Gbps]')
    plt.show()
    plt.close()

    plt.figure()
    plt.hist(np.asarray(upGSLRates) / 1e9, cumulative=1, histtype='step', density=True)
    plt.title('CDF - Uplink data rates')
    plt.ylabel('Empirical CDF')
    plt.xlabel('Data rate [Gbps]')
    plt.show()
    plt.close()

    plt.figure()
    plt.hist(np.asarray(downGSLRates) / 1e9, cumulative=1, histtype='step', density=True)
    plt.title('CDF - Downlink data rates')
    plt.ylabel('Empirical CDF')
    plt.xlabel('Data rate [Gbps]')
    plt.show()
    plt.close()


def plotSatelliteCongestionMap(earth, blocks, outdir, GTnumber, active_names=None):
    """
    Crea una congestion map per la rete satellitare seguendo esattamente la logica del simulatore originale.
    """
    if not blocks:
        # Nessun blocco disponibile per la congestion map satellitare
        return

    # Filtra i blocchi con almeno 100 pacchetti (come nel simulatore originale)
    filtered_blocks = []
    packet_counts = []
    for block in blocks:
        packet_count = getattr(block, 'packets', 0)
        packet_counts.append(packet_count)
        if packet_count >= 100:
            filtered_blocks.append(block)

    print(f"Packet counts found: {packet_counts[:10]}... (max: {max(packet_counts) if packet_counts else 0})")

    if not filtered_blocks:
        # print("Nessun blocco con almeno 100 pacchetti trovato")
        # Try with lower threshold for debug
        # print("Trying with lower threshold (10 packets)...")
        for block in blocks:
            packet_count = getattr(block, 'packets', 0)
            if packet_count >= 10:
                filtered_blocks.append(block)
        if not filtered_blocks:
            # print("Nessun blocco con almeno 10 pacchetti trovato, usando tutti i blocchi")
            filtered_blocks = blocks
        else:
            print(f"Found {len(filtered_blocks)} blocks with >=10 packets")

    # Extract only satellite segments from hybrid paths
    satellite_paths = []
    for block in filtered_blocks:
        # Get the full path (either QPath or path)
        full_path = None
        if hasattr(block, 'QPath') and block.QPath:
            full_path = block.QPath
        elif hasattr(block, 'path') and block.path:
            full_path = block.path

        if not full_path:
            continue

        # Extract satellite segments (nodes that look like "X_Y" pattern)
        current_satellite_segment = []

        for hop in full_path:
            if isinstance(hop, (list, tuple)) and hop:
                hop = hop[0]

            # Check if hop is a satellite (format: "X_Y" where X and Y are numbers)
            hop_str = str(hop)
            if '_' in hop_str and len(hop_str.split('_')) == 2:
                parts = hop_str.split('_')
                if parts[0].isdigit() and parts[1].isdigit():
                    current_satellite_segment.append(hop)
                else:
                    # Hit a non-satellite node - save current segment if valid
                    if len(current_satellite_segment) >= 2:
                        satellite_paths.append(current_satellite_segment)
                    current_satellite_segment = []
            else:
                # Hit a non-satellite node - save current segment if valid
                if len(current_satellite_segment) >= 2:
                    satellite_paths.append(current_satellite_segment)
                current_satellite_segment = []

        # Add final satellite segment if it exists
        if len(current_satellite_segment) >= 2:
            satellite_paths.append(current_satellite_segment)

    print(f"Found {len(satellite_paths)} satellite paths from {len(filtered_blocks)} filtered blocks")

    if satellite_paths:
        output_file = os.path.join(outdir, 'Congestion_Test', f"satellite_congestion_{len(active_names)}nodes.png")
        # Saving satellite congestion map to: {output_file}

        # Usa la logica originale del simulatore: plotMap con i path satellitari
        done = earth.plotMap(
            plotGT=True,
            plotSat=True,
            edges=False,
            save=True,
            paths=satellite_paths,  # Pass as list, not numpy array
            fileName=output_file
        )

        if done == -1:
            # Congestion map satellitare non disponibile
            pass
        else:
            # Satellite congestion map saved in: {output_file}
            # Verify if file was created
            if os.path.exists(output_file):
                print(f"FILE CREATO: {output_file}")
            else:
                print(f"FILE NON TROVATO: {output_file}")
    else:
        print("Nessun path satellitare trovato nei blocchi filtrati")


def plotTerrestrialCongestionMap(earth, paths, outdir, GTnumber, active_names=None):
    """
    Generate a congestion map specific for the terrestrial network.
    paths: list of paths (each is a list of hops)
    active_names: iterable with active node names
    """
    os.makedirs(outdir, exist_ok=True)

    # Usa il numero di nodi attivi invece di GTnumber
    active_count = len(active_names) if active_names else GTnumber
    fname = os.path.join(outdir, f"terrestrial_congestion_{active_count}nodes.png")

    if not paths:
        print("Error: No data available for plotting terrestrial congestion map.")
        return

    # Extract only terrestrial segments from hybrid paths
    terrestrial_paths = []
    for p in paths:
        # Split path into terrestrial segments, breaking at satellite nodes
        current_terrestrial_segment = []

        for hop in p:
            if isinstance(hop, (list, tuple)) and hop:
                hop = hop[0]

            # Check if hop is in terrestrial graph
            if hasattr(earth, "terr_graph") and earth.terr_graph and str(hop) in earth.terr_graph.nodes:
                current_terrestrial_segment.append(hop)
            else:
                # Hit a satellite node - save current terrestrial segment if valid
                if len(current_terrestrial_segment) >= 2:
                    terrestrial_paths.append(current_terrestrial_segment)
                current_terrestrial_segment = []  # Reset for next terrestrial segment

        # Add final terrestrial segment if it exists
        if len(current_terrestrial_segment) >= 2:
            terrestrial_paths.append(current_terrestrial_segment)

    if not terrestrial_paths:
        print("Error: No terrestrial paths found.")
        return

    # Calculate terrestrial edge usage
    edge_use = Counter()
    for p in terrestrial_paths:
        for u, v in zip(p, p[1:]):
            key = tuple(sorted((str(u), str(v))))
            edge_use[key] += 1

    if not edge_use:
        print("Error: No terrestrial edges extracted from paths.")
        return

    counts = np.array(list(edge_use.values()), dtype=float)
    cmin, cmax = counts.min(), counts.max()
    if cmax == cmin:
        norm = {e: 100.0 for e in edge_use}
    else:
        norm = {e: (10.0 + 90.0 * ((cnt - cmin) / (cmax - cmin))) for e, cnt in edge_use.items()}

    # Crea la figura
    fig, ax = plt.subplots(figsize=(16, 10), dpi=200)
    ax.set_facecolor("white")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.axis('off')

    # Aggiungi mappa di sfondo se disponibile
    if hasattr(earth, "background_img"):
        ax.imshow(earth.background_img, extent=(-180, 180, -90, 90), alpha=0.3, zorder=0)

    # Improved colormap for terrestrial part
    cmap = plt.cm.viridis  # Lighter and more distinguishable colors

    # Disegna gli archi terrestri
    for (u, v), val in edge_use.items():
        def _get_terrestrial_coords(node):
            if not hasattr(earth, "terr_graph") or not earth.terr_graph:
                return None
            if str(node) not in earth.terr_graph.nodes:
                return None
            nd = earth.terr_graph.nodes[str(node)]
            if "lon" in nd and "lat" in nd:
                return float(nd["lon"]), float(nd["lat"])
            return None

        c1 = _get_terrestrial_coords(u)
        c2 = _get_terrestrial_coords(v)
        if c1 is None or c2 is None:
            continue

        lon1, lat1 = c1
        lon2, lat2 = c2

        # Create a more realistic curve for terrestrial links
        lons = np.linspace(lon1, lon2, 100)
        lats = np.linspace(lat1, lat2, 100)

        col = cmap(norm[(u, v)] / 100.0)
        linewidth = max(1.0, min(8.0, 2.0 + 6.0 * (norm[(u, v)] - 10) / 90))

        ax.plot(lons, lats, '-', lw=linewidth, color=col, alpha=0.8, zorder=2)

        # Aggiungi freccia per indicare la direzione
        ax.scatter([lon2], [lat2], marker='>', s=max(20, min(100, 20 + 80 * (norm[(u, v)] - 10) / 90)),
                   color=col, alpha=0.9, zorder=3)

    # Evidenzia i nodi attivi
    if active_names:
        pts = _active_nodes_xy(earth, active_names)
        if pts:
            ax.scatter([p[1] for p in pts], [p[2] for p in pts],
                       s=150, facecolors='red', edgecolors='darkred', linewidths=2, zorder=5)
            # Etichette per i nodi attivi
            for name, lon, lat in pts:
                ax.text(lon, lat, name, fontsize=10, color='darkred', weight='bold',
                        ha='center', va='bottom', zorder=6,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

    # Colorbar migliorata
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=10, vmax=100))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, shrink=0.8)
    cbar.set_label("Relative Traffic Load (%)", fontsize=12, weight='bold')
    cbar.ax.tick_params(labelsize=10)

    # Titolo
    ax.set_title(f"Terrestrial Network Congestion Map - {active_count} Active Nodes", fontsize=16, weight='bold',
                 pad=20)

    plt.tight_layout()
    plt.savefig(fname, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    # Terrestrial congestion map salvata in: {fname}

    # Verify if file was created
    if os.path.exists(fname):
        print(f"FILE TERRESTRE CREATO: {fname}")
    else:
        print(f"FILE TERRESTRE NON TROVATO: {fname}")


# @profile
def RunSimulation(GTs, inputPath, outputPath, populationData, radioKM):
    start_time = datetime.now()

    # --- Read inputRL.csv
    inputParams = pd.read_csv(inputPath + "inputRL.csv")
    locations = inputParams['Locations'].copy()
    print('Nº of Active Terrestrial Nodes: ' + str(len(locations) - 31))

    testType = inputParams['Test type'][0]
    testLength = inputParams['Test length'][0]

    print('Routing metric: ' + pathing)

    # Main pair (if present in CSV)
    main_src = None
    main_dst = None
    if 'Source' in inputParams.columns and 'Destination' in inputParams.columns:
        try:
            main_src = str(inputParams['Source'][0])
            main_dst = str(inputParams['Destination'][0])
        except Exception:
            main_src = main_dst = None

    simulationTimelimit = testLength if testType != "Rates" else movementTime * testLength + 10

    firstGT = True
    for GTnumber in GTs:
        global CurrentGTnumber, Train, TrainThis, nnpath
        if FL_Test:
            global CKA_Values
        if ddqn:
            global nnpathTarget
        TrainThis = Train
        CurrentGTnumber = GTnumber

        if firstGT:
            firstGT = False
        else:
            nnpath = f'{outputPath}/NNs/qNetwork_{GTnumber - 1}GTs.h5'
            if ddqn:
                nnpathTarget = f'{outputPath}/NNs/qTarget_{GTnumber - 1}GTs.h5'

        if len(GTs) > 1:
            start_time_GT = datetime.now()

        env = simpy.Environment()

        # Shuffle locations if needed
        if mixLocs:
            firstLocs = locations[:max(GTs)]
            random.shuffle(firstLocs)
            locations[:max(GTs)] = firstLocs

        inputParams['Locations'] = locations[:GTnumber]
        print('----------------------------------')
        print('Time:')
        print(datetime.now().strftime("%H:%M:%S"))
        print('Locations:')
        print(inputParams['Locations'][:GTnumber])
        print(f'Movement Time: {movementTime}')
        print(f'Rotation Factor: {ndeltas}')
        print(f'Minimum epsilon: {MIN_EPSILON}')
        print(f'Reward for deliver: {ArriveReward}')
        print(f'Stop Loss: {stopLoss}, number of samples considered: {nLosses}, threshold: {lThreshold}')
        print('----------------------------------')

        earth1, _, _, _ = initialize(
            env, populationData, inputPath + 'Gateways.csv', radioKM, inputParams,
            movementTime, locations, outputPath, matching=matching,
            TerrestrialNodesLocation=None  # Disabled: not using CSV for terrestrial nodes
        )
        earth1.outputPath = outputPath

        # If init defined the observed pair, use it as main pair
        if hasattr(earth1, 'observed_pair') and earth1.observed_pair and len(earth1.observed_pair) == 2:
            try:
                ms, md = earth1.observed_pair
                main_src = str(ms)
                main_dst = str(md)
                print(f"Source/Destination pair (from inputRL.csv): {main_src} → {main_dst}")
            except Exception:
                pass

        print('Saving ISLs map...')
        islpath = outputPath + '/ISL_maps/'
        os.makedirs(islpath, exist_ok=True)
        earth1.plotMap(plotGT=True, plotSat=True, edges=True, save=True, outputPath=islpath, n=earth1.nMovs)
        plt.close()

        print('Initial path saved!')
        print('----------------------------------')

        # Save initial path BEFORE simulation starts
        if hasattr(earth1, 'active_terrestrial_nodes') and len(earth1.active_terrestrial_nodes) >= 2:
            src_name = earth1.active_terrestrial_nodes[0].name
            dst_name = earth1.active_terrestrial_nodes[1].name

            # Calculate initial path with constellation in initial position
            p_terr = getShortestPathTerrestrial(src_name, dst_name, earth1.terr_graph)
            p_hyb = compute_hybrid_path(src_name, dst_name, earth1)

            terr_cost = estimate_path_cost(earth1, p_terr) if p_terr else float('inf')
            hyb_cost = estimate_path_cost(earth1, p_hyb) if p_hyb else float('inf')

            if p_hyb and hyb_cost < terr_cost:
                initial_path = p_hyb
                path_type = "hybrid"
                # print(f'[INITIAL] Using HYBRID path: {len(initial_path)} nodes (cost: {hyb_cost:.4f})')
            else:
                initial_path = p_terr
                path_type = "terrestrial"
                print(f'[INITIAL] Using TERRESTRIAL path: {len(initial_path)} nodes (cost: {terr_cost:.4f})')

            if initial_path:
                output_file = outputPath + f"path_{src_name}_to_{dst_name}_initial.png"
                plotPathClean(earth1, initial_path, src_name, dst_name, output_file)

        env.process(simProgress(simulationTimelimit, env))
        startTime = time.time()
        env.run(simulationTimelimit)
        timeToSim = time.time() - startTime

        # =========================
        # Base metrics from received DataBlocks
        # =========================
        # created / received / in flight (stuck)
        try:
            created_count = len(createdBlocks)
        except Exception:
            created_count = 0
        recv_blocks = list(receivedDataBlocks)
        received_count = len(recv_blocks)
        stuck_count = max(0, created_count - received_count)

        print(f"DEBUG: Created blocks: {created_count}")
        print(f"DEBUG: Received blocks: {received_count}")
        print(f"DEBUG: Stuck blocks: {stuck_count}")
        print(
            f"DEBUG: AllLatenciesRows length: {len(allLatenciesRows) if 'allLatenciesRows' in locals() else 'Not defined'}")

        # Helper function to safely get queue latency
        def _get_queue_latency(block):
            """Safely extract queue latency from a DataBlock"""
            try:
                qt = block.getQueueTime()
                if isinstance(qt, (list, tuple)) and len(qt) > 0 and qt[0] is not None:
                    return float(qt[0])
                elif block.timeAtFirstTransmission is not None and block.creationTime is not None:
                    return max(0.0, float(block.timeAtFirstTransmission - block.creationTime))
            except Exception:
                if block.timeAtFirstTransmission is not None and block.creationTime is not None:
                    return max(0.0, float(block.timeAtFirstTransmission - block.creationTime))
            return 0.0

        # Calculate queue latencies
        queue_latencies = [_get_queue_latency(b) for b in recv_blocks]

        # Calculate mean latencies for received blocks
        if received_count > 0:
            mean_q = float(np.mean(queue_latencies)) if queue_latencies else 0.0

            # Helper function to flatten nested values
            def _flatten_values(values):
                """Flattens a list that may contain float and nested lists"""
                flattened = []
                for v in values:
                    if isinstance(v, (list, tuple)):
                        flattened.extend(_flatten_values(v))
                    else:
                        try:
                            flattened.append(float(v))
                        except (ValueError, TypeError):
                            flattened.append(0.0)
                return flattened

            # Helper function to safely extract latency values
            def _extract_latency_values(blocks, attr_name):
                """Extract and flatten latency values from blocks"""
                values = []
                for b in blocks:
                    val = getattr(b, attr_name, 0.0)
                    if isinstance(val, (list, tuple)):
                        values.extend(_flatten_values(val))
                else:
                    try:
                        values.append(float(val))
                    except (ValueError, TypeError):
                        values.append(0.0)
                return values

            # Calculate mean latencies
            tx_values = _extract_latency_values(recv_blocks, 'txLatency')
            prop_values = _extract_latency_values(recv_blocks, 'propLatency')
            mean_tx = float(np.mean(tx_values)) if tx_values else 0.0
            mean_prop = float(np.mean(prop_values)) if prop_values else 0.0
            total = mean_q + mean_tx + mean_prop
            p_q, p_tx, p_prop = (0.0, 0.0, 0.0)
            if total > 0:
                p_q = 100.0 * mean_q / total
                p_tx = 100.0 * mean_tx / total
                p_prop = 100.0 * mean_prop / total
            print(f"Created DataBlocks:  {created_count}")
            print(f"Received DataBlocks: {received_count}")
            print(f"Stuck (in-flight):   {stuck_count}")
            print("---- Latency Breakdown (means) ----")
            print(f"Queue:        {mean_q:.6f} s ({p_q:.1f}%)")
            print(f"Transmission: {mean_tx:.6f} s ({p_tx:.1f}%)")
            print(f"Propagation:  {mean_prop:.6f} s ({p_prop:.1f}%)")
            print('-----------------------------------')

            # throughput/latency aggregata (sui ricevuti)
            sim_time = float(env.now)
            total_data_bits = sum(getattr(b, 'size', 0) for b in recv_blocks)
            throughput_mbps = (total_data_bits / sim_time) / 1e6 if sim_time > 0 else 0.0
            avg_total_latency = float(np.mean([
                (queue_latencies[i] if i < len(queue_latencies) else 0.0)
                + getattr(b, 'txLatency', 0.0)
                + getattr(b, 'propLatency', 0.0)
                for i, b in enumerate(recv_blocks)
            ])) if recv_blocks else 0.0

            print(f"Received blocks: {received_count}")
            print(f"Simulation time: {sim_time:.2f} s")
            print(f"Average Throughput: {throughput_mbps:.2f} Mbps")
            print(f"Average total latency: {avg_total_latency:.4f} s")
            print(f"Transmission: {mean_tx:.4f} s")
            print(f"Propagation: {mean_prop:.4f} s")

            # Plot dei risultati principali
            plt.figure(figsize=(12, 8))

            # Subplot 1: Throughput
            plt.subplot(2, 2, 1)
            plt.bar(['Average Throughput'], [throughput_mbps], color='blue', alpha=0.7)
            plt.ylabel('Mbps')
            plt.title('Average Throughput')
            plt.grid(True, alpha=0.3)

            # Subplot 2: Latency breakdown
            plt.subplot(2, 2, 2)
            latency_components = ['Queue', 'Transmission', 'Propagation']

            # Calculate latency values for plotting (reuse existing values)
            mean_queue = mean_q
            mean_tx_plot = mean_tx
            mean_prop_plot = mean_prop

            latency_values = [mean_queue, mean_tx_plot, mean_prop_plot]
            colors = ['red', 'orange', 'green']
            plt.bar(latency_components, latency_values, color=colors, alpha=0.7)
            plt.ylabel('Time (s)')
            plt.title('Latency Breakdown')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)

            # Subplot 3: Data blocks
            plt.subplot(2, 2, 3)
            blocks_data = ['Received', 'Stuck', 'Lost']
            lost_count = max(0, created_count - received_count - stuck_count)
            blocks_values = [received_count, stuck_count, lost_count]
            plt.pie(blocks_values, labels=blocks_data, autopct='%1.1f%%', startangle=90)
            plt.title('Data Blocks Status')

            # Subplot 4: Summary
            plt.subplot(2, 2, 4)
            plt.text(0.1, 0.8, f'Simulation Time: {sim_time:.2f} s', fontsize=12, transform=plt.gca().transAxes)
            total_blocks = received_count + stuck_count + lost_count
            plt.text(0.1, 0.6, f'Total Blocks: {total_blocks}', fontsize=12, transform=plt.gca().transAxes)
            success_rate = received_count / total_blocks * 100 if total_blocks > 0 else 0
            plt.text(0.1, 0.4, f'Success Rate: {success_rate:.1f}%', fontsize=12, transform=plt.gca().transAxes)
            plt.text(0.1, 0.2, f'Average Latency: {avg_total_latency:.4f} s', fontsize=12,
                     transform=plt.gca().transAxes)
            plt.title('Simulation Summary')
            plt.axis('off')

            plt.tight_layout()
            # plt.savefig(outputPath + 'simulation_results.png', dpi=300, bbox_inches='tight')  # Rimosso - non richiesto
            # plt.show()
            plt.close()
        else:
            print("No blocks received. Check links and path")

        # Blocks to work with for plotting
        blocks = recv_blocks

        # =========================
        # Post-processing / plotting
        # =========================
        if testType == "Rates":
            plotRatesFigures()
        else:
            # ---------- ROBUST DATASET CONSTRUCTION FOR plotSaveAllLatencies ----------
            allLatenciesRows = []
            rows_ok = 0
            rows_err = 0

            for b in blocks:
                try:
                    # Total queue time
                    q_total = 0.0
                    try:
                        qt = b.getQueueTime()  # [total, [per-hop]]
                        if isinstance(qt, (list, tuple)) and len(qt) > 0 and qt[0] is not None:
                            q_total = float(qt[0])
                    except Exception:
                        if getattr(b, 'timeAtFirstTransmission', None) is not None and getattr(b, 'creationTime',
                                                                                               None) is not None:
                            q_total = max(0.0, float(b.timeAtFirstTransmission - b.creationTime))

                    tx = float(getattr(b, 'txLatency', 0.0) or 0.0)
                    prop = float(getattr(b, 'propLatency', 0.0) or 0.0)
                    tot = q_total + tx + prop

                    creation_time = getattr(b, 'creationTime', None)
                    arrival_time = None
                    if getattr(b, 'checkPoints', None):
                        try:
                            arrival_time = float(b.checkPoints[-1])
                        except Exception:
                            arrival_time = None
                    if arrival_time is None:
                        base = creation_time if creation_time is not None else getattr(b, 'timeAtFirstTransmission',
                                                                                       None)
                        if base is None:
                            base = 0.0
                        arrival_time = float(base) + float(tot)

                    if creation_time is None:
                        try:
                            creation_time = float(arrival_time) - float(tot)
                        except Exception:
                            creation_time = 0.0

                    src = getattr(getattr(b, 'source', None), 'name', None) or str(
                        getattr(getattr(b, 'source', None), 'ID', ''))
                    dst = getattr(getattr(b, 'destination', None), 'name', None) or str(
                        getattr(getattr(b, 'destination', None), 'ID', ''))

                    p_sig = getattr(b, 'QPath', None) or getattr(b, 'path', None) or []

                    allLatenciesRows.append([
                        float(creation_time),  # Creation Time
                        float(tot),  # Latency
                        float(arrival_time),  # Arrival Time
                        src,  # Source
                        dst,  # Destination
                        bool(getattr(b, 'isNewPath', False)),
                        getattr(b, 'oldPath', []),
                        getattr(b, 'newPath', []),
                        p_sig,
                        float(q_total)  # queueTime
                    ])
                    rows_ok += 1
                except Exception:
                    rows_err += 1
                    continue

            queue_fractions = []
            for b in recv_blocks:
                try:
                    if b.totLatency > 0:
                        qt = b.getQueueTime()
                        if isinstance(qt, (list, tuple)) and len(qt) > 0 and qt[0] is not None:
                            queue_time = float(qt[0])
                        else:
                            if b.timeAtFirstTransmission is not None and b.creationTime is not None:
                                queue_time = max(0.0, float(b.timeAtFirstTransmission - b.creationTime))
                            else:
                                queue_time = 0.0

                        queue_fraction = queue_time / b.totLatency
                        queue_fractions.append(min(1.0, max(0.0, queue_fraction)))  # Clamp to [0,1]
                except Exception:
                    continue

            earth1.queues = queue_fractions

            print(f'DataBlocks lost: {earth1.lostBlocks}')

            # Generate essential plots
            if blocks:
                # Generate throughput analysis plots
                plot_packet_latencies_and_uplink_downlink_throughput(
                    blocks, outputPath, bins_num=50, save=True, plot_separately=plotAllThro
                )

                # Generate throughput CDF
                plot_throughput_cdf(blocks, outputPath, bins_num=100, save=True, plot_separately=plotAllThro)

                # Generate individual path latency plots
                n_active = len(getattr(earth1, 'active_terrestrial_nodes', []))
                plotSavePathLatencies(outputPath, n_active, blocks)

                print("Throughput and latency plots generated successfully!")
                print("=" * 50)
            else:
                print("Warning: No blocks data available for plotting")

            if pathing in ("Deep Q-Learning", "Q-Learning"):
                save_plot_rewards(outputPath, earth1.rewards, GTnumber)
                if not onlinePhase:
                    eps = earth1.DDQNA.epsilon if pathing == "Deep Q-Learning" else earth1.epsilon
                else:
                    eps = earth1.LEO[0].sats[0].DDQNA.epsilon if pathing == "Deep Q-Learning" else earth1.epsilon
                if Train:
                    epsDF = save_epsilons(outputPath, eps, GTnumber)
                    save_training_counts(outputPath, earth1.trains, GTnumber)
                else:
                    epsDF = None
                if allLatenciesRows:
                    plotSaveAllLatencies(outputPath, GTnumber, allLatenciesRows, epsDF)
                elif blocks:
                    print("Using received blocks for AllLatencies plot")
                    print(
                        f"DEBUG: blocks type: {type(blocks)}, length: {len(blocks) if hasattr(blocks, '__len__') else 'N/A'}")
                    plotSaveAllLatencies(outputPath, GTnumber, blocks, epsDF)
                    # Also create the 4-panel AllLatencies plot
                    plotAllLatencies(outputPath, GTnumber, blocks)
                else:
                    print("Warning: No latency data available for AllLatencies plot")
                    plotSaveAllLatencies(outputPath, GTnumber, [], epsDF)
            else:
                if allLatenciesRows:
                    plotSaveAllLatencies(outputPath, GTnumber, allLatenciesRows)
                elif blocks:
                    print("Using received blocks for AllLatencies plot")
                    plotSaveAllLatencies(outputPath, GTnumber, blocks)
                    # Also create the 4-panel AllLatencies plot
                    plotAllLatencies(outputPath, GTnumber, blocks)
                else:
                    print("Warning: No latency data available for AllLatencies plot")
                    plotSaveAllLatencies(outputPath, GTnumber, [])

            last_path = None
            if main_src and main_dst and blocks:
                from collections import Counter
                cnt = Counter()
                for b in blocks:
                    bs = getattr(getattr(b, 'source', None), 'name', None)
                    bd = getattr(getattr(b, 'destination', None), 'name', None)
                    if bs == main_src and bd == main_dst:
                        p = getattr(b, 'QPath', None) or getattr(b, 'path', None)
                        if p:
                            cnt[tuple(p)] += 1
                if cnt:
                    last_path = list(cnt.most_common(1)[0][0])

            if last_path is None and blocks:
                if hasattr(earth1, "active_terrestrial_nodes") and len(earth1.active_terrestrial_nodes) >= 2:
                    target_source = earth1.active_terrestrial_nodes[0].name
                    target_dest = earth1.active_terrestrial_nodes[1].name
                    main_src = target_source
                    main_dst = target_dest
                else:
                    target_source = "Unknown"
                    target_dest = "Unknown"
                    main_src = target_source
                    main_dst = target_dest
                lb = None

                for block in blocks:
                    source_name = getattr(getattr(block, 'source', None), 'name', 'Unknown')
                    dest_name = getattr(getattr(block, 'destination', None), 'name', 'Unknown')
                    if source_name == target_source and dest_name == target_dest:
                        lb = block
                        break

                if lb is None:
                    lb = blocks[0]
                    source_name = getattr(getattr(lb, 'source', None), 'name', 'Unknown')
                    dest_name = getattr(getattr(lb, 'destination', None), 'name', 'Unknown')
                    main_src = source_name
                    main_dst = dest_name
                else:
                    source_name = getattr(getattr(lb, 'source', None), 'name', 'Unknown')
                    dest_name = getattr(getattr(lb, 'destination', None), 'name', 'Unknown')
                    main_src = source_name
                    main_dst = dest_name

                last_path = getattr(lb, "path", None) or getattr(lb, "QPath", None)
                if last_path:
                    print(
                        f"[plotShortestPath] First few hops: {[hop[0] if isinstance(hop, (list, tuple)) else hop for hop in last_path[:5]]}")

            if last_path:
                title_txt = f"Path: {main_src} → {main_dst}" if (main_src and main_dst) else "Path"
                path_nodes = [hop[0] if isinstance(hop, (list, tuple)) else hop for hop in last_path]

                truncated_path = []
                destination_found = False
                for i, node in enumerate(path_nodes):
                    truncated_path.append(node)
                    if node == main_dst:
                        destination_found = True
                        break

                earth1.last_path = truncated_path
                earth1.last_src = main_src
                earth1.last_dst = main_dst

                # Determine if path is hybrid (contains satellites)
                satellite_ids = [str(sat.ID) for plane in earth1.LEO for sat in plane.sats]
                has_satellites = any(node in satellite_ids for node in truncated_path)

            else:
                print("[plotShortestPath] No path available: skipping.")

            # ---------- Queue Analysis ----------
            if not onlinePhase:
                # Calculate queue fractions for plotQueues using existing queue_latencies
                queue_fractions = []
                for i, b in enumerate(recv_blocks):
                    try:
                        # Calculate total latency if not already done
                        if not hasattr(b, 'totLatency') or b.totLatency == 0:
                            b.totLatency = b.getTotalTransmissionTime()

                        if b.totLatency > 0 and i < len(queue_latencies):
                            queue_fraction = queue_latencies[i] / b.totLatency
                            queue_fractions.append(min(1.0, max(0.0, queue_fraction)))  # Clamp to [0,1]
                    except Exception:
                        continue

                n_active = len(getattr(earth1, 'active_terrestrial_nodes', []))
                plotQueues(queue_fractions, outputPath, n_active)

            # ---------- Congestion Analysis ----------
            try:
                if blocks:
                    paths_for_congestion = []
                    for b in blocks:
                        p = getattr(b, 'QPath', None) or getattr(b, 'path', None)
                        if p and len(p) >= 2:
                            paths_for_congestion.append(p)
                    if paths_for_congestion:
                        print("Generating separate congestion maps...")

                        # Extract active node names
                        active_names = []
                        if hasattr(earth1, "active_terrestrial_nodes") and earth1.active_terrestrial_nodes:
                            active_names = [str(node.name) for node in earth1.active_terrestrial_nodes]
                        elif hasattr(earth1, "gateways") and earth1.gateways:
                            active_names = [str(gw.name) for gw in earth1.gateways]

                        # Check if the directory exists
                        congestion_dir = os.path.join(outputPath, 'Congestion_Test')
                        print(f"Congestion directory: {congestion_dir}")
                        print(f"Directory exists: {os.path.exists(congestion_dir)}")

                        # Create the directory (if it doesn't exist)
                        os.makedirs(congestion_dir, exist_ok=True)
                        print(f"Directory created/verified: {os.path.exists(congestion_dir)}")

                        # Terrestrial congestion map
                        print("Generating terrestrial congestion map...")
                        plotTerrestrialCongestionMap(
                            earth1, paths_for_congestion, os.path.join(outputPath, 'Congestion_Test'), GTnumber,
                            active_names=active_names
                        )

                        # Satellite congestion map
                        print("Generating satellite congestion map...")
                        plotSatelliteCongestionMap(
                            earth1, blocks, os.path.join(outputPath, 'Congestion_Test'), GTnumber,
                            active_names=active_names
                        )
                    else:
                        print("Congestion map: no paths available in blocks.")
                else:
                    print("[Congestion] No blocks: skipping figure.")
            except Exception as e:
                print(f"Congestion map for all routes not available ({e})")

        if pathing == 'Q-Learning':
            saveQTables(outputPath, earth1)
        elif pathing == 'Deep Q-Learning':
            saveDeepNetworks(outputPath + '/NNs/', earth1)

        # Cleanup
        receivedDataBlocks.clear()
        createdBlocks.clear()
        upGSLRates.clear()
        downGSLRates.clear()
        interRates.clear()
        intraRate.clear()
        gc.collect()
        del earth1
        del env

        if len(GTs) > 1:
            print('----------------------------------')
            print('Time:')
            end_time_GT = datetime.now()
            print(end_time_GT.strftime("%H:%M:%S"))
            print('----------------------------------')
            elapsed_time_GT = end_time_GT - start_time_GT
            print(f"Elapsed time for {GTnumber} GTs: {elapsed_time_GT}")
            print('----------------------------------')

    print('----------------------------------')
    print('Time:')
    end_time = datetime.now()
    print(end_time.strftime("%H:%M:%S"))
    print('----------------------------------')
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time}")
    print('----------------------------------')


###############################################################################
##############################     Main     ###################################
###############################################################################

if __name__ == '__main__':
    os.makedirs(outputPath, exist_ok=True)
    sys.stdout = Logger(outputPath + 'logfile.log', mode="w", encoding="utf-8")
    sys.stderr = sys.stdout

    RunSimulation(GTs, './', outputPath, populationMap, radioKM=rKM)