import sys
import pyomo.environ as pyo
import os
import matplotlib.pyplot as plt
import logging
import numpy as np
import math
import time
import random
import itertools
import pandas as pd
import inequalipy as ineq
import glob

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


def get_data():

    distances = pd.concat(map(pd.read_csv, glob.glob(os.path.join("./data/", "*-distances.csv"))), ignore_index= True)
    populations = pd.concat(map(pd.read_csv, glob.glob(os.path.join("./data/", "*-population.csv"))), ignore_index= True)
    destinations = pd.concat(map(pd.read_csv, glob.glob(os.path.join("./data/", "*-destinations.csv"))), ignore_index= True)
    
    print(distances)
    print(populations)
    print(destinations)

    #all currently open destinations (including bad ones)
    open_current = destinations.loc[destinations['dest_type']== 'supermarket', 'id_dest'].unique().tolist()

    print(open_current)
    # all destinations list (including bad ones)
    destinations = destinations.loc[destinations['dest_type'].isin(['supermarket','bg_centroid']), 'id_dest'].unique().tolist()
    print(destinations)

    logger.info('getting rid of zero population entries')
    #only get populations that are >0
    populations = populations[populations['U7B001'] > 0]
    print(populations)
    #rename the colums to be origin and population
    populations.rename(columns={'id_orig': 'origin', 'U7B001':'population'}, inplace=True)
    #get only the colums of population and origin id
    populations = populations[['population', 'origin']]
    #make the index the origin id
    populations.set_index(['origin'], inplace=True)
    print(populations)
    #total population of the city
    total_pop=sum(populations['population'])
    print(total_pop)


    logger.info('importing the distances as dataframe')
    #distances (including the bad ones)
    distances = distances.loc[distances['id_orig'].isin(populations.index)]
    print(distances)
    # make sure that the destinations are in our destinations list
    distances = distances.loc[distances['id_dest'].isin(destinations)]
    print(distances)
    #rename the columns
    distances.rename(columns={'id_orig': 'origin', 'id_dest': 'destination','network':'distance'}, inplace=True)
    #get only the columns we care about
    distances = distances[['origin', 'destination', 'distance']]
    #make multiindex dataframe
    distances.set_index(['origin', 'destination'], inplace=True)
    #sort dataframe
    distances.sort_index(level=[0,1], inplace=True)
    #get only distances above > 0
    distances = distances[distances['distance'] > 0]
    print(distances)

    logger.info('defining sets as lists')
    #origins (list)
    origins_list = distances.index.get_level_values(0).unique().tolist()
    #get destinations as list with only 
    destinations_list = distances.index.get_level_values(1).unique().tolist()
    print(destinations_list)

    #make sure we only have the populations of "good" origins
    populations = populations.loc[origins_list]
    print(populations)
    #make sure the open current doesn't contain any "bad" destinations
    open_current = list(set(open_current).intersection(destinations))
    print('total population',total_pop)
    print('number of open stores',len(open_current))
    print(open_current)
 


    logger.info('finding distances to stores open')
    #get distances for all origins to all destinations that are already open
    distances_open = distances.loc[pd.IndexSlice[:, open_current], :]
    print(distances_open)
    #populations_nearest = populations.loc[distances_open.groupby('origin')['distance'].min().index.get_level_values(0)]
    logger.info('finding nearest distances to stores open')
    #get distance to the nearest open store for each origin
    nearest = distances_open.groupby('origin')['distance'].min().values
    print(len(nearest))
    # calculate kappa using inequalipy package

    logger.info('finding populations for weights')
    nearest_df = distances_open.groupby('origin')['distance'].min().to_frame()
    print(nearest_df)
    populations_nearest_df= populations.loc[nearest_df.index]

    kappa = ineq.kolmpollak.calc_kappa(nearest, epsilon = -1, weights=populations_nearest_df["population"].values)
    
    print(kappa)


    return (kappa)

kappa=get_data()


print(kappa)
