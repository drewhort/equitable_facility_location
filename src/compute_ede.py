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

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


city=int(sys.argv[1])
kappa= -0.0003973603173152847
alpha=-kappa

def get_data(city):

    file_path = './data/'
    #file_path = './'
   
    #city number
    city_string=str(city)

    #import the appropriate CSV file
    populations = pd.read_csv(file_path + city_string + '-population.csv')
    destinations= pd.read_csv(file_path + city_string + '-destinations.csv')
    distances = pd.read_csv(file_path + city_string + '-distances.csv')

    logger.info('defining sets and parameters')

    #all currently open destinations (including bad ones)
    open_current = destinations.loc[destinations['dest_type']== 'supermarket', 'id_dest'].unique().tolist()
    print('open_current',open_current)
    
    # all destinations list (including bad ones)
    destinations = destinations.loc[destinations['dest_type'].isin(['supermarket','bg_centroid']), 'id_dest'].unique().tolist()
    
    print('destinations',destinations)

    logger.info('getting rid of zero population entries')
    #only get populations that are >0
    populations = populations[populations['U7B001'] > 0]
    
    print('populations',populations)
    
    #rename the colums to be origin and population
    populations.rename(columns={'id_orig': 'origin', 'U7B001':'population'}, inplace=True)
    #get only the colums of population and origin id
    populations = populations[['population', 'origin']]
    #make the index the origin id
    populations.set_index(['origin'], inplace=True)
    total_pop=sum(populations['population'])
    print('total_pop',total_pop)

    logger.info('importing the distances as dataframe')
    #distances (including the bad ones)
    distances = distances.loc[distances['id_orig'].isin(populations.index)]
    print('distances',distances)
    # make sure that the destinations are in our destinations list
    distances = distances.loc[distances['id_dest'].isin(destinations)]
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
    print('distances',distances)

    logger.info('defining sets as lists')
    #origins (list)
    origins_list = distances.index.get_level_values(0).unique().tolist()
    #get destinations as list with only 
    destinations_list = distances.index.get_level_values(1).unique().tolist()

    #make sure we only have the populations of "good" origins
    populations = populations.loc[origins_list]
    #make sure the open current doesn't contain any "bad" destinations
    open_current = list(set(open_current).intersection(set(destinations_list)))
    


    logger.info('calculating ede')
    #get distances for all origins to all destinations that are already open
    distances_open = distances.loc[pd.IndexSlice[:, open_current], :]
    print('distances open',distances_open)

    populations_nearest = populations.loc[distances_open.groupby('origin')['distance'].min().index.get_level_values(0)]
    print('populations_nearest',populations_nearest)
    
    total_pop=sum(populations_nearest['population'])
    print('total pop',total_pop)

    #get distance to the nearest open store for each origin
    nearest = distances_open.groupby('origin')['distance'].min().values
    
    print('nearest',nearest)
    # calculate kappa using inequalipy package

    logger.info('Computing current ede value for ' + city_string)
    
    nearest_df = distances_open.groupby('origin')['distance'].min().to_frame()
    populations_nearest_df= populations.loc[nearest_df.index]
    print('nearest_df',nearest_df)
    print('populations_nearest_df',populations_nearest_df)

    #using inequalipy to compute ede value
    ede = ineq.kolmpollak.ede(nearest, kappa=-0.0003973603173152847, weights=populations_nearest['population'].values)
    print('values', len(populations_nearest['population'].values))

    #getting number of census blocks, used for experimenting
    num_blocks=len(nearest_df.index)
    print('num blocks', num_blocks)
    
    #expression used in optimization model
    kp_value=sum(populations_nearest_df['population'][r]*np.exp(alpha*nearest_df['distance'][r]) for r in nearest_df.index)
    
    #how we compute the ede score using the optimal value from the model, which is computed using expression above
    kp_value_meters=(1/alpha)*np.log((1/(total_pop))*kp_value)

    
    print('current ede calculated using inequalipy: ', ede, ' current ede calculated by sum: ',kp_value_meters)



    return (ede, total_pop, open_current,kp_value_meters,kp_value)

ede, total_pop, open_current,kp_value_meters,kp_value=get_data(city)



add = pd.DataFrame([[city,ede, total_pop, len(open_current),kp_value_meters,kp_value]], columns=['city', 'ede',' total_pop',' open_current','kp_value_meters','kp_value'])
output_path= 'ede_500cities.csv'
add.to_csv(output_path, mode='a', header=not os.path.exists(output_path))
