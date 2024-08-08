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

#input city number 
city=int(sys.argv[1])
#define parameter epsilon
epsilon=-1
#define number of facilities to open
num_to_open=int(sys.argv[2])




def get_data(city,num_to_open,epsilon):
    """
    Retrieves, cleans, and formats data

    Parameters
    ----------
    city : integer
        value 0-499 corresponds to city number of the 500 largest U.S. cities 
    num_to_open : integer
        number of facilities user desires to add
    epsilon : float
        The inequality aversion parameter from the Atkinson formulae.
            If epsilon < 0 then the quantity is undesirable (less is better).

    Returns
    -------
     kappa : float
        The inequality aversion parameter from the Kolm-Pollak formulae.
            If kappa > 0 then the quantity is desirable (more is better).
            
      city_string :
      total_pop :
      origins_list :
      destinations_list :
      populations_dict :
      distances_not_transformed_dict :
      distances_transformed_dict :
      alpha :
      open_current :
      open_total :
      
    """

    logger.info('importing the data')
    
    #define file path
    file_path = './data/'
    
    #city number
    city_string=str(city)
    
    logger.info('importing CSV files')
    
    #import the appropriate CSV file
    populations = pd.read_csv(file_path + city_string + '-population.csv')
    destinations= pd.read_csv(file_path + city_string + '-destinations.csv')
    distances = pd.read_csv(file_path + city_string + '-distances.csv')
    

    logger.info('defining sets and parameters')

    #all currently open destinations (including bad ones)
    open_current = destinations.loc[destinations['dest_type']== 'supermarket', 'id_dest'].unique().tolist()

    # all destinations list (including bad ones)
    destinations = destinations.loc[destinations['dest_type'].isin(['supermarket','bg_centroid']), 'id_dest'].unique().tolist()

    logger.info('getting rid of zero population entries')
    #only get populations that are >0
    populations = populations[populations['U7B001'] > 0]
    #rename the colums to be origin and population
    populations.rename(columns={'id_orig': 'origin', 'U7B001':'population'}, inplace=True)
    #get only the colums of population and origin id
    populations = populations[['population', 'origin']]
    #make the index the origin id
    populations.set_index(['origin'], inplace=True)
    total_pop=sum(populations['population'])


    logger.info('importing the distances as dataframe')
    #distances (including the bad ones)
    distances = distances.loc[distances['id_orig'].isin(populations.index)]
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

    logger.info('defining sets as lists')
    #origins (list)
    origins_list = distances.index.get_level_values(0).unique().tolist()
    #get destinations as list with only 
    destinations_list = distances.index.get_level_values(1).unique().tolist()

    logger.info('getting rid of bad origins and destinations')
    #make sure we only have the populations of "good" origins
    populations = populations.loc[origins_list]
    #make sure the open current doesn't contain any "bad" destinations
    open_current = list(set(open_current).intersection(set(destinations_list)))
    #number of stores to open total
    open_total = num_to_open + len(open_current)


    logger.info('calculating kappa')
    #get distances for all origins to all destinations that are already open
    distances_open = distances.loc[pd.IndexSlice[:, open_current], :]
    populations_nearest = populations.loc[distances_open.groupby('origin')['distance'].min().index.get_level_values(0)]
    #get distance to the nearest open store for each origin
    nearest = distances_open.groupby('origin')['distance'].min().values
    # calculate kappa using inequalipy package
    kappa = ineq.kolmpollak.calc_kappa(nearest, epsilon = epsilon, weights=populations_nearest['population'].values)
    # get alpha
    alpha = -kappa
    logger.info('Computing current ede value for ' + city_string)


    logger.info('tranforming distances')
    #transforming distances
    distances['transformed'] = distances.distance.apply(lambda d:np.exp(alpha*d))
    #dan's way of converting to dataframe
    distances_transformed = distances.reset_index()[['origin', 'destination', 'transformed']].copy().rename(columns={'transformed':'distance'})
    #set multi index
    distances_transformed.set_index(['origin', 'destination'], inplace=True)
    #sort dataframe
    distances.sort_index(level=[0,1], inplace=True)

    logger.info('tranforming paramters into dictionaries')
    #to get all the right pairs of origins and destinations in the dict
    distances_transformed_df = distances_transformed.reset_index()
    distances_transformed_dict = dict(zip(zip(distances_transformed_df["origin"], distances_transformed_df.destination), distances_transformed_df.distance))

    distances_not_transformed_df = distances.reset_index()
    distances_not_transformed_dict = dict(zip(zip(distances_not_transformed_df["origin"], distances_not_transformed_df.destination), distances_not_transformed_df.distance))

    #populations_dict=populations.to_dict()
    populations_dict = {row['origin']:row['population'] for row in populations.reset_index().to_dict('records')}


    return (city_string, total_pop, origins_list, destinations_list, populations_dict, distances_not_transformed_dict, distances_transformed_dict, alpha, open_current, open_total)


city_string, total_pop, origins_list, destinations_list, populations_dict, distances_not_transformed_dict, distances_transformed_dict, alpha, open_current, open_total=get_data(city,num_to_open,epsilon)

#sets are python lists
#set of origins (origins_list)
#set of destinations (destinations_list)
#set of open current destinations (open_current)

#parameters are python dictionaries
#number of stores (k)
#population of residential area r (p_r)
#distance between residential area r and service s (d_r,s)
#kappa

#variables
#x_s indicator variable for whether location s has an open service (for all s in S)
#y_{r,s} indicator variable for whether residential area r is assigned to locaiton s

logger.info('building model in pyomo')

model = pyo.ConcreteModel()

#Binary variable that is 1 if service s is open
model.x=pyo.Var(destinations_list, within=pyo.Binary)
# y[r,s] - 1 if  r that is served by service s, 0 otherwise
model.y=pyo.Var(origins_list,destinations_list, within=pyo.Binary)


#Objective: Minimize the population weighted kolm-pollak distance for residential areas (r) and services (s)

model.u = pyo.Var(domain=pyo.NonNegativeReals)

def obj_rule(model):
    return model.u

model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)
        
        
def minimax_rule(model,r,s):
    return model.y[r,s]*distances_not_transformed_dict[r,s] <= model.u
    
model.minimax_constraint = pyo.Constraint(distances_not_transformed_dict.keys(), rule=minimax_rule)


#def pcent_rule(model):
#    return sum(populations_dict[r]*distances_not_transformed_dict[r,s]*model.y[r,s] for r, s in distances_not_transformed_dict)
  
#model.pcent = pyo.Objective(rule=pcent_rule, sense=pyo.minimize)

#Constraints

#residential area r is only assigned to service s if service s is open.
def open_rule(model,r,s):
    return model.y[r,s]-model.x[s]<=0

model.open_constraint = pyo.Constraint(origins_list,destinations_list,rule=open_rule)

#each origin is only assigned to a single destination
def single_rule(model,r):
    return sum(model.y[r,s] for s in destinations_list if (r,s) in distances_not_transformed_dict) == 1

model.single_constraint=pyo.Constraint(origins_list, rule=single_rule)

#sum of open destinations  should equal the number we want to open
def open_total_rule(model):
    return sum(model.x[s] for s in destinations_list)==open_total

model.open_total_constraint=pyo.Constraint(rule=open_total_rule)

#forces open service locations that are already there
def currently_open_rule(model,s):
    return model.x[s]==1

model.currently_open_constraint=pyo.Constraint(open_current,rule=currently_open_rule)

logger.info('solving model')
#result = pyo.SolverFactory('glpk').solve(model)
result = pyo.SolverFactory('gurobi').solve(model)
result.write()
wall_time=result.solver.wall_time
time=result.solver.time
logger.info('model solved')

# Print solution (if optimal)
if (result.solver.termination_condition == pyo.TerminationCondition.optimal):
    logger.info('optimal solution found')
    print(f'Objective = {pyo.value(model.obj)}')
    #av_dist = pyo.value(model.obj)
    #av_dist_meters=av_dist/total_pop
    av_dist=sum(pyo.value(model.y[r,s])*distances_not_transformed_dict[r,s]*populations_dict[r] for r, s in distances_not_transformed_dict)
    av_dist_meters=av_dist/total_pop


    results = pd.DataFrame()
    optimal_values = [pyo.value(model.x[s]) for s in destinations_list]
    optimal_values_df = pd.DataFrame(optimal_values)
    print(optimal_values_df)

    new_facilities=[]
    for s in destinations_list:
        if pyo.value(model.x[s]) == 1:
            new_facilities.append(s)
    print(new_facilities)


    distances_list=[]
    for r in origins_list:
        dist=sum(pyo.value(model.y[r,s])*distances_not_transformed_dict[r,s] for s in destinations_list if (r, s) in distances_not_transformed_dict)
        distances_list.append(int(float(dist)))
    max_dist=max(distances_list)
    

    kp_value=sum(pyo.value(model.y[r,s])*distances_transformed_dict[r,s]*populations_dict[r] for r, s in distances_not_transformed_dict)
    kp_value_meters=(1/alpha)*np.log(1/total_pop*kp_value)

    add = pd.DataFrame([[city,open_current,num_to_open,alpha,new_facilities,av_dist_meters,kp_value_meters,max_dist,distances_list,wall_time,time]], columns=['city','open_current','num_to_open','alpha','new_facilities','av_dist_meters','kp_value_meters','max_dist','distances_list','wall_time','time'])
    output_path= 'pcent'+str(num_to_open)+'.csv'
    add.to_csv(output_path, mode='a', header=not os.path.exists(output_path))


else:
    print(f'Solver termination condition: {result.solver.termination_condition}')

    add = pd.DataFrame([[city,len(open_current),result.solver.termination_condition,"n/a","n/a","n/a","n/a","n/a",wall_time,time]], columns=['city','open_current','num_to_open','new_facilities','av_dist_meters','kp_value_meters','max_dist','distances_list','wall_time','time'])

    output_path= 'pcent'+str(num_to_open)+'.csv'
    add.to_csv(output_path, mode='a', header=not os.path.exists(output_path))

