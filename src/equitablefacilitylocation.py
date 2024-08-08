import pyomo.environ as pyo
import sys
import numpy as np
import pandas as pd
import inequalipy as ineq
import logging

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


def import_data_files(file_path,city_num):
    """
    Load data for a given city.

    Parameters:
    file_path (str): The path to the directory containing the data files.
    city_num (int): The city number to load data for.

    Returns:
    tuple: Contains dataframes for origins(with populations), destinations, distances.
    """   
    logger.info('importing the data')
    
    origins_df = pd.read_csv(f'{file_path}{city_num}-population.csv')
    destinations_df = pd.read_csv(f'{file_path}{city_num}-destinations.csv')  
    distances_df = pd.read_csv(f'{file_path}{city_num}-distances.csv')
    
    return origins_df,destinations_df,distances_df
    
    
def clean_and_format_data(origins_df,destinations_df,distances_df,num_to_open=None):
    '''
    Clean data and format data appropriately.
    
    Parameters:
    origins_df (pandas.DataFrame): DataFrame containing origin information.
    destinations_df (pandas.DataFrame) DataFrame containing destination information.
    distances_df (pandas.DataFrame) DataFrame containing distance information.
    num_to_open (int, default=None): optional input for number of desired facilities to open when using PMED, PCENT, KPL measures.
    
    
    Returns:
    tuple: Contains cleaned dataframes and lists for populations, destinations, distances, open current list, total population, and total number to ope.
    
    '''
    
    logger.info('cleaning the data')
    
    #only get populations that are >0
    origins_df = origins_df[origins_df['U7B001'] > 0]

    #distances (including the bad ones)
    distances_df = distances_df.loc[distances_df['id_orig'].isin(origins_df['id_orig'].tolist())] 
    # make sure that the destinations are in our destinations list
    distances_df = distances_df.loc[distances_df['id_dest'].isin(destinations_df['id_dest'].tolist())] 
    #get only distances above >= 0
    distances_df = distances_df[distances_df['network'] >= 0]

    #origins (list)
    origins_list=distances_df['id_orig'].unique().tolist()
    #get destinations as list with only 
    destinations_list=distances_df['id_dest'].unique().tolist()
    
    #make sure we only have the populations of "good" origins
    origins_df = origins_df[origins_df["id_orig"].isin(origins_list)]
    
    #all currently open destinations (including bad ones)
    open_current_list = destinations_df.loc[destinations_df['dest_type']== 'supermarket', 'id_dest'].unique().tolist()
    
    #definining dictionaries    
    distances_dict = dict(zip(zip(distances_df['id_orig'], distances_df['id_dest']), distances_df['network']))
    origins_dict = dict(zip(origins_df['id_orig'],origins_df['U7B001']))
    
    total_pop=sum(origins_df['U7B001'])
    
    if num_to_open:
         open_total = num_to_open + len(open_current_list)
    else: open_total= 'n/a'
    
    return total_pop, open_current_list, origins_list, destinations_list, origins_df, destinations_df, distances_df, origins_dict, distances_dict, open_total
    


def compute_kappa(open_current_list,origins_df,distances_df,aversion_parameter):
    '''
    Use inequalipy to compute kappa value of given parameters
    
    Parameters: 
    open_current_list (list): list of currently open facilities
    distances_df (pandas.Dataframe): contains information about the network distance traveled from all origins to all destinations
    aversion_parameter (float): user defined aversion to inequality parameter. Larger in magnitude signifies larger penalty of inequality.
    
    Returns: 
    kappa (float): the appropriate value of kappa given the open facilities and populations of origins.
    
    '''

    logger.info('computing kappa')
    
    #get distances for all origins to all destinations that are already open
    distances_open_df = distances_df[distances_df["id_dest"].isin(open_current_list)]
    distances_open_df = distances_open_df.loc[distances_open_df.groupby('id_orig')['network'].idxmin()]
    distances_open_df = pd.merge(distances_open_df, origins_df, on='id_orig', how='left')
    # calculate kappa using inequalipy package
    kappa = ineq.kolmpollak.calc_kappa(distances_open_df['network'].values, epsilon = aversion_parameter, weights=distances_open_df['U7B001'].values)

    return kappa
    
    
def transform_distances(distances_df,kappa):
    '''
    Transform distances before optimization to lessen computational burden
    
    Parameters:
    distances_df (pandas.DataFrame): Contains distance information
    kappa (float): kappa value to be used in the optimization model
    
    Returns: distances_df with additional transformed distance column and distances_transformed_dict
    
    '''
    
    logger.info('transforming distances')
    
    #transforming distances
    distances_df['transformed'] = distances_df.network.apply(lambda d:np.exp(-kappa*d))
    #to get all the right pairs of origins and destinations in the dict
    distances_transformed_dict = dict(zip(zip(distances_df["id_orig"], distances_df["id_dest"]), distances_df["transformed"]))
    
    return distances_transformed_dict
    
    
def create_pyomo_model(kappa,origins_list,destinations_list,origins_dict,distances_dict,distances_transformed_dict,open_current_list,open_total,total_pop,measure,kp_threshold=None):
    '''
    Defines pyomo model, adds appropriate variables and constraints.
     
    Parameters: 
    kappa (float) : kappa value to be used in the optimization model
    origins_list (list) : list of origins (r)
    destinations_list (list) : list of destinations (s)
    origins_dict (dict) : dictionary of origins and populations for each origin
    distances_dict (dict) : contains information about the network distance traveled from all origins to all destinations
    distances_transformed_dict (dict) : transformed network distance traveled from all origins to all destinations
    open_current_list (list) : list of currently open facilities 
    open_total (int) : number of currently open facilities + number of desired facilities to add
    total_pop (int) : total population for all residential areas
    measure (str): Defines which measure to use in model, PCENT, PMED, KPL, or KPCON.
    kp_threshold (float,default=None) : Upper bound for Kolm-Pollak value to be used in KPCON model.

    
    Returns:
    model
    '''
    
    logger.info('building model in pyomo')
    
    
    #DEFINE DIFFERENT MODEL OBJECTIVE FUNCTIONS

    #KPL Objective: Minimize the population weighted kolm-pollak distance for residential areas (r) and services (s)
    def kpl_rule(model):
        return sum(origins_dict[r]*distances_transformed_dict[r,s]*model.y[r,s] for r, s in distances_transformed_dict)       
        
        
    #PMED Objective: Minimize the population weighted average distance for residential areas (r) and services (s)
    def pmed_rule(model):
        return sum(origins_dict[r]*distances_dict[r,s]*model.y[r,s] for r, s in distances_dict)
            

    #PCENT Objective: Minimize the maximum distance any resident area has to travel
    def pcent_rule(model):
        return model.u
            
    def minimax_rule(model,r,s):
        return model.y[r,s]*distances_dict[r,s] <= model.u
        
    #KPCON Objective: Minimize total number of facilities to open
    def kpcon_obj_rule(model):
        return sum(model.x[s] for s in destinations_list)
          

    #DEFINE MODEL CONSTRAINTS

    #residential area r is only assigned to service s if service s is open.
    def open_rule(model,r,s):
        return model.y[r,s]-model.x[s]<=0


    #each origin is only assigned to a single destination
    def single_rule(model,r):
        return sum(model.y[r,s] for s in destinations_list if (r,s) in distances_transformed_dict) == 1


    #sum of open destinations  should equal the number we want to open
    def open_total_rule(model):
        return sum(model.x[s] for s in destinations_list)==open_total


    #forces open service locations that are already there
    def currently_open_rule(model,s):
        return model.x[s]==1
        
    #bounding the population weighted kolm-pollak distance for residential areas (r) and services (s) from above by the kp_threshold
    def kpcon_rule(model):
        return sum(origins_dict[r]*distances_transformed_dict[r,s]*model.y[r,s] for r, s in distances_transformed_dict)-kp_threshold <= 0 
    
    #define model
    model= pyo.ConcreteModel()
    
    #Binary variable that is 1 if service s is open
    model.x=pyo.Var(destinations_list, within=pyo.Binary)
    # y[r,s] - 1 if  r that is served by service s, 0 otherwise
    model.y=pyo.Var(origins_list,destinations_list, within=pyo.Binary)
    
    if measure=='PCENT':
        model.u = pyo.Var(domain=pyo.NonNegativeReals)
        model.minimax_constraint = pyo.Constraint(distances_dict.keys(), rule=minimax_rule)
        model.pcent = pyo.Objective(rule=pcent_rule, sense=pyo.minimize)
    
    if measure=='PMED':
        model.pmed = pyo.Objective(rule=pmed_rule, sense=pyo.minimize)
    
    if measure=='KPL':
        model.kpl = pyo.Objective(rule=kpl_rule, sense=pyo.minimize)
    
    if measure=='KPCON':
        model.kpcon_obj=pyo.Objective(rule=kpcon_obj_rule,sense=pyo.minimize)
        model.kp_con = pyo.Constraint(rule=kpcon_rule)
    else: 
        model.open_total_constraint=pyo.Constraint(rule=open_total_rule)  
        
    model.open_constraint = pyo.Constraint(origins_list,destinations_list,rule=open_rule)
    model.single_constraint = pyo.Constraint(origins_list, rule=single_rule)
    model.currently_open_constraint = pyo.Constraint(open_current_list,rule=currently_open_rule)
    
    
    return model
    

def solve_model(model,solver):
    '''
    Solve the pyomo model
    
    Parameters:
    model : pyomo model
    solver (string) : desired solver to be used in optimization
    
    Returns:
    results
    model
    wall_time
    time 
    '''
    
    logger.info('solving the model')
    
    
    results = pyo.SolverFactory(str(solver)).solve(model, tee=True)
    results.write()
    wall_time=results.solver.wall_time
    time=results.solver.time
    termination_condition=results.solver.termination_condition
        
    return results,model,wall_time,time,termination_condition
    
def get_new_facilities(model,destinations_list):
    '''
    Get a list of the new facilities from the optimal solution of an instance
    
    Parameters:
    model : solved model
    destinations_list : list of destinations (s)
    
    Returns:
    new_facilities : list of existing facilities and additional facilities added in optimal solution
    '''

    new_facilities=[]
    for s in destinations_list:
        if pyo.value(model.x[s]) == 1:
            new_facilities.append(s)
    print(new_facilities)
            
    return new_facilities

def get_results(model, kappa, total_pop, distances_dict,origins_dict,destinations_list):
    '''
    Get objective values from solved model instance and compute approximate kolm pollak values
    
    Parameters: 
    model : solved model
    kappa : Kappa value computed based on optimal solution from model
    total_pop : total population for all residential areas
    distances_dict :  contains information about the network distance traveled from all origins to all destinations
    origins_dict : dictionary of origins and populations for each origin
    destinations_list : list of destinations (s)
    
    Returns:
    kpl_value : Value of the linearized Kolm-Pollak measure
    kp_value :  Kolm-Pollak EDE value
    pmed_value : Population weighted average distance 
    pcent_value : Maximum distance traveled by any residential area
    
    '''
    new_facilities=[]
    for s in destinations_list:
        if pyo.value(model.x[s]) == 1:
            new_facilities.append(s)
    print(new_facilities)
    
    kpl_value=sum(origins_dict[r]*np.exp(-kappa*distances_dict[r,s])*pyo.value(model.y[r,s]) for r, s in distances_dict)
    
    print('kpl value',kpl_value)
    
    kp_value=(-1/kappa)*np.log(1/total_pop*kpl_value)
    
    print('kp value',kp_value)
    
    pmed_value=(1/total_pop)*sum(origins_dict[r]*distances_dict[r,s]*pyo.value(model.y[r,s]) for r, s in distances_dict)
    
    print('pmed value',pmed_value)
    
    
    distances_list=[]
    for r in origins_dict.keys():
        dist=sum(pyo.value(model.y[r,s])*distances_dict[r,s] for s in destinations_list if (r, s) in distances_dict)
        distances_list.append(int(float(dist)))
    pcent_value=max(distances_list)
    
    print('pcent value', pcent_value)
    
        
    return kpl_value, kp_value, pmed_value, pcent_value
    
 
