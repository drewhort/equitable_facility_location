import equitablefacilitylocation as efl
import pyomo.environ as pyo
import pandas as pd
import sys
import os



#define parameters and file paths    
    
FILE_PATH = '/home/drew/data/'
CITY_NUM = int(sys.argv[1])
NUM_TO_OPEN=int(sys.argv[2])
#which model to run options are KPL, PMED, PCENT, KPCON
MEASURE=str(sys.argv[3])
AVERSION_PARAMETER=-1


#choice of solver
SOLVER = "gurobi"
#SOLVER = pyo.SolverFactory('scip', executable="./scipoptsuite-8.0.4/build/bin/scip") 


#import  and clean the data based on user defined parameter    
ORIGINS_DF,DESTINATIONS_DF,DISTANCES_DF = efl.import_data_files(FILE_PATH,CITY_NUM)
TOTAL_POP,OPEN_CURRENT_LIST,ORIGINS_LIST,DESTINATIONS_LIST,ORIGINS_DF,DESTINATIONS_DF,DISTANCES_DF,ORIGINS_DICT,DISTANCES_DICT,OPEN_TOTAL = efl.clean_and_format_data(ORIGINS_DF,DESTINATIONS_DF,DISTANCES_DF,NUM_TO_OPEN)


#get initial kappa value based on the existing facilities
KAPPA = efl.compute_kappa(OPEN_CURRENT_LIST,ORIGINS_DF,DISTANCES_DF,AVERSION_PARAMETER)
#transform distances based on kappa for use in optimization model
DISTANCES_TRANSFORMED_DICT = efl.transform_distances(DISTANCES_DF,KAPPA)


MODEL = efl.create_pyomo_model(KAPPA,ORIGINS_LIST,DESTINATIONS_LIST,ORIGINS_DICT,DISTANCES_DICT,DISTANCES_TRANSFORMED_DICT,OPEN_CURRENT_LIST,OPEN_TOTAL,TOTAL_POP,MEASURE)
RESULTS,MODEL,WALL_TIME,TIME, TERMINATION_CONDITION= efl.solve_model(MODEL,SOLVER)
NEW_FACILITIES = efl.get_new_facilities(MODEL,DESTINATIONS_LIST)


if (TERMINATION_CONDITION == pyo.TerminationCondition.optimal):
    
    #get a list of optimal facilities including currently open
    NEW_FACILITIES = efl.get_new_facilities(MODEL,DESTINATIONS_LIST)

    #recompute kappa based on optimal solution from solve of MODEL
    KAPPA_OPT = efl.compute_kappa(NEW_FACILITIES, ORIGINS_DF, DISTANCES_DF, AVERSION_PARAMETER)
    #getting relevant values based on optimal solution
    KPL_VALUE,KP_VALUE, PMED_VALUE, PCENT_VALUE = efl.get_results(MODEL, KAPPA_OPT, TOTAL_POP, DISTANCES_DICT,ORIGINS_DICT,DESTINATIONS_LIST)
    
    #get objective value (number of additional facilities)
    NUM_NEW_FACILITIES=len(NEW_FACILITIES)-len(OPEN_CURRENT_LIST)
    
else: 
    NEW_FACILITIES = 'n/a'
    KAPPA_OPT = 'n/a'
    KPL_VALUE = 'n/a'
    KP_VALUE = 'n/a'
    PMED_VALUE = 'n/a'
    PCENT_VALUE = 'n/a'
    NUM_NEW_FACILITIES = 'n/a'
    
    
ADD = pd.DataFrame([[CITY_NUM,MEASURE,SOLVER,NUM_NEW_FACILITIES,NEW_FACILITIES,KAPPA,KAPPA_OPT,KP_VALUE,PMED_VALUE,PCENT_VALUE,TIME,WALL_TIME]], columns=['city','measure', 'solver', 'number of new facilities','new facilities','initial kappa','optimal kappa','kp value','average distance','maximum value','time','wall time'])
OUTPUT_PATH= str(MEASURE)+'.csv'
ADD.to_csv(OUTPUT_PATH, mode='a', header=not os.path.exists(OUTPUT_PATH))
