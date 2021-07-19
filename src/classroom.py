# imports
import numpy as np
import pandas as pd
import math
from param import parameterized
from scipy import stats
import sys
import json
import os
import matplotlib.pyplot as plt
import param
if 'config' in sys.path:
    print('config success')
else:
    sys.path.insert(0, 'config')

# import from infection.py
from infection import generate_infectivity_curves, plot_infectivity_curves, return_aerosol_transmission_rate


class BaseStudent(param.Parameterized):
    '''
    student agent base class

    :param student_id:              int     unique student ID number
    :param x:                       float32 x coordinate
    :param y:                       float32 y coordinate
    :param z:                       float32 z coordinate # unused as of 7/18
    :param initial:                 string  is this the initially infected student?
    :param infectious:              boolean have they begun emitting infectious particles?
    :param mask:                    boolean are they wearing a mask?
    :param is_infected:             boolean initially susceptible student that has been infected at a previous step
    :param indiv_breathing_rate:    float32 individual breathing rate
    '''
    student_id = param.Integer(default=0, doc='student id')
    x = param.Number(default=0.0, bounds=(-1.0, 1.0))
    y = param.Number(default=0.0, bounds=(-1.0, 1.0))
    initial = param.Boolean(default=False)
    infectious = param.Boolean(default=False)
    mask = param.Boolean(default=False)
    is_infected = param.Boolean(default=False)
    indiv_breathing_rate = param.Number(default=0.0)

class BaseClassroom(param.Parameterized):
    '''
    classroom agent base class

    # input
    :param n_students:          int     number of students in the classroom
    :param n_infected:          int     number of initially infected students
    :param n_adults:            int     number of adults in the classroom
    :param floor_area:          int     room floor area in meters
    :param room_height:         int     room height in meters
    :param outdoor_ach:         float32 ach level outside the room
    :param indoor_ach:
    :param merv_rating:
    :param recirc_rate:
    :param vent_location:
    :param vent_size:           surface area of vent
    :param door_locations:      (x,y,z) of door locations
    :param door_size:           surface area of door
    
    # calculated later
    :param direction:           matrix flow direction
    :param velocity:            matrix velocity
    :param mean_breathing_rate  float32 mean breathing rate

    ################# Make these into User Input
    w1 = (25, 0)
    w2 = (75, 0)
    door = (20, 96)
    vent = (50, 96) ############### make slider for this maybe ##########
    window_size = 8 # 40 centimeters diameter
    vent_size = 4 # 20 centimeters diameter

    temp_loc = {'w1': (25, 0),
    'w2': (75, 0),
    'door': (20, 96),
    'vent': (50, 96),
    'window_size': 8,
    'vent_size': 4}

    '''

    # TODO: setup defaults
    floor_area = param.Number(default=10.0, bounds=(0.0, None), doc='floor area')
    height = param.Number(default=1.0, bounds=(0.0, None), doc='height')
    outdoor_ach = param.Number(default=0.0, bounds=(0.0, None), doc='outdoor air exchange rate')
    indoor_ach = param.Number(default=2.0, bounds=(0.0, None), doc='baseline indoor ACH')
    merv_rating = param.Number(default=0.0, bounds=(0.0, None), doc='merv rating')
    recirc_rate = param.Number(default=0.0, bounds=(0.0, None), doc='recirculation rate')
    vent_locations = param.Dict(default={"None": (0.0, 0.0)}, doc='ventilation locations')
    vent_size = param.Number(default=0.0, bounds=(0.0, None), doc='ventilation size')
    window_locations = param.Dict(default={"None": (0.0, 0.0)}, doc='window locations')
    window_size = param.Number(default=0.0, bounds=(0.0, None), doc='window size')
    door_locations = param.Dict(default={"None": (0.0, 0.0)}, doc='door locations')
    door_size = param.Number(default=0.0, bounds=(0.0, None), doc='door size')

def load_parameters(filepath):
    '''
    Loads input and output directories
    '''
    try:
        with open(filepath) as fp:
            parameter = json.load(fp)
        print(filepath + ' load success')
    except FileNotFoundError:
        try:
            with open('../' + filepath) as fp:
                parameter = json.load(fp)
            print('../' + filepath + ' load success')
        except:
            with open('../../' + filepath) as fp:
                parameter = json.load(fp)
            print('../' + filepath + ' load success')

    return parameter

large_class = load_parameters('config/large_classroom.json')

small_class = load_parameters('config/small_classroom.json')
# print(os.getcwd(), '#############################################')
# print(os.listdir(), os.listdir('config'))

#
aerosol_params = load_parameters('results/aerosol_data_.json')
dp = load_parameters('results/default_data_.json')

initial_spread = np.array([[0, 0, 0, .03, 0, 0, 0],[0, 0, .04, .1, .03, 0, 0],[0, .03, .1, .2, .1, .04, 0],[.03, .1, .2, .3, .2, .1, .03],[0, .04, .1, .2, .1, .03, 0],[0, 0, .03, .1, .04, 0, 0],[0, 0, 0, .03, 0, 0, 0]])



def get_distance_class(student_pos, this_id, initial_id):
    x1, y1 = student_pos[initial_id]
    x2, y2 = student_pos[this_id]
    return x1, x2, y1, y2

def get_incoming(x, y, old, class_flow_direction, class_flow_velocity):
    select_dict = load_parameters('config/neighbor_logic.json')
    neighb = []
    count = 0
    x_max = old.shape[0] - 1
    y_max = old.shape[1] - 1

    if x == 0:
        x_range = [0, 1]
    elif x == x_max:
        x_range = [-1, 0]
    else:
        x_range = [-1, 0, 1]

    if y == 0:
        y_range = [0, 1]
    elif y == y_max:
        y_range = [-1, 0]
    else:
        y_range = [-1, 0, 1]

    this_val = old[x][y]
    this_direction = class_flow_direction[x][y]
    this_velocity = class_flow_velocity[x][y]

    dict_iterate_count = 0
    for i in x_range:
        for j in y_range:
            dict_iterate_count += 1
            if i == 0 and j == 0:
                # disperse .1 to all 8 neighboring squares
                for x_ in x_range:
                    for y_ in y_range:
                        direction = class_flow_direction[x_][y_]
                        if direction in select_dict[str([i, j])]:
                            idx = select_dict[str([i, j])].index(direction)
                            vel_idx = class_flow_velocity[x + i][y + j]
                            magnitude = .1
                            value = old[x + i][y + j]
                            neighb.append(value * magnitude)

            else:
                # factors in neighboring cell
                direction = int(class_flow_direction[x + i][y + j])
                # update direction function

                if direction in select_dict[str([i, j])]:
                    idx = select_dict[str([i, j])].index(direction)

                    vel_idx = int(class_flow_velocity[x + i][y + j])
                    # print(vel_idx, idx)
                    magnitude = select_dict["mag_array"][vel_idx]
                    risk = select_dict["risk_array"][idx]
                    value = old[x + i][y + j]
                    neighb.append(value * magnitude * risk)
                # else nothing moves in

    if len(neighb) > 0: # replaces self.value
        new_val = this_val * (1 - .5 * this_velocity) + np.mean(neighb)
    else:
        new_val = this_val
    return new_val

def air_effects(i, j, oldQ):
    '''
    i, j: x y locations

    oldQ: old quanta at that cube

    get neighbors directions and magnitude
    determine % in and % out
    '''
    # This is depracated as of 5/28
    return oldQ

def initial_conc_cough(init_x, init_y, initial_spread, old):
    '''
    Initially infected student distributing at each step to nearby

    TODO: Include code to catch students being next to walls

    TODO: Implement these as input variables
    Current Assumptions:
    Respiratory Activity = 2.04 quanta / ft^3
    Breathing Rate = .29 ft^3 / min


    '''
    new = old.copy()
    for i in range(len(initial_spread)):
        for j in range(len(initial_spread[0])):
            new[init_y + j - 3][init_x + i - 3] += initial_spread[i][j]

    return new

def normalize_(matrix):
    '''
    Make everything 0.01-1

    I dislike mpl vmin and vmax
    '''
    max_ = 0
    new = np.zeros(matrix.shape)
    for y in range(len(matrix)):
        for x in range(len(matrix[y])):
            if matrix[y][x] > max_:
                max_ = matrix[y][x]
    for y in range(len(matrix)):
        for x in range(len(matrix[y])):
            new[y][x] = matrix[y][x] / max_

    return new

def distribute(new, ach, initial, dir_matrix, vel_matrix, loc):
    '''
    0 1 2
    7 8 3
    6 5 4
    corners and edges: direction face inwards

    near open window: TODO


    '''
    out = np.zeros(new.shape)
    idx_ = str(initial)
    init_x, init_y = loc[idx_]
    dir_ref = {"-1,1": 0,
           "0,1": 1,
           "1,1": 2,
           "1,0": 3,
           "1,-1": 4,
           "0,-1": 5,
           "-1,-1": 6,
           "-1,0": 7,
           "0,0": 8}

    for y in range(len(new)):
        for x in range(len(new[0])):
            conc = new[y][x]

            # corners
            if ((y == 0) and (x == 0)): # bottom left
                iter_arr_x = [0, 1]
                iter_arr_y = [0, 1]
                d = 2
                v = vel_matrix[y][x]
            elif ((y==0) and (x==len(new[0]) - 1)): # bottom right
                iter_arr_x = [-1, 0]
                iter_arr_y = [0, 1]
                d = 0
                v = vel_matrix[y][x]
            elif ((y == len(new) - 1) and (x == len(new[0]) - 1)): # top right
                iter_arr_x = [-1, 0]
                iter_arr_y = [-1, 0]
                d = 6
                v = vel_matrix[y][x]
            elif ((y == len(new)-1) and (x == 0)): # top left
                iter_arr_x = [0, 1]
                iter_arr_y = [-1, 0]
                d = 4
                v = vel_matrix[y][x]
            # edges
            elif (y == 0): # bottom
                iter_arr_x = [-1, 0, 1]
                iter_arr_y = [0, 1]
                d = 1
                v = vel_matrix[y][x]
            elif (y == len(new) - 1): # top
                iter_arr_x = [-1, 0, 1]
                iter_arr_y = [-1, 0]
                d = 5
                v = vel_matrix[y][x]
            elif (x == 0): # left
                iter_arr_x = [0, 1]
                iter_arr_y = [-1, 0, 1]
                d = 3
                v = vel_matrix[y][x]
            elif (x == len(new[0]) - 1): # right
                iter_arr_x = [-1, 0]
                iter_arr_y = [-1, 0, 1]
                d = 7
                v = vel_matrix[y][x]
            # window


            # everywhere else
            else:
                iter_arr_x = [-1, 0, 1]
                iter_arr_y = [-1, 0, 1]
                d = dir_matrix[y][x]
                v = vel_matrix[y][x]
            min_ = 1
            airflow = ach * v / 60
            for i in iter_arr_x:
                for j in iter_arr_y:
                    idx = str(i) + ',' + str(j)
                    if d == dir_ref[idx]:
                        out[y + j][x + i] += airflow * conc * (1 - .02 * v)
                    elif (i ==0) and (j == 0):
                        out[y][x] += (1 - airflow) * conc * (1 - .02 * v)
                        if out[y][x] < 0:
                            out[y][x] = 0
                    else:
                        out[y + j][x + i] += .02 * v * conc
                    if out[y + j][x + i] < min_:
                        min_ = out[y + j][x + i]
    return out, min_

def concentration_distribution_(ach, direction_matrix, velocity_matrix, loc):
    first = np.zeros((100, 100))
    initial = 0
    temp = initial_conc_cough(25, 40, initial_spread, first)
    vent_, min_0 = distribute(temp, ach, initial, direction_matrix, velocity_matrix, loc)
    temp_arr_ = []
    vent_arr = []
    min_arr = []
    normed_arr = []
    for i in range(180): # sim duration
        temp = initial_conc_cough(25, 40, initial_spread, vent_)
        vent_, min_i = distribute(temp, ach, initial, direction_matrix, velocity_matrix, loc)
        normed = normalize_(vent_)
        temp_arr_.append(temp)
        vent_arr.append(vent_)
        min_arr.append(min_i)
        normed_arr.append(normed)
    return temp_arr_, vent_arr, min_arr, normed_arr

# out_matrix = np.array(np.zeros(shape=concentration_array[0].shape))
#     max_val = 0
#     for conc in range(len(concentration_array)):
#         for y in range(concentration_array[conc].shape[0]):
#             for x in range(concentration_array[conc].shape[1]):
#                 out_matrix[y][x] += (concentration_array[conc][y][x] / len(concentration_array))
#                 if  (concentration_array[conc][y][x] / len(concentration_array)) > max_val:
#                     max_val =  (concentration_array[conc][y][x] / len(concentration_array))

#     concentration_ = out_matrix
#     num_errors = 0
#     for conc in range(len(concentration_array)):
#         for y in range(concentration_array[conc].shape[0]):
#             for x in range(concentration_array[conc].shape[1]):
#                 if out_matrix[y][x] < 0:
#                     out_matrix[y][x] = 0
#                     num_errors += 1
#                 # out_matrix[y][x] = out_matrix[y][x] / max # is this necessary?

def make_new_heat(old, class_flow_pos, class_flow_direction, class_flow_velocity, init_infected_ = None):
    '''
    1 minute step used to calculate concentration_distribution iteratively
    '''
    kids =list(class_flow_pos.keys())
    if init_infected_ == None:
        init_infected_ = np.random.choice(kids)
    else:
        init_infected_ = init_infected_

    initial_loc = class_flow_pos[init_infected_]
    new = old.copy()
    out = old.copy()
    # spatial
    for i in range(len(old)):
        for j in range(len(old[i])):
            dist = math.sqrt(((initial_loc[0] - i)**2) + (initial_loc[1] - j)**2)
            new_val = old[i][j] + (1/(2.02 ** dist)) # chu emission
            new[i][j] = new_val

            ##################################################
    for i in range(len(new)):
        for j in range(len(new[i])):
            neighbor_val = get_incoming(i, j, new, class_flow_direction, class_flow_velocity)
            air_val = air_effects(i, j, neighbor_val)
            out[i][j] = air_val  #
    return out, init_infected_


def make_velocity_distance(c_instance, v_d_arguments):
    '''
    This is a proxy function for the airflow rate at higher values
    Allows for uneven distribution of viral particles in the room (i.e. Beyond Well-Mixed Room)

    Other attempts exist with more complex models, but this is simple enough to demonstrate the basic idea for now
    - 7/18 - BM

    Input:
    c_instance = BaseClass instance
    
    v_d_arguments = {
    :param floor_area:          Ar          Area of the room
    :param height:              H           Height of the room
    :param window_locations:    [(), ()]    List of tuples for the locations of windows
    :param window_size:         Aw          Surface area of the windows
    :param vent_locations:      [(), ()]    List of tuples for the locations of ventilators
    :param vent_size:           Av          Surface area of the ventilators
    :param door_locations:      [(), ()]    List of tuples for the locations of doors
    :param door_size:           Ad          Surface area of the doors
    }

    :method as follows:
    1. generate a room as a matrix of 0s
    2. at each timestep:
        2.1. for each initial infected student, emit and disperse viral quanta proportionally to the timestep length
        2.2. disperse the quanta in various directions according 

    :viral quanta emission rate:
    :viral infectiousness by mass:


    :return direction: 
    :return velocity:




    old params:
    window1, window2, door_location, vent_location, window_size, vent_size
    '''


    # function describes: y such that slope(x) = y
    temp = [[0 for col in range(100)] for row in range(100)] # size of room in 10 cm blocks
    x1 =  range(100)
    # up left
    line1_ =  [(i**2 / 100) for i in x1]
    ytemp = 100 - line1_[int(vent_location[0]- vent_size/2)]
    y1 = [i + ytemp for i in line1_]
    # These lines can be edited based on # of vents ##############################
    # left
    w1left = window1[0] - window_size/2
    w1right = window1[0] + window_size/2
    v1left = vent_location[0] - vent_size / 2
    v1right = vent_location[0] + vent_size / 2
    x2 = [w1left, v1left]
    y2 = [0, 100]
    m2 = 100/(v1left-w1left)
    b2 = 0 - m2 * w1left
    # down left
    x3 = [w1right, v1right]
    y3 = [0, 100]
    m3 = 100/(vent_location[0]-w1right)
    b3 = 0 - m3 * w1right
    # down
    w2left = window2[0] - window_size/2
    w2right = window2[0] + window_size/2
    x4 = [w2left, v1right]
    m4 = 100/(vent_location[0]-w2left)
    b4 = 0 - m4 * w2left
    # down right
    x5 = [w2right, v1right]
    m5 = 100/(v1right-w2right)
    b5 = 0 - m5 * w2right
    # right
    line6_ =  [((100 - i)**2 / 100) for i in x1]
    ytemp = 100 - line6_[int(v1right)]
    y6 = [i + ytemp for i in line6_]
    # between windows
    center = (window2[0] + window1[0]) / 2
    x7 = [i for i in range(100)]
    curve_down = [-(((i-center) / 5)**2) + center/4 for i in range(100)]




    #########################################

    # define direction plot
    for i in range(100):
        for j in range(100):
            # VENT MATH TO MAKE SURE THIS DOESN'T MESS UP ###################### TODO: BUG WITH NEGATIVE SLOPES: WTF DO I DO
            if j < curve_down[i]: # between windows
                if i < center - 5:
                    temp[j][i] = 0
                elif i > center + 5:
                    temp[j][i] = 2
                else:
                    temp[j][i] = 1
            elif (j > y1[i]): # top left
                temp[j][i] = 0
            elif (j > m2 * i + b2): # left edge
                temp[j][i] = 7
            elif (j > m3 * i + b3): # window 1
                temp[j][i] = 6
            elif (j < m4 * i + b4): # window 1 and 2
                temp[j][i] = 5
            elif (j < m5 * i + b5): # window 2
                temp[j][i] = 4
            elif (j < y6[i]): # window 2
                temp[j][i] = 3
            else:  # top right
                temp[j][i] = 2
            # edges

            # corners

            # barriers
    direction = temp.copy() # get from temp var
    #     r1 = mpl.patches.Rectangle((vent_location[0], 96), 10, 3, color='lightblue')

    ######################################### redo this for arbitrary width
    # velocity
    xtemp = range(100)
    w1 = window1
    w2 = window2
    i1 = [((i - vent_location[0]) / 3) **2 + 75 for i in xtemp]
    i2 = [((i - vent_location[0]) / 3) **2 + 50 for i in xtemp]
    i3 = [((i - vent_location[0]) / 3) **2 + 25 for i in xtemp]
    i4 = [((i - vent_location[0]) / 3) **2 for i in xtemp]

    i5 = [-(((i - w1[0])/3)**2) + 10 for i in xtemp]
    i6 = [-(((i - w1[0])/3)**2) + 15 for i in xtemp]
    i7 = [-(((i - w1[0])/3)**2) + 20 for i in xtemp]
    i8 = [-(((i - w1[0])/3)**2) + 25 for i in xtemp]

    i9 = [-(((i - w2[0])/3)**2) + 10 for i in xtemp]
    i10 = [-(((i - w2[0])/3)**2) + 15 for i in xtemp]
    i11 = [-(((i - w2[0])/3)**2) + 20 for i in xtemp]
    i12 = [-(((i - w2[0])/3)**2) + 25 for i in xtemp]
    blah = np.ones((100, 100))
    for i in range(100):
        for j in range(100):
            if j < i5[i]:
                blah[j][i] = 4
            elif j < i6[i]:
                blah[j][i] = 3.3
            elif j < i7[i]:
                blah[j][i] = 2.6
            elif j < i8[i]:
                blah[j][i] = 2
            elif j < i9[i]:
                blah[j][i] = 4
            elif j < i10[i]:
                blah[j][i] = 3.3
            elif j < i11[i]:
                blah[j][i] = 2.6
            elif j < i12[i]:
                blah[j][i] = 2

            elif j > i1[i]:
                blah[j][i] = 5
            elif j > i2[i]:
                blah[j][i] = 4
            elif j > i3[i]:
                blah[j][i] = 3
            elif j > i4[i]:
                blah[j][i] = 2
            else:
                blah[j][i] = 1
    velocity = blah.copy()

    return direction, velocity # and other useful variables

def classroom_simulation(sim_arguments, v_d_arguments, aerosol_arguments):
    '''
    Primary Simulation Function

    Inputs: 
    sim_arguments = {
    :param n_students:          number of students in each sim
    :param n_initial:           number of initial infected students  
    :param n_adults:            number of adults in each sim
    :param mask:                likelihood of student having mask
    :param n_sims:              number of simulations with new initial students 
    :param duration_mins_step:  time step in minutes
    :param duration_steps_day:  number of steps in a day
    :param duration_days_sim:   number of days to simulate      
    :param seating_chart:       (x,y) locations for student seating
    }
    v_d_arguments = {
    :param window_locations:    (x,y,z) of window locations
    :param window_size:         surface area of window
    :param vent_locations:      (x,y,z) of vent locations
    :param vent_size:           surface area of vent
    :param door_locations:      (x,y,z) of door locations
    :param door_size:           surface area of door
    }
    :method as follows:
    # loop through n simulations ->
    # d days in each simulation ->
    # s steps in each day ->
    # i infectious students -> 
    # u uninfected students ->
    # tranmission_rate_by_step_for_student_pair

    --> mean number of infections


    # output needed: 
    1. input params (user + default + calculated)
    2. 
    3. density estimation of exposure rate
    4. hist of mask wearing: y = num students, x = infection rate
    5. hist of ventilation: y = num students, x = infection rate

    6. scatter of transmission methods: infection rate vs distance
    7. scatter of transmission methods: infection rate vs time
    8. scatter of transmission methods: distance vs time with % thresholds for orders of magnitude



    :return input_params: dictionary of input parameters
    :return chance_nonzero:         chance that more than 0 students are infected
    :return concentration_array:    array of concentration heatmaps for every step
    :return t_avg_by_step:          array of average infection rate of all students by step
    :return t_avg_by_day:           array of average infection rate of all students by day
    :return t_avg_by_sim:           array of average infection rate of all students by sim

    Needed Asol Params
    aerosol_params['floor_area'], aerosol_params['mean_ceiling_height'],
    aerosol_params['air_exchange_rate'], aerosol_params['aerosol_filtration_eff'], 
    aerosol_params['relative_humidity'], aerosol_params['breathing_flow_rate'], 
    aerosol_params['exhaled_air_inf'], aerosol_params['max_viral_deact_rate'], 
    aerosol_params['mask_passage_prob']
    '''
    # initialize variables
    temp = generate_infectivity_curves()
    infectiousness_curves = plot_infectivity_curves(temp, plot=False)
    student_dict = {}
    student_loc = {}
    # init students
    for i in range(n_students):
        # generate BaseStudent instance with 'initial' set to False
        student_temp = BaseStudent(student_id=i, initial=False)
        student_loc[i] = {'x': student_temp.x, 'y': student_temp.y, 'z': student_temp.z}
        student_dict[i] = student_temp 

    ################## not working rn #########################3
    for initial_ in range(n_initial):
        np.random.choice(list(student_dict.keys()), p=list(student_dict.values()).count(True) / sim_arguments[n_students])

    class_instance = BaseClassroom() # store class_instance.distance and .velocity
    # update class_instance with user input

    # make the below into a parameter of BaseClassroom?
    direction, velocity = make_velocity_distance(class_instance, v_d_arguments)

    # make the below into a parameter of BaseClassroom?
    aerosol = return_aerosol_transmission_rate(aerosol_arguments) 

    # conc_dist vars = concentration_array, vent_arr, min_arr, normed_arr
    airflow_proxy = concentration_distribution_(aerosol_params['air_exchange_rate'], direction, velocity, student_dict)

    


    run_average_array = []
    # loop through n simulations ->
    for sim in range(class_arguments[n_sims]):
        if sim == int(n_sims / 4):
            print('25\% complete ...')
        elif sim == int(n_sims / 2):
            print('50\% complete ...')
        elif sim == 3 * int(n_sims / 4):
            print('75\% complete ...')
        elif sim == int(n_sims - 1):
            print('99\% complete ...')
        #########################



        # initialize student by random selection# initial
        initial_inf_id = np.random.choice(list(who_infected_class.keys()))
        init_inf_dict[initial_inf_id] += 1
        # initialize time until symptoms based on curve
        init_time_to_symp = int(np.round(stats.lognorm.rvs(l_shape, l_loc, l_scale, size=1)[0], 0))
        # fix overflow errors (unlikely but just in case)
        if init_time_to_symp >= 18:
            init_time_to_symp = 17
        if init_time_to_symp <= 0:
            init_time_to_symp = 0
        # initialize infectivity of student
        init_infectivity = inf_df.iloc[init_time_to_symp].gamma

        temp_average_array = temp_rates.copy()
        # print(temp_average_array)

        run_chance_of_0 = 1

        ######################

        # d days in each simulation ->
        for day in range(class_arguments[n_days]):
            # s steps in each day ->
            for step in range(class_arguments[n_steps]):
                # i infectious students -> 
                for initial_s in range(class_arguments[n_initial]):
                    # u uninfected students ->
                    # if initial is infectious: calculate pairwise transmission at this step


                    # number of infections 
                    pass
                
        print('100\% complete! Plotting output ...')

    # output:
    all_parameters = {
        'user_params': {
            'room': {

            },
            'human': {

            },
            'simulation': {

            },
            'classroom': {

            },
            'vent': {

            },
        },
        'default_params': {

        },
        'calculated_params': {
            'infectivity': 'temp'
        }
    }

    # # output needed: 
    # 1. input params (user + default + calculated)
    # 2. 
    # 3. density estimation of exposure rate
    # 4. hist of mask wearing: y = num students, x = infection rate
    # 5. hist of ventilation: y = num students, x = infection rate

    # 6. scatter of transmission methods: infection rate vs distance
    # 7. scatter of transmission methods: infection rate vs time
    # 8. scatter of transmission methods: distance vs time with % thresholds for orders of magnitude
