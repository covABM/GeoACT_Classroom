# imports
import numpy as np
import pandas as pd
from param import *
from scipy import stats
import json
import matplotlib.pyplot as plt
import param
# import from infection.py
from src.infection import *

class BaseStudent(param.Parameterized):
    '''
    student agent base class

    :param id:              str     unique student ID number
    :param x:                       float32 x coordinate
    :param y:                       float32 y coordinate
    :param z:                       float32 z coordinate # unused as of 7/18
    :param initial:                 string  is this the initially infected student?
    :param infectious:              boolean have they begun emitting infectious particles?
    :param mask:                    boolean are they wearing a mask?
    :param is_infected:             boolean initially susceptible student that has been infected at a previous step
    :param indiv_breathing_rate:    float32 individual breathing rate
    :param time_to_symptoms:        float32 time tosymptoms
    :param infectivity:             float32 infectivity
    '''
    id = param.Integer(default=0, doc='student id')
    x = param.Number(default=50, bounds=(5, 95))
    y = param.Number(default=50, bounds=(5, 95))
    initial = param.Boolean(default=False)
    infectious = param.Boolean(default=False)
    mask = param.Boolean(default=False)
    is_infected = param.Boolean(default=False)
    indiv_breathing_rate = param.Number(default=0.0)
    time_to_symptoms = param.Number(default=0.0)
    infectivity = param.Number(default=0.0)

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
    '''

    # TODO: setup defaults
    floor_area = param.Number(default=1000, bounds=(1000, 4000), doc='floor area')
    height = param.Number(default=12, bounds=(12, 24), doc='height')
    outdoor_ach = param.Number(default=2, bounds=(2, 8), doc='outdoor air exchange rate')
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

def get_incoming(x, y, old, class_flow_direction, class_flow_velocity):
    select_dict = load_parameters('config/neighbor_logic.json')
    neighb = []
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

def initial_conc_cough(init_dict, old):
    '''
    Initially infected student distributing at each step to nearby grid squares

    TODO: Include code to catch students being next to walls

    :param init_x:
    :param init_y:
    :param old:

    :return:
    '''
    initial_spread = np.array([[0, 0, 0, .03, 0, 0, 0],[0, 0, .04, .1, .03, 0, 0],[0, .03, .1, .2, .1, .04, 0],[.03, .1, .2, .3, .2, .1, .03],[0, .04, .1, .2, .1, .03, 0],[0, 0, .03, .1, .04, 0, 0],[0, 0, 0, .03, 0, 0, 0]])


    new = old.copy()
    for init in init_dict.keys():
        init_x = init_dict[init][0]
        init_y = init_dict[init][1]
        for i in range(len(initial_spread)):
            for j in range(len(initial_spread[0])):
                new[init_y + j - 3][init_x + i - 3] += initial_spread[i][j]
    return new

def normalize_(matrix):
    '''
    :param matrix:

    :return: normalized matrix w/ <0 changed to 0
    '''
    max_ = 0
    new = np.zeros(matrix.shape)
    for y in range(len(matrix)):
        for x in range(len(matrix[y])):
            if matrix[y][x] > max_:
                max_ = matrix[y][x]
            elif matrix[y][x] < 0:
                matrix[y][x] = 0
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

    ** moves viral particles around in the proxy @ each step

    :param new:
    :param ach:
    :param initial:
    :param dir_matrix:
    :param vel_matrix:
    :param loc:
    :return:
    '''
    out = np.zeros(new.shape)
    # init_x, init_y = loc[initial]
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

def concentration_distribution_(ach, direction_matrix, velocity_matrix, loc, init_dict):
    '''
    TODO: allow for conc_cough to be have > 1 initial infected
    
    :param ach:
    :param direction_matrix:
    :param velocity_matrix:
    :param loc:
    :param init_x:
    :param init_y:

    :return temp_arr_:
    :return vent_arr_:
    :return min_arr:
    :return normed_arr:         Use this one
    '''


    first = np.zeros((100, 100))
    initial = 0
    
    temp = initial_conc_cough(init_dict, first)
    vent_, min_0 = distribute(temp, ach, initial, direction_matrix, velocity_matrix, loc)
    temp_arr_ = []
    vent_arr = []
    min_arr = []
    normed_arr = []
    for i in range(180): # sim duration
        temp = initial_conc_cough(init_dict, vent_)
        vent_, min_i = distribute(temp, ach, initial, direction_matrix, velocity_matrix, loc)
        normed = normalize_(vent_)
        temp_arr_.append(temp)
        vent_arr.append(vent_)
        min_arr.append(min_i)
        normed_arr.append(normed)
    return temp_arr_, vent_arr, min_arr, normed_arr

def make_velocity_distance(v_d_arguments):
    '''
    This is a proxy function for the airflow rate at higher values
    Allows for uneven distribution of viral particles in the room (i.e. Beyond Well-Mixed Room)

    Other attempts exist with more complex models, but this is simple enough to demonstrate the basic idea for now
    - 7/18 - BM
    Current iteration only works with small classroom room type
    - 7/20 - BM

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

    :return direction:          Distance matrix for concentration fluid dynamics sim
    :return velocity:           Velocity matrix for concentration fluid dynamics sim
    '''
    window_location = v_d_arguments['window_locations']
    window_size = v_d_arguments['window_size']
    vent_location = v_d_arguments['vent_locations']
    vent_size = v_d_arguments['vent_size']
    # door_location = v_d_arguments['door_locations']
    # door_size = v_d_arguments['door_size']

    # function describes: y such that slope(x) = y
    temp = [[0 for col in range(100)] for row in range(100)] # size of room in 10 cm blocks
    x1 =  range(100)
    # up left
    line1_ =  [(i**2 / 100) for i in x1]
    ytemp = 100 - line1_[int(vent_location[0][0] - vent_size/2)]
    y1 = [i + ytemp for i in line1_]
    # These lines can be edited based on # of vents ##############################
    # left
    w1left = window_location[0][0] - window_size/2
    w1right = window_location[0][1] + window_size/2
    v1left = vent_location[0][0] - vent_size / 2
    v1right = vent_location[0][1] + vent_size / 2
    # x2 = [w1left, v1left]
    # y2 = [0, 100]
    m2 = 100/(v1left-w1left)
    b2 = 0 - m2 * w1left
    # down left
    # x3 = [w1right, v1right]
    # y3 = [0, 100]
    m3 = 100/(vent_location[0][0]-w1right)
    b3 = 0 - m3 * w1right
    # down
    w2left = window_location[1][0] - window_size/2
    w2right = window_location[1][1] + window_size/2
    # x4 = [w2left, v1right]
    m4 = 100/(vent_location[0][0]-w2left)
    b4 = 0 - m4 * w2left
    # down right
    # x5 = [w2right, v1right]
    m5 = 100/(v1right-w2right)
    b5 = 0 - m5 * w2right
    # right
    line6_ =  [((100 - i)**2 / 100) for i in x1]
    ytemp = 100 - line6_[int(v1right)]
    y6 = [i + ytemp for i in line6_]
    # between windows
    center = (window_location[1][0] + window_location[0][0]) / 2
    # x7 = [i for i in range(100)]
    curve_down = [-(((i-center) / 5)**2) + center/4 for i in range(100)]

    # define direction plot
    for i in range(100):
        for j in range(100):
            # VENT MATH TO MAKE SURE THIS DOESNT FAIL
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
    
    # velocity
    xtemp = range(100)
    w1 = window_location[0]
    w2 = window_location[1]
    i1 = [((i - vent_location[0][0]) / 3) **2 + 75 for i in xtemp]
    i2 = [((i - vent_location[0][0]) / 3) **2 + 50 for i in xtemp]
    i3 = [((i - vent_location[0][0]) / 3) **2 + 25 for i in xtemp]
    i4 = [((i - vent_location[0][0]) / 3) **2 for i in xtemp]

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

def classroom_simulation(sim_arguments, v_d_arguments, wmr, base_srt):
    '''
    Primary Simulation Function

    Inputs: 
    sim_arguments = {
    :param n_students:          number of students in each sim
    :param n_initial:           number of initial infected students  
    :param n_adults:            number of adults in each sim
    :param student_mask_percent:likelihood of student having mask
    :param adult_mask_percent:  likelihood of adult having mask
    :param n_sims:              number of simulations with new initial students 
    :param mins_per_class:  time step in minutes
    :param classes_per_day:  number of steps in a day
    :param days_per_simulation:   number of days to simulate      
    :param seats:       (x,y) locations for student seating
    :param floor_area:          (x,y) dimensions of room
    }
    v_d_arguments = {
    :param window_locations:    (x,y,z) of window locations
    :param window_size:         surface area of window
    :param vent_locations:      (x,y,z) of vent locations
    :param vent_size:           surface area of vent
    :param door_locations:      (x,y,z) of door locations
    :param door_size:           surface area of door
    :param air_exchange_rate:                 outdoor ACH
    }
    :param aerosol_t_hourly:    Aerosol Transmission rate by hour (Bazant)

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
    '''
    # initialize variables
    inf_params = generate_infectivity_curves()
    infectiousness_curves = plot_infectivity_curves(inf_params, plot=False)
    print(infectiousness_curves)
    l_shape, l_loc, l_scale = inf_params['s_l_s_symptom_countdown']

    student_array = [BaseStudent(id = i, initial = False) for i in range(0, sim_arguments['n_students'])]
    # loop through students and assign seating!!
    for student in student_array:
        student.x = sim_arguments['seats'][student.id][0]
        student.y = sim_arguments['seats'][student.id][1]
        # print(student.id, student.x, student.y)

    student_loc_dict = {str(i): (student_array[i].x, student_array[i].y) for i in range(len(student_array))}
   
    # class_instance = BaseClassroom() # store class_instance.distance and .velocity
    # # update class_instance with user input

    # make the below into a parameter of BaseClassroom?
    direction, velocity = make_velocity_distance(v_d_arguments)

    # conc_dist vars = concentration_array, vent_arr, min_arr, normed_arr

    
    # step_t_sum_rate = crt_step + srt_step + lrt_step


   
    simnum = []
    daynum = []
    stepnum = []
    minnum = []
    infected_id_list = []
    other_id_list = []
    close_range_transmission = []
    shared_room_transmission = []
    long_range_transmission = []
    transmission_by_minute = []
    transmission_by_hour = []
    cause_of_infection = []
    infection_occurs = []



    run_average_array = []
    print('Sims start')
    # loop through n simulations ->
    for sim in range(sim_arguments['n_sims']):
        if sim == int(sim_arguments['n_sims'] / 4):
            print('25\% complete ...')
        elif sim == int(sim_arguments['n_sims'] / 2):
            print('50\% complete ...')
        elif sim == 3 * int(sim_arguments['n_sims'] / 4):
            print('75\% complete ...')
        elif sim == int(sim_arguments['n_sims'] - 1):
            print('99\% complete ...')
        #########################
        init_inf_ids = []
        inf_loc = {}

        initial_ids = np.random.choice(sim_arguments['n_students'], sim_arguments['n_initial'], replace=False)

        # loop through initial infected
        for i in initial_ids:
            # choose random student to be initially infected
            # initialize infectivity curve
            student_array[i].time_to_symptoms = int(np.round(stats.lognorm.rvs(l_shape, l_loc, l_scale, size=1)[0], 0))
            # fix overflow errors (unlikely but just in case)
            if student_array[i].time_to_symptoms >= 18:
                student_array[i].time_to_symptoms = 17
                print('overflow up')
            elif student_array[i].time_to_symptoms <= 0:
                student_array[i].time_to_symptoms = 0
                print('overflow down')
            # initialize infectivity of student
            # init_time_to_symp[str(i)] = student_array[i].infectivity.time_to_symptoms
            student_array[i].infectivity = infectiousness_curves.iloc[student_array[i].time_to_symptoms].gamma
            student_array[i].is_infected = True
            init_inf_ids.append(student_array[i].id)
            inf_loc[str(student_array[i].id)] = (student_array[i].x, student_array[i].y)
            # init_infectivity[str(i)] = student_array
            # infectiousness_curves[student_array[i].infectivity.time_to_symptoms].gamma
            # student_array[i].is_infected = True
            # print(student_array[initial_].id, student_array[initial_].is_infected, temp_inf)
        


        # airflow proxy
        if wmr == False:
            temp_conc, vent_conc, min_conc, norm_conc = concentration_distribution_(v_d_arguments['air_exchange_rate'], direction, velocity, sim_arguments['seats'], init_dict=inf_loc)
            airflow_proxy = norm_conc
        else:
            airflow_proxy = 0

        run_chance_zero = 1 # setup for chance nonzero
        # d days in each simulation ->
        for day in range(sim_arguments['days_per_simulation']):
            # update infectivity
            for student in student_array:
                # print(student.id, student.is_infected, student.time_to_symptoms)
                if student.is_infected == True and student.infectivity != 0:
                    # update infectivity based on curve
                    student.time_to_symptoms += day 
                    if student.time_to_symptoms >= 18:
                        student.time_to_symptoms = 17
                        print('overflow above')
                    student.infectivity = infectiousness_curves.iloc[student.time_to_symptoms].gamma
                    print('update infection')
                elif student.is_infected:
                    student.time_to_symptoms = int(np.round(stats.lognorm.rvs(l_shape, l_loc, l_scale, size=1)[0], 0))
                    student.infectivity = infectiousness_curves.iloc[student.time_to_symptoms].gamma
                    print('new infection!')
                else:
                    pass               
            # loop through steps (class periods)
            for step in range(sim_arguments['classes_per_day']):
                # loop through minutes per class
                for min in range(sim_arguments['mins_per_class']):
                    # loop through infectious students
                    for student_i in student_array:
                        if student_i.is_infected == False:
                            pass
                        elif student_i.infectious == True:
                            for student_o in student_array:
                                if student_o.is_infected == True:
                                    pass
                                else:
                                    raw_dist = math.sqrt( ((student_o.y - student_i.y)**2) + ((student_o.x - student_i.x)**2) )
                                    distance = (raw_dist / 100) * math.sqrt(sim_arguments['floor_area'])
                                    # generate Bool for transmission and string for cause of infection    
                                    new_infection, cause, t_rates = minute_transmission(student_o.id, student_i.id, base_srt, airflow_proxy=airflow_proxy, wmr=wmr, seats=student_loc_dict, mask_var=sim_arguments['student_mask_percent'], floor_area=sim_arguments['floor_area'], initial_infect=student_i.infectivity, distance=distance)
                                    simnum.append(sim)
                                    daynum.append(day)
                                    stepnum.append(step)
                                    minnum.append(min)
                                    infected_id_list.append(student_i.id)
                                    other_id_list.append(student_o.id)
                                    close_range_transmission.append(t_rates['crt_minute'])
                                    shared_room_transmission.append(t_rates['srt_minute'])
                                    long_range_transmission.append(t_rates['lrt_minute'])
                                    transmission_by_minute.append(t_rates['t_rate_by_minute'])
                                    transmission_by_hour.append(t_rates['t_rate_by_hour'])
                                    cause_of_infection.append(cause)
                                    infection_occurs.append(new_infection)
                                    if new_infection == True:
                                        student_o.is_infected = True
                        else:
                            infectious_ = np.random.choice([True, False], p=[student_i.infectivity, 1 - student_i.infectivity])
                            if infectious_ == True: 
                                student_i.infectious = True
                            else:
                                pass
                    # end of each step
            # iterate infectivity of each infected student

            # end of each day
        
        # end of each simulation
        print('100\% complete! Plotting output ...')
    t_rate_dict = {
        'Simulation #': simnum,
        'Day #': daynum,
        'Step #': stepnum,
        'Minute #': minnum,
        'Infected ID': infected_id_list,
        'Other ID': other_id_list,
        'Close Range Transmission': close_range_transmission,
        'Shared Room Transmission': shared_room_transmission,
        'Long Range Transmission': long_range_transmission,
        'Transmission by Minute': transmission_by_minute,
        'Transmission by Hour': transmission_by_hour,
        'Cause of Infection': cause_of_infection,
        'Infection Occurs': infection_occurs
    }
    # output:
    param_output = {
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
    output_df = pd.DataFrame.from_dict(t_rate_dict)

    # # output needed: 
    # 1. input params (user + default + calculated)
    # 2. 
    # 3. density estimation of exposure rate
    # 4. hist of mask wearing: y = num students, x = infection rate
    # 5. hist of ventilation: y = num students, x = infection rate

    # 6. scatter of transmission methods: infection rate vs distance
    # 7. scatter of transmission methods: infection rate vs time
    # 8. scatter of transmission methods: distance vs time with % thresholds for orders of magnitude
    return param_output, output_df