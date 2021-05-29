# imports
import numpy as np
import pandas as pd
import math
from scipy import stats
import sys
import json
import os
import matplotlib.pyplot as plt

sys.path.insert(0, 'config')
# import from infection.py
from infection import generate_infectivity_curves, plot_infectivity_curves, return_aerosol_transmission_rate

def load_parameters(filepath):
    '''
    Loads input and output directories
    '''
    try:
        with open(filepath) as fp:
            parameter = json.load(fp)
    except:
        try:
            with open('../' + filepath) as fp:
                parameter = json.load(fp)
        except:
            with open('../../' + filepath) as fp:
                parameter = json.load(fp)

    return parameter

large_class = load_parameters('config/large_classroom.json')

small_class = load_parameters('config/small_classroom.json')
print(os.getcwd(), '#############################################')
print(os.listdir(), os.listdir('config'))
select_dict = load_parameters('config/neighbor_logic.json')

#
aerosol_params = load_parameters('results/aerosol_data_.json')
dp = load_parameters('results/default_data_.json')

def get_distance_class(student_pos, this_id, initial_id):
    x1, y1 = student_pos[initial_id]
    x2, y2 = student_pos[this_id]
    return x1, x2, y1, y2

def get_incoming(x, y, old, class_flow_direction, class_flow_velocity):
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

def make_new_heat(old, class_flow_pos, class_flow_direction, class_flow_velocity, init_infected_ = None):
    '''
    1 minute step used to calculate concentration_distribution iteratively
    '''
    kids =list(class_flow_pos.keys())
    if not init_infected_:
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
            new_val = old[i][j] + (1/(2.02 ** dist)) # 1 quanta per step by distance
            new[i][j] = new_val

            ##################################################
    for i in range(len(new)):
        for j in range(len(new[i])):
            neighbor_val = get_incoming(i, j, new, class_flow_direction, class_flow_velocity)
            air_val = air_effects(i, j, neighbor_val)
            out[i][j] = air_val  # +=???
    return out, init_infected_

def concentration_distribution(num_steps, num_sims, class_flow_pos, class_flow_direction, class_flow_velocity, room_size):
    '''
    Simulate distribution of concentration after
    30 steps
    100 runs
    random initial student/infectivity
    '''
    array_size = [int(room_size.split('x')[0]), int(room_size.split('x')[1])]
    nothings = np.zeros(array_size)
    avg_array = nothings.copy()
    temp, initial = make_new_heat(nothings, class_flow_pos, class_flow_direction, class_flow_velocity, init_infected_=None)
    temp_array = []

    for step in range(num_steps):
        temp, initial = make_new_heat(temp, class_flow_pos, class_flow_direction, class_flow_velocity, init_infected_=initial)
        temp_array.append(temp)

    for i in range(len(temp_array)):
        # timesteps
        for y in range(len(temp_array[0])):
            for x in range(len(temp_array[0][0])):
                avg_array[y][x] += (temp_array[i][y][x] / len(temp_array))

    # plt.subplot(1, 2, 1)
    # directions = plt.matshow(class_flow_direction, cmap=plt.get_cmap("OrRd"))
    #
    # plt.subplot(1, 2, 2)
    # velocity = plt.matshow(class_flow_velocity, cmap=plt.get_cmap("OrRd"))
    # plt.axis('off')
    # plt.savefig('results/direction_velocity.png', dpi=300)

    plt.close()
    plt.subplot()
    plt.matshow(avg_array)
    plt.savefig('results/temp.png', dpi=300)

    return temp_array, avg_array

def make_velocity_distance(window1, window2, door_location, vent_location, window_size, vent_size):
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
    m3 = 100/(v1right-w1right)
    b3 = 0 - m3 * w1right
    # down
    w2left = window2[0] - window_size/2
    w2right = window2[0] + window_size/2
    x4 = [w2left, v1left]
    m4 = 100/(v1left-w2left)
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

    # define direction plot
    for i in range(100):
        for j in range(100):
            if j < curve_down[i]: # between windows
                temp[j][i] = 3
            elif (j > y1[i]): # top left
                temp[j][i] = 7
            elif (j > m2 * i + b2): # left edge
                temp[j][i] = 4
            elif (j > m3 * i + b3): # window 1
                temp[j][i] = 1
            elif (j < m4 * i + b4): # window 1 and 2
                temp[j][i] = 2
            elif (j < m5 * i + b5): # window 2
                temp[j][i] = 3
            elif (j < y6[i]): # window 2
                temp[j][i] = 6
            else:  # top right
                temp[j][i] = 9
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
def class_sim(n_students, mask, n_sims, duration, initial_seating, loc_params): # do 1 trip with given params
    '''
    in:
    mask %
    windows up / down
    students: 28 or 56

    '''
    if initial_seating == "small":
        seat_dict = load_parameters('config/small_classroom.json')
    elif initial_seating == "large":
        seat_dict = load_parameters('config/large_classroom.json')
    else:
        seat_dict = load_parameters('config/small_classroom.json') ## TODO
        print('Left to be implemented: Library, Sports, Gym, Theatre')
    flow_seating = {key: value for key, value in seat_dict.items() if int(key) < n_students}
    # initialize model run data storage
    who_infected_class = {str(i): 0 for i in range(len(flow_seating.keys()))}
    init_inf_dict = who_infected_class.copy()
    n_steps = int(int(duration) / 5)
    transmission_class_rates = {i: [] for i in flow_seating.keys()}
    temp_rates = transmission_class_rates.copy()
    averaged_all_runs = transmission_class_rates.copy()

    #### Future: loc_params gives these from user input/ slider
    w1 = (25, 0)
    w2 = (75, 0)
    door = (20, 96)
    vent = (50, 96)
    window_size = 8
    vent_size = 2
    ################################################3
    direction, velocity = make_velocity_distance(w1, w2, door, vent, window_size, vent_size)

    # get infective_df
    temp = generate_infectivity_curves()
    inf_df = plot_infectivity_curves(temp, plot= False)
    sls_symp_count, x_symp_count, s_l_s_infectivity_density, x_infectivity, distance_multiplier = temp
    l_shape, l_loc, l_scale = sls_symp_count
    g_shape, g_loc, g_scale = s_l_s_infectivity_density

    # print(dp, 'default')
    aerosol = return_aerosol_transmission_rate(aerosol_params['floor_area'], aerosol_params['mean_ceiling_height'], aerosol_params['air_exchange_rate'], aerosol_params['aerosol_filtration_eff'], aerosol_params['relative_humidity'], aerosol_params['breathing_flow_rate'], aerosol_params['exhaled_air_inf'], aerosol_params['max_viral_deact_rate'], aerosol_params['mask_passage_prob'])
    concentration_array, avg_matrix = concentration_distribution(n_steps, n_sims, seat_dict, direction, velocity, room_size='100x100')
    # return average concentration over run
    print('ok', len(concentration_array))
    print(len(concentration_array[0]))
    print(len(concentration_array[0][0]))
    out_matrix = np.array(np.zeros(shape=concentration_array[0].shape))
    max_val = 0
    for conc in range(len(concentration_array)):
        for y in range(concentration_array[conc].shape[0]):
            for x in range(concentration_array[conc].shape[1]):
                out_matrix[y][x] += (concentration_array[conc][y][x] / len(concentration_array))
                if  (concentration_array[conc][y][x] / len(concentration_array)) > max_val:
                    max_val =  (concentration_array[conc][y][x] / len(concentration_array))

    concentration_ = out_matrix
    for conc in range(len(concentration_array)):
        for y in range(concentration_array[conc].shape[0]):
            for x in range(concentration_array[conc].shape[1]):
                if out_matrix[y][x] < 0:
                    out_matrix[y][x] *= -1
                # out_matrix[y][x] = out_matrix[y][x] / max

    # print(concentration_, 'concentration')
    run_average_array = []
    for run in range(n_sims):
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


        for step in range(n_steps): # infection calculated for 5-minute timesteps
            # class trip 1way
            # iterate through students
            # print(seat_dict.keys())
            for student_id in flow_seating.keys():
                if student_id != initial_inf_id:
                    # masks wearing %
                    cwd = os.getcwd()
                    if isinstance(mask, str):
                        mask_ = int(mask.split('%')[0]) / 100
                        masks = np.random.choice([.1, 1], p=[mask_, 1-mask_])
                    else:
                        # print(mask, type(mask))
                        masks = np.random.choice([.1, 1], p=[mask, 1-mask])

                    x1, x2, y1, y2 = get_distance_class(flow_seating, student_id, initial_inf_id)
                    distance = math.sqrt(((.3 * (x2 - x1))**2)+((.3 * (y2-y1))**2))
                    chu_distance = 1 / (2.02 ** distance)

                    # for concentraion calculation
                    air_y, air_x = flow_seating[str(student_id)]
                    # print(student_id, 'id', air_x, air_y)

                    # proxy for concentration
                    air_flow = concentration_[air_y][air_x]

                    transmission = (init_infectivity * chu_distance * masks) + (air_flow * aerosol)
                    if transmission > 0.03:
                        # print('why')
                        transmission = .03
                        # print(air_flow, 'af')# * aerosol)
                    # calculate transmissions
                    if np.random.choice([True, False], p=[transmission, 1-transmission]):
                        who_infected_class[student_id] += 1
                    # if infected:
                    run_chance_of_0 *= (1-transmission)

                    # output temp is for each step
                    temp_average_array[student_id].append(transmission)
        run_average_array.append(1 - run_chance_of_0) # add chance of nonzero to array
        # takes average over model run
        for id in flow_seating.keys():
            if len(temp_average_array[id]) > 0:
                transmission_class_rates[id] = np.mean(temp_average_array[id])
    # takes average over all runs
    for id in flow_seating.keys():
        averaged_all_runs[id] = np.mean(transmission_class_rates[id])

    # average risk of >= 1 infection across all model runs
    if len(run_average_array) == 0:
        print('Sim failed')
    run_avg_nonzero = np.mean(run_average_array)
    # print('initially infected counts', init_inf_dict)
    # OUTPUT AVERAGE LIKELIHOOD OF >= 1 INFECTION
    # print(n_students)
    return averaged_all_runs, concentration_array, out_matrix, run_avg_nonzero
