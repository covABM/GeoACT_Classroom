# import
import json
from os import O_TEMPORARY
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import math 
from src.classroom import *
from src.infection import *

def load_parameters_av(filepath):
    '''
    Loads input and output directories

    Handles script running anywhere (terminal/notebook/etc...)
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

class user_viz():
    '''
    simulation class that holds bulk of vars and functions

    :function init:
    :function load_parameters:
    :function merv_to_eff:
    :function generate_class_seating:
    :function model_run:
    
    '''
    def __init__(self, targets, parent=None):
        '''
        Generate local instance of room based on user and default inputs

        ToDo list:
        Improve Seating Chart and assignment for n students
        More classroom customization

        :param targets:
        '''
        super(user_viz, self).__init__()

        self.input_params = targets
        # for i in self.input_params:
        #     print(i, self.input_params[i])

        # Simulation Parameters
        self.room_type = self.input_params['room_type']
        self.n_students = self.input_params['num_students']
        self.n_initial_students = self.input_params['num_initial']
        self.n_adults = self.input_params['num_adults']
        self.n_sims = self.input_params['n_sims']
        self.age_group = self.input_params['age_group']
        self.mins_per_class = self.input_params['mins_per_class']
        self.classes_per_day = self.input_params['classes_per_day']
        self.days_per_simulation = self.input_params['days_per_simulation']
        self.well_mixed_room = self.input_params['well_mixed_room']
        self.ventilation = self.input_params['ventilation']

        # Human Parameters
        self.mean_breathing_rate = self.input_params['breathing_rate']
        self.respiratory_activity = self.input_params['respiratory_activity']
        self.student_mask_percent = self.input_params['student_mask_percent']
        self.adult_mask_percent = self.input_params['adult_mask_percent']
        self.mask_protection_rate = self.input_params['mask_protection_rate']

        # Room Parameters
        if self.input_params['floor_area'] == 0:
            if self.room_type == 'small':
                self.floor_area = 900
            elif self.room_type == 'large':
                self.floor_area = 2000
            else:
                print('Please select Valid Room Type... Defaulting to small classroom')
        else:
            self.floor_area = self.input_params['floor_area']

        self.room_height = self.input_params['room_height']
        self.vent_size = self.input_params['vent_size']
        self.vent_locations = self.input_params['vent_locations']
        self.window_size = self.input_params['window_size']
        self.window_locations = self.input_params['window_locations']
        self.door_size = self.input_params['door_size']
        self.door_location = self.input_params['door_locations']
        self.seating_chart = self.input_params['seating_chart']

        # Vent Parameters
        # ACH is outdoor air_exchange-rate
        if self.input_params['ach_level'] == 0:
            if self.ventilation == 'Closed Windows':
                self.air_exchange_rate = 0.3 # TESTING for self.ventilation closed
            elif self.ventilation== 'Open Windows':
                self.air_exchange_rate = 2
            elif self.ventilation=='Mechanical':
                self.air_exchange_rate= 5
            elif self.ventilation=='Open Windows and Fans':
                self.air_exchange_rate = 6
            elif self.ventilation=='Better Mechanical':
                self.air_exchange_rate= 9
            elif self.ventilation=='Outdoors':
                self.air_exchange_rate= 20
            else:
                print('ERROR in setting ACH... Defaulting to 6 (Open Windows and Fans)')
                self.air_exchange_rate = 6
        else:
            self.air_exchange_rate = self.input_params['ach_level']
        
        
        self.merv_level = self.input_params['merv_level']
        self.recirc_rate = self.input_params['recirc_rate']
        self.relative_humidity = self.input_params['relative_humidity']
        self.primary_outdoor_air_fraction = self.input_params['primary_outdoor_air_fraction']

        # Advanced Parameters
        self.strain = self.input_params['strain']
        self.crit_drop_radius = self.input_params['crit_droplet_radius']
        self.viral_deact_rate = self.input_params['viral_deact_rate']
        self.immunity_rate = self.input_params['immunity_rate']
        self.child_vax_rate = self.input_params['child_vax_rate']
        self.adult_vax_rate = self.input_params['adult_vax_rate']
        self.viral_infectivity = self.input_params['viral_infectivity']
        if self.input_params['merv_level'] == 0:
            self.aerosol_filtration_eff = self.input_params['aerosol_filtration_eff']
        else:
            print('aerosol filtration derived from MERV level')
            # TODO: add default value for this based MERV level
            self.aerosol_filtration_eff = 0.06 / self.merv_level # default value

        self.output_filepath = "output/"

    def load_parameters(self, filepath):
        '''
        Loads seating from input directories

        :param filepath: path of json file to load
        '''
        # print(os.getcwd(), 'av_cwd')
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

    def merv_to_eff(self):
        '''
        convert merv level effective ACH level

        TODO: use effective drop radius not critical radius

        :return: 
        '''


        merv_dict = [
            {'merv': 1, '0.3-1': 0.01, '1-3': 0.01, '3-10': 0.01},
            {'merv': 2, '0.3-1': 0.01, '1-3': 0.01, '3-10': 0.01},
            {'merv': 3, '0.3-1': 0.01, '1-3': 0.01, '3-10': 0.01},
            {'merv': 4, '0.3-1': 0.01, '1-3': 0.01, '3-10': 0.01},
            {'merv': 5, '0.3-1': 0.01, '1-3': 0.01, '3-10': 0.2},
            {'merv': 6, '0.3-1': 0.01, '1-3': 0.01, '3-10': 0.35},
            {'merv': 7, '0.3-1': 0.01, '1-3': 0.01, '3-10': 0.50},
            {'merv': 8, '0.3-1': 0.01, '1-3': 0.20, '3-10': 0.70},
            {'merv': 9, '0.3-1': 0.01, '1-3': 0.35, '3-10': 0.75},
            {'merv': 10, '0.3-1': 0.01, '1-3': 0.50, '3-10': 0.80},
            {'merv': 11, '0.3-1': 0.2, '1-3': 0.65, '3-10': 0.85},
            {'merv': 12, '0.3-1': 0.35, '1-3': 0.80, '3-10': 0.90},
            {'merv': 13, '0.3-1': 0.50, '1-3': 0.85, '3-10': 0.90},
            {'merv': 14, '0.3-1': 0.75, '1-3': 0.90, '3-10': 0.95},
            {'merv': 15, '0.3-1': 0.85, '1-3': 0.90, '3-10': 0.95},
            {'merv': 16, '0.3-1': 0.95, '1-3': 0.95, '3-10': 0.95},
            {'merv': 17, '0.3-1': 0.9997, '1-3': 0.9997, '3-10': 0.9997},
            {'merv': 18, '0.3-1': 0.99997, '1-3': 0.99997, '3-10': 0.99997},
            {'merv': 19, '0.3-1': 0.999997, '1-3': 0.999997, '3-10': 0.999997},
            {'merv': 20, '0.3-1': 0.9999997, '1-3': 0.9999997, '3-10': 0.9999997},
        ]
        if self.merv_level == 0:
            return 0
        eff = 0
        merv = np.floor(max(self.merv_level, min(1, 20)))
        for item in merv_dict:
            if item['merv'] == merv:
                if self.crit_drop_radius < 1:
                    eff = item['0.3-1']
                elif self.crit_drop_radius < 3:
                    eff = item['1-3']
                else:
                    eff = item['3-10']

        return eff

    def generate_class_seating(self):
        '''
        Based on:
        - class type
        - seating chart
        - number of students
        - number of adults
        
        Return dict of student_id: [x, y] for a good seating chart
        
        grid = .2 -> .8

        '''
        # square room ezpz
        max_width = int(self.floor_area / 10)
        min_width = 5
        max_length = int(self.floor_area / 10 - 10) # teacher in blank space
        min_length = 5
        num_seats_each_way = int(math.ceil(max_width / math.sqrt(self.n_students)))

        # create seats based on class type and seating chart
        if self.seating_chart == 'grid':
            x_s = [x for x in range(min_width, max_width + 1, num_seats_each_way)]
            y_s = [y for y in range(min_length, max_length + 1, num_seats_each_way)]
            x_y_combo = [(x, y) for x in x_s for y in y_s]
            seats = {}
            for i in range(self.n_students):
                seats[i] = x_y_combo[i]
        elif self.seating_chart == 'circular':
            pass
        else:
            print('please enter valid seating')
            pass
        print(seats)
        return seats

    # function to run model with user input
    def model_run(self):
        '''
        Updated 7/7/21 with _bus code

        1 SETUP
        2 LOADING
        3 CONCENTRATION
        4 MASK HISTOGRAM
        5 AIRFLOW HISTOGRAM
        6 SCATTER
        7 RISK VS TIME

        Sim Variables
        class_arguments = {
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
        :param air_exchange_rate:   ACH exchange rate of air between room and outside
        }
        :param aerosol_t_rate:      aerosol transmission rate

        :return:
        '''
        # update 7/18 
        # 0 Variables
        # Calculated parameters
        

        # 1 SETUP + CALCULATED VARIABLES
        input_args = {
            'floor_area': self.floor_area,
            'room_height': self.room_height,
            'air_exchange_rate': self.air_exchange_rate,
            'breathing_flow_rate': self.mean_breathing_rate,
            'aerosol_filtration_eff': self.aerosol_filtration_eff,
            'relative_humidity': self.relative_humidity,
            'exhaled_air_inf': self.viral_infectivity,
            'max_viral_deact_rate': self.viral_deact_rate,
            'mask_passage_prob': self.mask_protection_rate,
            'max_aerosol_radius': self.crit_drop_radius,
            'primary_outdoor_air_fraction': self.primary_outdoor_air_fraction
        }

        self.baseline_srt = return_aerosol_transmission_rate(input_args)

        self.seats = self.generate_class_seating()
        sim_arguments = {
            'n_students': self.n_students,
            'n_initial': self.n_initial_students,
            'n_adults': self.n_adults,
            'student_mask_percent': self.student_mask_percent,
            'adult_mask_percent': self.adult_mask_percent,
            'n_sims': self.n_sims,
            'mins_per_class': self.mins_per_class,
            'classes_per_day': self.classes_per_day,
            'days_per_simulation': self.days_per_simulation,
            'seats': self.seats,
            'floor_area': self.floor_area
        }
        v_d_arguments = {
            'window_locations': self.window_locations,
            'window_size': self.window_size,
            'vent_locations': self.vent_locations,
            'vent_size': self.vent_size,
            'door_location': self.door_location,
            'door_size': self.door_size,
            'air_exchange_rate': self.air_exchange_rate
        }

        # 2 USER SIMULATIONS
        param_dict, output_df = classroom_simulation(sim_arguments, v_d_arguments, wmr=self.well_mixed_room, base_srt=self.baseline_srt)
        print('output df columns', output_df.columns)

        output_df.to_csv(self.output_filepath + 'sim_data.csv', index=False)
        # return param_dict, output_df
        # 3 DENSITY ESTIMATION OF /STEP TRANSMISSION RATE

        # out of practice!!!!

        plt.figure(figsize=(10, 10))
        density_df = output_df.copy()
        density_plot = density_df.groupby(['Day #', 'Step #', 'Minute #'])['Transmission by Minute'].mean()
        plt.hist(density_plot)

        density_filename = self.output_filepath + 'density_plot.png'
        plt.savefig(density_filename, dpi=300)


        # 4 MASK HISTOGRAM

        # make the plots from the dataframe!!!!

        # 5 AIRFLOW HISTOGRAM

        # 6 SCATTER LRT vs SRT vs SRT
        # infection rate vs distance
        # infections rate avg vs time
        # infection boundary of time vs distance

        # range_test_plot = density_

        # 7 RISK VS TIME






        # # 1 SETUP
        # print('Model Setup...')

        # # class_sim()

        # # run class model
        # class_seating = self.generate_class_seating()

        # # seat var is room type
        # # make varying input setup for ventilation

        # class_trip, conc_array, out_mat, chance_nonzero = class_sim(n_students = int(self.students_var), mask = self.mask_var, n_sims = self.number_simulations, duration = self.duration, initial_seating = self.room_type, loc_params=temp_loc) # replace default with selected

        # ### Validate using chance_nonzero



        # self.chance_nonzero = chance_nonzero
        # # print(chance_nonzero, 'more than none?')
        # self.conc_array = conc_array
        # self.class_trips.append(class_trip)
        # # print('model_run start')
        # plt.figure(figsize=(5,4))#, dpi=300)
        # plt.gcf().set_size_inches(5,4)
        # # plt.gcf().set_size_inches(5,4)
        # # ax = plt.gca()
        # pd.Series(class_trip).plot.kde(lw=2, c='r')
        # plt.title('Density estimation of exposure')
        # # plt.xlim(0, .004)
        # # print(plt.xticks())

        # # set x ticks
        # temp_x = np.array(plt.xticks()[0])
        # str_x = np.array([str(round(int * 100, 2))+'%' for int in temp_x])
        # plt.xticks(temp_x, str_x)

        # plt.ticklabel_format(axis="x")#, style="sci", scilimits=(0,0))

        # plt.yticks(np.arange(0, 3500, 700), np.arange(0, 3500, 700) / 3500)

        # ##### This is temporary chill tf out ####
        # # rescale y axis to be % based
        # plt.xlabel('Likelihood of exposure to infectious dose of particles                         ')
        # plt.ylabel('Density estimation of probability of occurrence')
        # plt.savefig('results/window_curve.png', dpi=300)
        # # plt.show()
        # print('model_run complete!')

        # # temp variables
        # self.chance_nonzero = 0
        # self.conc_array = 0

        


        # # 2 LOADING
        # plt.figure()

        # if self.room_type == 'small':
        #     self.seat_dict = load_parameters_av(filepath='config/small_classroom.json')
        # elif self.room_type == 'large':
        #     self.seat_dict = load_parameters_av(filepath='config/large_classroom.json')
        # # implement SEATING CHART OPTIONS ############################

        # # 3 CONCENTRATION + 1

        # print('Plotting Concentration ...')
        # x_arr = []
        # y_arr = []
        # for i in self.seat_dict.items(): ################### change seating
        #     x_arr.append(i[1][1])
        #     y_arr.append(i[1][0] * 1.5 + 1) # seat fix
        # rot = mpl.transforms.Affine2D().rotate_deg(180)

        # # Set up Figure
        # fig, ax1 = plt.subplots()
        # plt.matshow(out_mat, cmap="OrRd", norm=mpl.colors.LogNorm())
        # plt.gcf().set_size_inches(2,2)
        # plt.suptitle('Viral Concentration Heatmap', fontsize=7.5)
        # plt.axis('off')
        # plt.text(.1, .01, '\nSample proxy for air flow after ' + str(self.duration) + ' minutes\n', fontsize=4)
        # plt.savefig(output_filepath + '_concentration.png', dpi=300)
        # plt.close()

        # # 4 MASK HISTOGRAM

        # fig_mask, ax_mask = plt.subplots()
        # mask_values = [70, 80, 90, 100]
        # mask_legend = [str(i) + '%' for i in mask_values]
        # print('ml', mask_legend)

        # for mask_ in mask_values:
        #     '''
        #     Note: Masks as referred to here are in terms of face masks

        #     coding 'masks' will be noted when used
        #     '''
        #     class_trip_mask, conc_array_mask_mask, out_mat_mask, chance_nonzero_mask = class_sim(n_students = int(self.students_var), mask = self.mask_var, n_sims = self.number_simulations, duration = self.duration, initial_seating = self.room_type, loc_params=temp_loc) # replace default with selected


        #     sns.distplot(list(class_trip_mask[2].values()), ax=ax_mask, rug=True, kde=False, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1})

        # fig_mask.savefig(output_filepath + '_masks.png', dpi=300)

        # 5 AIRFLOW HISTOGRAM



        # 6 SCATTER
        # 7 RISK VS TIME



        # print('Windows...')
        # fig2, ax2 = plt.subplots()
        # window_types = [0, 6]
        # win_out_df = pd.DataFrame(columns=window_types)
        # temp = 0.051
        # temp_step = 0.001

        # add dynamic x range ToDo
        #
        # for w in window_types:
        #     bus_out_array, conc_array, out_mat, chance_nonzero, avg_mat = class_sim(int(self.students_var), self.mask_var, self.number_simulations, self.duration, self.seat_var, w) # WINDOW
        #     x_range = [.051, .102, .153, .204]
        #
        #     ## 7/4 TODO: why is it all going wrong
        #
        #
        #     for i in range(len(x_range)):
        #         if x_range[i] < max(bus_out_array[2].values()):
        #             pass
        #         else:
        #             temp = x_range[i]
        #             temp_step = 0.001 * (i + 1)


            # TODO: Check all values of KDE are positive

            # pd.Series(bus_out_array[2]).plot.kde(alpha=.5, ax=ax2)
            # pd.Series(bus_out_array[2]).plot.hist(alpha=.5, ax=ax2)

            ###############################################

            # SEABORN
            # sns.distplot(list(bus_out_array[2].values()), ax=ax2, rug=True, kde=False, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1})


        # fig2.legend(['Windows Closed', 'Windows Open 6 Inches'])
        # plt.xlabel('Mean likelihood of transmission at each step')
        # plt.ylabel('Number of students with this average risk of transmission')
        # seat_filepath_2 = output_filepath + '_windows.png'
        # fig2.savefig(seat_filepath_2, dpi=300)
        # plt.close(fig2)
        # print('Windows complete!')

        # Hist 3 Masks

        # print('Masks...')
        # fig3 = plt.figure(3)
        # mask_amount = [1, .9, .8, .7]
        # print('start masks')
        # colorlist = ['blue', 'green', 'yellow', 'red']
        # count_ = 0
        #
        # for m in mask_amount:
        #     bus_out_array, conc_array, out_mat, chance_nonzero, avg_mat = class_sim(int(self.students_var), m, self.number_simulations, self.duration, self.seat_var, self.window_var) # SEATING
        #     pd.Series(bus_out_array[2]).plot.hist(bins=np.arange(0, 0.056, 0.001), alpha=.5, color=colorlist[count_])
        #     count_ += 1
        # plt.legend(['100% Mask compliance', '90% Mask compliance', '80% Mask compliance', '70% Mask compliance'])
        # plt.xlabel('Mean likelihood of transmission at each step')
        # plt.ylabel('Number of students with this average risk of transmission')
        # seat_filepath_3 = output_filepath + '_masks.png'
        # fig3.savefig(seat_filepath_3, dpi=300)
        # plt.close(fig3)
        # print('Masks complete!')



        # 5 SCATTER/KDE + 2




        # 6 T_RISK AVERAGE + 1
