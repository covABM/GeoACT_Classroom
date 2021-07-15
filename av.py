# import
import json
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
if 'src' in sys.path:
    print('success')
else:
    sys.path.insert(0, 'src')
from classroom import class_sim
from infection import return_aerosol_transmission_rate

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
    def __init__(self, targets, parent=None):
        '''
        Generate local instance of room based on user and default inputs

        TODO list:
        Improve Seating Chart and assignment for n students
        More classroom customization


        Make input from user not just default

        '''
        super(user_viz, self).__init__()
        self.input_params = load_parameters_av(filepath='results/default_data_.json')

        # Simulation Parameters
        self.simulation_defaults = load_parameters_av(filepath='config/simulation_defaults.json')
        print(self.simulation_defaults, 'sim_defaults')

        # Setup
        self.room_type = self.simulation_defaults["setup"]["room_type"] 
        self.age_group = self.simulation_defaults['setup']['age_group']
        self.number_of_students = self.simulation_defaults["setup"]["num_students"]
        self.number_of_adults = self.simulation_defaults["setup"]["num_adults"]

        # Duration
        self.mins_per_event = self.simulation_defaults["duration"]["mins_per_event"]
        self.events_per_day = self.simulation_defaults["duration"]["events_per_day"]
        self.days_per_sim = self.simulation_defaults["duration"]["days_per_sim"]

        # Room Parameters
        self.room_defaults = load_parameters_av(filepath='config/room_defaults.json')
        # Type
        self.room_area = self.room_defaults[self.room_type]["area"]
        self.room_height = self.room_defaults[self.room_type]
        self.vent_locations = self.room_defaults[self.room_type]
        self.window_location = self.room_defaults[self.room_type]
        if self.room_type == 'small':
            self.seat_dict = load_parameters_av(filepath='config/small_classroom.json')
        elif self.room_type == 'large':
            self.seat_dict = load_parameters_av(filepath='config/large_classroom.json')
        elif self.room_type in ['library', 'gym', 'sports practice', 'band room', 'theatre']:
            self.seat_dict = load_parameters_av(filepath='config/small_classroom.json')
            print('Other options under development- please reach out if you need them expedited!')
        else:
            print('Error! Problem loading seating')

        # Ventilation Parameters
        self.vent_defaults = load_parameters_av(filepath='config/vent_defaults.json')
        self.outdoor_ach = self.vent_defaults["outdoor_ach"]
        self.merv_type = self.vent_defaults["MERV_rating"]
        self.recirc_rate = self.vent_defaults["recirculation_rate"]
        self.relative_humidity = self.vent_defaults["relative_humidity"]

        # Human Parameters
        self.human_defaults = load_parameters_av(filepath='config/human_defaults.json')
        # Behavior
        self.indiv_breathing_rate = self.human_defaults["behavior"]["breathing_rate"]
        self.respiratory_activity = self.human_defaults["behavior"]["respiratory_activity"]
        self.student_mask_percent = self.human_defaults["behavior"]["student_mask_wearing_percent"] # Likelihood a given student is wearing their mask to its full effect
        self.adult_mask_percent = self.human_defaults["behavior"]["adult_mask_wearing_percent"]
        self.mask_protection = self.human_defaults["behavior"]["mask_protection"]
        self.mean_breathing_rate = 'Take average of indiv_breathing_rate' # or just make this another default like Bazant

        # Advanced Parameters
        self.advanced_defaults = load_parameters_av(filepath='config/advanced_defaults.json')
        # Constrained by lack of certainty
        self.strain = self.advanced_defaults["constrained"]["strain"]
        self.crit_drop_radius = self.advanced_defaults["constrained"]["crit_drop_radius"]
        self.viral_deact_rate = self.advanced_defaults["constrained"]["viral_deact_rate"]
        self.immunity_rate = self.advanced_defaults["constrained"]["immunity_rate"]
        self.child_vax = self.advanced_defaults["constrained"]["child_vacc_rate"]
        self.adult_vax = self.advanced_defaults["constrained"]["adult_vacc_rate"]

        self.vaccination_effects = self.advanced_defaults["constrained"]["v"]
        self.viral_infectivity = self.advanced_defaults["constrained"][""]# per virion
        # Proxy for complex problem
        self.chu_proxy_vars = self.advanced_defaults["constrained"][""]
        self.chen_proxy_vars = self.advanced_defaults["constrained"][""]
        self.user_units = self.advanced_defaults["constrained"][""]
        self.initial_infected = self.advanced_defaults["constrained"][""]
        # Calculated during simulation
        self.indoor_WMR_ach = self.advanced_defaults["constrained"][""]

        # Other Initializations







        self.room_size = self.input_params["room_size"] # in m
        self.students_var = self.input_params["number_students"]
        self.mask_var = self.input_params["mask_wearing_percent"]
        self.window_var = self.input_params["windows"]
        self.duration = self.input_params["duration"] #
        self.number_simulations = self.input_params["number_simulations"]

        self.input_params2 = load_parameters_av(filepath='results/aerosol_data_.json')
        self.mask_eff = self.input_params2["mask_passage_prob"]
        self.room_type = self.input_params2["room_type"]
        self.class_trips = []
        if self.room_type == 'test':
            print('###################################################################')

        # class dimensions
        # width and height are relatively standard: ergo area is 2.3 * L
        self.room_size = self.input_params["room_size"] # sq m
        self.floor_area = self.input_params2["floor_area"]

        # functions
        self.relative_airflow = 'placeholder'

    def load_parameters(self, filepath):
        '''
        Loads seating from input directories
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

    def generate_class_seating(self):
        '''
        Based on:
        - class type
        - seating chart
        - number of students
        - number of adults


        Return dict of student_id: [x, y] for a good seating chart
        Updates:
        7/13 | assign using json file for small/large classrooms

        ToDo: Improve seating chart options and generation
        '''
        # evaluate temp based on # students
        num_kids = self.students_var
        temp_dict = {}
        for i in range(int(num_kids)):
            temp_dict[str(i)] = self.seat_dict[str(i)]
        return temp_dict

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

        Scatterplot:
        - Short vs Long range transmission rates (Y) over Distance (X)
        - Short vs Long range transmission rate Means (Y) over time (X)

        '''
        # 1 SETUP
        print('Model Setup...')

        # class_sim()

        # run class model
        class_seating = self.generate_class_seating()

        # seat var is room type
        # make varying input setup for ventilation
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

        class_trip, conc_array, out_mat, chance_nonzero = class_sim(n_students = int(self.students_var), mask = self.mask_var, n_sims = self.number_simulations, duration = self.duration, initial_seating = self.room_type, loc_params=temp_loc) # replace default with selected

        ### Validate using chance_nonzero
        '''
        Start here!


        '''

        self.chance_nonzero = chance_nonzero
        # print(chance_nonzero, 'more than none?')
        self.conc_array = conc_array
        self.class_trips.append(class_trip)
        # print('model_run start')
        plt.figure(figsize=(5,4))#, dpi=300)
        plt.gcf().set_size_inches(5,4)
        # plt.gcf().set_size_inches(5,4)
        # ax = plt.gca()
        pd.Series(class_trip).plot.kde(lw=2, c='r')
        plt.title('Density estimation of exposure')
        # plt.xlim(0, .004)
        # print(plt.xticks())

        # set x ticks
        temp_x = np.array(plt.xticks()[0])
        str_x = np.array([str(round(int * 100, 2))+'%' for int in temp_x])
        plt.xticks(temp_x, str_x)

        plt.ticklabel_format(axis="x")#, style="sci", scilimits=(0,0))

        plt.yticks(np.arange(0, 3500, 700), np.arange(0, 3500, 700) / 3500)

        ##### This is temporary chill tf out ####
        # rescale y axis to be % based
        plt.xlabel('Likelihood of exposure to infectious dose of particles                         ')
        plt.ylabel('Density estimation of probability of occurrence')
        plt.savefig('results/window_curve.png', dpi=300)
        # plt.show()
        print('model_run complete!')

        # temp variables
        self.chance_nonzero = 0
        self.conc_array = 0

        output_filepath = "output/class_simulation_" + str(self.students_var) + '_' + str(self.mask_var)+ '_' + str(self.number_simulations) + '_' + str(self.duration) + '_' + str(self.room_type) + '_' + str(self.window_var) # str(i) for i in [self.inputs]



        # 2 LOADING
        plt.figure()

        if self.room_type == 'small':
            self.seat_dict = load_parameters_av(filepath='config/small_classroom.json')
        elif self.room_type == 'large':
            self.seat_dict = load_parameters_av(filepath='config/large_classroom.json')
        # implement SEATING CHART OPTIONS ############################

        # 3 CONCENTRATION + 1

        print('Plotting Concentration ...')
        x_arr = []
        y_arr = []
        for i in self.seat_dict.items(): ################### change seating
            x_arr.append(i[1][1])
            y_arr.append(i[1][0] * 1.5 + 1) # seat fix
        rot = mpl.transforms.Affine2D().rotate_deg(180)

        # Set up Figure
        fig, ax1 = plt.subplots()
        plt.matshow(out_mat, cmap="OrRd", norm=mpl.colors.LogNorm())
        plt.gcf().set_size_inches(2,2)
        plt.suptitle('Viral Concentration Heatmap', fontsize=7.5)
        plt.axis('off')
        plt.text(.1, .01, '\nSample proxy for air flow after ' + str(self.duration) + ' minutes\n', fontsize=4)
        plt.savefig(output_filepath + '_concentration.png', dpi=300)
        plt.close()

        # 4 MASK HISTOGRAM

        fig_mask, ax_mask = plt.subplots()
        mask_values = [70, 80, 90, 100]
        mask_legend = [str(i) + '%' for i in mask_values]
        print('ml', mask_legend)

        for mask_ in mask_values:
            '''
            Note: Masks as referred to here are in terms of face masks

            coding 'masks' will be noted when used
            '''
            class_trip_mask, conc_array_mask_mask, out_mat_mask, chance_nonzero_mask = class_sim(n_students = int(self.students_var), mask = self.mask_var, n_sims = self.number_simulations, duration = self.duration, initial_seating = self.room_type, loc_params=temp_loc) # replace default with selected


            sns.distplot(list(class_trip_mask[2].values()), ax=ax_mask, rug=True, kde=False, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1})

        fig_mask.savefig(output_filepath + '_masks.png', dpi=300)

        # 5 AIRFLOW HISTOGRAM



        # 6 SCATTER
        # 7 RISK VS TIME



        '''
        TODO:
        All plots
        '''


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
