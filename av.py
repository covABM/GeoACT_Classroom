# import
import json
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
sys.path.insert(0, 'src')
from classroom import class_sim
from infection import return_aerosol_transmission_rate

def load_parameters_av(filepath):
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



class user_viz():
    def __init__(self, parent=None):
        super(user_viz, self).__init__()
        # separate input params
        self.input_params = load_parameters_av(filepath='results/default_data_.json')
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
        if self.room_type == 'small':
            self.seat_dict = load_parameters_av(filepath='config/small_classroom.json')
        elif self.room_type == 'large':
            self.seat_dict = load_parameters_av(filepath='config/large_classroom.json')
        elif self.room_type in ['library', 'gym', 'sports practice', 'band room', 'theatre']:
            self.seat_dict = load_parameters_av(filepath='config/small_classroom.json')
            print('These activities are still being implemented- please reach out if you need them expedited')
        else:
            print('Error! Problem loading seating')
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
        based on full vs zigzag vs edge
        based on number of students
        '''

        # evaluate temp based on # students
        num_kids = self.students_var
        temp_dict = {}
        for i in range(int(num_kids)):
            temp_dict[str(i)] = self.seat_dict[str(i)]
        # print(temp)
        # print(temp_dict)
        return temp_dict

    def plot_class_seating(self):
        '''
        plot avg based on temp dict

        TODO: background of class
        '''
        t_dict = self.generate_class_seating()
        x_arr = []
        y_arr = []
        # print('class_seat_figure')
        plt.figure(figsize=(2,2))
        plt.gcf().set_size_inches(2,2)
        # plt.gcf().set_size_inches(2,2)
        for i in t_dict.items():
            x_arr.append(i[1][1])
            y_arr.append(i[1][0])
        # im = plt.imread('results/class_img.png')
        # plt.imshow(im)
        plt.title('Approximate Seating Chart', fontsize=7)
        plt.xticks(np.array([1.2, 1.8, 2.2, 3.8, 4.2, 4.8]))
        plt.yticks(np.arange(-.5, 23.5, 1))
        plt.grid(True)
        # plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')
        plt.scatter(x=x_arr, y=y_arr)#, marker='_')
        plt.xticks(c='w')
        plt.yticks(c='w')
        # plt.axis('off') # set axis to be blank
        # plt.show()

        plt.savefig('results/seating_plot.png', dpi=300)
        print('plot seating complete!')

        return

    def conc_heat(self, class_seating, class_trip, conc_array, out_mat, chance_nonzero):
        '''
        average over model runs: out_matrix averages
        '''
        rot = mpl.transforms.Affine2D().rotate_deg(180)
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.matshow(out_mat, cmap="OrRd", norm=mpl.colors.LogNorm())
        plt.arrow(-2,24,0,-26, head_width=0.2, head_length=0.2, fc='k', ec='k')
        plt.gcf().set_size_inches(2,2)
        # plt.suptitle('Relative Airflow Heatmap', fontsize=7.5)
        plt.annotate(xy=(-1, -1), text='front', fontsize=5)
        plt.annotate(xy=(-1, 24), text='back', fontsize=5)
        plt.axis('off')


        ax2 = fig.add_subplot(1,2,2)
        ax2.matshow(out_mat, cmap="OrRd")#, norm=mpl.colors.LogNorm())
        plt.arrow(-2,24,0,-26, head_width=0.2, head_length=0.2, fc='k', ec='k')
        plt.gcf().set_size_inches(2,2)
        plt.suptitle('Relative Airflow Heatmap', fontsize=7.5)
        plt.annotate(xy=(-1, -1), text='front', fontsize=5)
        plt.annotate(xy=(-1, 24), text='back', fontsize=5)
        # log scale vs regular scale + 'be not afraid'
        plt.axis('off')
        fig.text(.1, .01, 'These heatmaps show relative airflow within the cabin \nof the class in terms of its effect on concentration \nof COVID-19 Particles (averaged across 100 simulations)', fontsize=4)
        plt.savefig('results/relative_airflow.png', dpi=300)

        print('relative airflow complete!')
        return
    # function to run model with user input
    def model_run(self):
        '''
        Updated 7/7/21 with _bus code

        1 SETUP
        2 LOADING
        3 CONCENTRATION + 2
        4 HISTOGRAMS + 3
        5 SCATTER/KDE + 2
        6 T_RISK AVERAGE + 1

        '''
        # 1 SETUP
        print('Model Running...')

        # class_sim()

        # run class model
        class_seating = self.generate_class_seating()
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

        class_trip, conc_array, out_mat, chance_nonzero = class_sim(int(self.students_var), self.mask_var, self.number_simulations, self.duration, self.room_type, loc_params=temp_loc) # replace default with selected

        print('HUH', chance_nonzero)

        self.conc_heat(class_trip, conc_array, out_mat, chance_nonzero)
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

        ##### what the fuck do i do with this????####
        # rescale y axis to be % based
        plt.xlabel('Likelihood of exposure to infectious dose of particles                         ')
        plt.ylabel('Density estimation of probability of occurrence')
        plt.savefig('results/window_curve.png', dpi=300)
        # plt.show()
        print('model_run complete!')

        # temp variables
        self.chance_nonzero = 0
        self.conc_array = 0
        print('Seating...')
        fig1, ax1 = plt.subplots()
        seat_types = ['full', 'window', 'zigzag']
        for s in seat_types:
            bus_out_array, conc_array, out_mat, chance_nonzero, avg_mat = class_sim(int(self.students_var), self.mask_var, self.number_simulations, self.duration, s, self.window_var) # SEATING
            pd.Series(bus_out_array[2]).plot.hist(bins=np.arange(0, 0.12, 0.001), alpha=.5, ax=ax1)





        # 2 LOADING
        plt.figure()

        output_filepath = "output/class_simulation"
        # seating_chart = self.seat_var
        #
        # # seating chart ################################# TODO
        # if seating_chart == "grid":
        #     seat_dict = self.load_parameters()
        # elif seating_chart == "circles":
        #     seat_dict = self.load_parameters()
        # else:
        #     seat_dict = self.load_parameters()
        #     print("ERROR")
        # implement SEATING CHART OPTIONS ############################

        # 3 CONCENTRATION + 2

        print('Plot Concentration')
        x_arr = []
        y_arr = []
        for i in self.seat_dict.items(): ################### change seating
            x_arr.append(i[1][1])
            y_arr.append(i[1][0] * 1.5 + 1) # seat fix
        rot = mpl.transforms.Affine2D().rotate_deg(180)

        # Set up Figure
        fig, ax1 = plt.subplots()
        plt.matshow(out_mat, cmap="OrRd", norm=mpl.colors.LogNorm())
        plt.arrow(-2,24,0,-26, head_width=0.2, head_length=0.2, fc='k', ec='k')
        plt.scatter(x=x_arr, y=y_arr, s=5, c='k')
        plt.gcf().set_size_inches(2,2)
        plt.annotate(xy=(-1, -1), text='front', fontsize=5)
        plt.annotate(xy=(-1, 24), text='back', fontsize=5)
        plt.axis('off')

        ax2 = fig.add_subplot(1,2,2)
        ax2.matshow(out_mat, cmap="OrRd")#, norm=mpl.colors.LogNorm())
        plt.arrow(-2,24,0,-26, head_width=0.2, head_length=0.2, fc='k', ec='k')
        plt.scatter(x=x_arr, y=y_arr, s=5)
        plt.gcf().set_size_inches(2,2)
        plt.suptitle('Viral Concentration Heatmap', fontsize=7.5)
        plt.annotate(xy=(-1, -1), text='front', fontsize=5)
        plt.annotate(xy=(-1, 24), text='back', fontsize=5)
        plt.axis('off')
        fig.text(.1, .01, 'Concentration of viral particles in the\nLeft heatmap is Log Normalized for visibility', fontsize=4)
        plt.savefig(output_filepath + 'concentration.png', dpi=300)
        plt.close()


        # 4 HISTOGRAMS + 3

        # HISTOGRAMS
        # Hist 1 Seating

        plt.legend(['Full Occupancy Seating', 'Window Seats Only', 'Zigzag Seating'])
        plt.xlabel('Mean likelihood of transmission at each step')
        plt.ylabel('Number of students with this average risk of transmission')
        seat_filepath = output_filepath + '_seating.png'
        fig1.savefig(seat_filepath, dpi=300)
        print('seating complete')
        plt.close(fig1)

        # Hist 2 Windows
        '''
        TODO:
        Dynamic X minmax
        '''


        print('Windows...')
        fig2, ax2 = plt.subplots()
        window_types = [0, 6]
        win_out_df = pd.DataFrame(columns=window_types)
        temp = 0.051
        temp_step = 0.001

        # add dynamic x range ToDo

        for w in window_types:
            bus_out_array, conc_array, out_mat, chance_nonzero, avg_mat = class_sim(int(self.students_var), self.mask_var, self.number_simulations, self.duration, self.seat_var, w) # WINDOW
            x_range = [.051, .102, .153, .204]

            ## 7/4 TODO: why is it all going wrong


            for i in range(len(x_range)):
                if x_range[i] < max(bus_out_array[2].values()):
                    pass
                else:
                    temp = x_range[i]
                    temp_step = 0.001 * (i + 1)


            # TODO: Check all values of KDE are positive

            # pd.Series(bus_out_array[2]).plot.kde(alpha=.5, ax=ax2)
            # pd.Series(bus_out_array[2]).plot.hist(alpha=.5, ax=ax2)

            ###############################################

            # SEABORN
            sns.distplot(list(bus_out_array[2].values()), ax=ax2, rug=True, kde=False, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1})


        fig2.legend(['Windows Closed', 'Windows Open 6 Inches'])
        plt.xlabel('Mean likelihood of transmission at each step')
        plt.ylabel('Number of students with this average risk of transmission')
        seat_filepath_2 = output_filepath + '_windows.png'
        fig2.savefig(seat_filepath_2, dpi=300)
        plt.close(fig2)
        print('Windows complete!')

        # Hist 3 Masks

        print('Masks...')
        fig3 = plt.figure(3)
        mask_amount = [1, .9, .8, .7]
        print('start masks')
        colorlist = ['blue', 'green', 'yellow', 'red']
        count_ = 0

        for m in mask_amount:
            bus_out_array, conc_array, out_mat, chance_nonzero, avg_mat = class_sim(int(self.students_var), m, self.number_simulations, self.duration, self.seat_var, self.window_var) # SEATING
            pd.Series(bus_out_array[2]).plot.hist(bins=np.arange(0, 0.056, 0.001), alpha=.5, color=colorlist[count_])
            count_ += 1
        plt.legend(['100% Mask compliance', '90% Mask compliance', '80% Mask compliance', '70% Mask compliance'])
        plt.xlabel('Mean likelihood of transmission at each step')
        plt.ylabel('Number of students with this average risk of transmission')
        seat_filepath_3 = output_filepath + '_masks.png'
        fig3.savefig(seat_filepath_3, dpi=300)
        plt.close(fig3)
        print('Masks complete!')



        # 5 SCATTER/KDE + 2




        # 6 T_RISK AVERAGE + 1
