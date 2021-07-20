# imports
import argparse
import sys
if 'src' not in sys.path:
    sys.path.insert(0, 'src')
from av import *
import json

def main(args):
    '''
   7/18 |  Reformat to use param better

    1. load user input

    2. create local instance of user_viz class with user input and defaults
    user_viz has methods:
    load_from_json()
    generate_seating()
    model_run()
    # model_run() is the the primary method that runs the simulation !
    model_run() calls class_sim() several times:
    # a. baseline w/ user input
    # b. baseline vs other mask options
    # c. baseline vs other vent and window options

    3. call user_viz.model_run()
    '''

    # 1. load user input using argparse
    print('args are:')
    print(args)

    # 3. generate model instance
    user_viz = user_viz(args)

    # 4. run model
    user_viz.model_run()












    # with open('config/default.json') as f:
    #     default_data = json.load(f)
    # with open('config/aerosol.json') as g:
    #     aerosol_data = json.load(g)



    

    # floor_area = max(aerosol_data['floor_area'], 1000) # THIS IS DEFAULT WHAT SHOULD THE ROOM SIZE BE
    # room_type = args.room_type # string
    # duration = args.duration # in minutes
    # number_of_students = args.num_students # total occupancy
    # mask = args.mask_wearing # in %
    # windows = args.windows # string : see below
    # indoors = args.indoors # 1 if yes, 0 if no
    # vent_loc = args.vent_locations # list of tuples
    # window_loc = args.window_locations # list of xy tuples
    # num_sims_ = 100 # default
    # mask_eff_ = .1 # default
    # air_exchange_rate = aerosol_data['air_exchange_rate']

   





    # ########## MAKE THE BELOW INTO USER INPUTS


    # # update aerosol to results
    # aerosol_data["floor_area"] = floor_area
    # aerosol_data["mask_passage_prob"] = mask_eff_
    # aerosol_data["mean_ceiling_height"] = 6.07
    # aerosol_data["air_exchange_rate"] = air_exchange_rate
    # aerosol_data["primary_outdoor_air_fraction"] = 0.2
    # aerosol_data["aerosol_filtration_eff"] = 0
    # aerosol_data["relative_humidity"] = 0.69
    # aerosol_data["breathing_flow_rate"] = 0.5
    # aerosol_data["max_aerosol_radius"] = 2
    # aerosol_data["exhaled_air_inf"] = 0
    # aerosol_data["max_viral_deact_rate"] = 0.3

    # # update default to results
    # default_data["class_length"] = int(duration)
    # default_data["duration"] = int(duration)
    # default_data["number_students"] = int(number_of_students)
    # default_data["mask_var"] = int(mask)
    # default_data["window_var"] = windows
    # default_data["number_simulations"] = int(num_sims_)
    # aerosol_data["room_type"] = room_type
    # aerosol_data["room_size"] = floor_area
    # # default_data["seating_choice"] = seating_

    # with open('results/default_data_.json', 'w') as f:
    #     json.dump(default_data, f)
    # with open('results/aerosol_data_.json', 'w') as g:
    #     json.dump(aerosol_data, g)
    # print('json loading done')



    # model_run_args = []


    av_out = user_viz(targets = args)
    
    print('Simulation complete! Please check /output folder for output.')

    return



# def main_2(args_):
#     '''
#     Run functions with new json inputs
#     '''

#     av_out = user_viz()

#     av_out.model_run(args_)

#     return

if __name__ == '__main__':
    '''
    Input Parameters:
        Room:
    A       Floor area
    H       Height
    Q       Outdoor Ach
            MERV level
    Qr      Recirculation rate
            Humidity
            Breathing Rate

        Human:
    Qb      Breathing Rate 
            Respiratory Activity
    M%      Mask wearing %
    Ma%     Adult Mask %
    Pm      Mask Protection

        Simulations:
            Room Type   # Sets Default values for Room Parameters
            Age group
            Duration: Mins/Event
            Events/day
            Days/simulation
            Num Students
            Num Adults
            Well Mixed Room ?

        Advanced:
            Strain
            Aerosol size cutoff
            viral deactivation rate
            Chu / Chen proxy
            Vent distribution proxy
            Vaccinated/Immune
            Viral Infectivity
            Units
            Expected Symptom Dates



    '''
    parser = argparse.ArgumentParser()

    # Simulation Arguments
    # setup
    parser.add_argument('-r', '--room_type', type=str, default='small', help='Room Type: large or small')
    parser.add_argument('-ns', '--num_students', type=int, default=25, help='Number of Students in Class')
    parser.add_argument('-na', '--num_adults', type=int, default=1, help='Number of Adults in Class')
    parser.add_argument('-a', '--age_group', type=str, default='<15 years', help='Age Group of students in simulation')
    # duration
    parser.add_argument('-mpc', '--mins_per_class', type=int, default=60, help='Number of minutes per class')
    parser.add_argument('-cpd', '--classes_per_day', type=int, default=3, help='Number of classes per day')
    parser.add_argument('-dps', '--days_per_simulation', type=int, default=5, help='Number of days per simulation')
    parser.add_argument('-wmr', '--well_mixed_room', type=bool, default=False, help='True: Use Wells-Riley Well-Mixed Room Model; False: Include airflow proxy')


    # Human Arguments
    parser.add_argument('-qb', '--breathing_rate', type=int, default=0.29, help='Breathing Rate') 
    parser.add_argument('-ra', '--respiratory_activity', type=str, default='speaking', help='Respiratory Activity')
    parser.add_argument('-sm', '--student_mask_percent', type=int, default=100, help='Student mask wearing %')
    parser.add_argument('-am', '--adult_mask_percent', type=int, default=100, help='Adult mask wearing %')
    parser.add_argument('-pm', '--mask_protection_rate', type=int, default=.1, help='Mask Protection %')

    # Room Arguments
    parser.add_argument('-fa', '--floor_area', type=int, default=1000, help='Floor Area') #  area in ft^2
    parser.add_argument('-rh', '--room_height', type=int, default=12.0, help='Height') # height in feet
    parser.add_argument('-vl', '--vent_locations', type=tuple, help='optional vent location specifics') # x, y location
    parser.add_argument('-vs', '--vent_size', type=int, default=10, help='Size of the room')
    parser.add_argument('-wl', '--window_locations', type=tuple, help='optional window location specifics') # x, y location
    parser.add_argument('-ws', '--window_size', type=int, default=10, help='Size of the room')
    parser.add_argument('-dl', '--door_locations', type=tuple, help='optional door location specifics') # x, y location
    parser.add_argument('-ds', '--door_size', type=int, default=10, help='Surface area of doors')
    parser.add_argument('-sc', '--seating_chart', type=str, default='grid', help='Seating Chart')

    # Vent Arguments
    parser.add_argument('-ach', '--ach_level', type=int, default=0, help='ACH level')
    parser.add_argument('-mr', '--merv_level', type=int, default=0, help='MERV level')
    parser.add_argument('-qr', '--recirc_rate', type=int, default=0, help='Recirculation Rate')
    parser.add_argument('-h', '--relative_humidity', type=int, default=0, help='Relative Humidity')


    # Advanced Arguments
    
    parser.add_argument('-st', '--strain', type=str, default='B.1.427/429 (California)', help='Strain')
    parser.add_argument('-cdr', '--crit_droplet_radius', type=int, default=2, help='cutoff radius for more infectious particles: currently 2 microns')
    parser.add_argument('-vdr', '--viral_deact_rate', type=int, default=0, help='Viral Deactivation Rate')
    parser.add_argument('-i', '--immunity_rate', type=int, default=0, help='Vaccinated/Immune')
    parser.add_argument('-cvr', '--child_vax_rate', type=int, default=0, help='Child Vaccination Rate')
    parser.add_argument('-avr', '--adult_vax_rate', type=int, default=100, help='Adult Vaccination Rate')

    args = parser.parse_args()

    main(args)
