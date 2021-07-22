# imports
import argparse
from av import *

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

    :method as follows:
    # loop through n simulations ->
    # d days in each simulation ->
    # s steps in each day ->
    # i infectious students -> 
    # u uninfected students ->
    # tranmission_rate_by_step_for_student_pair
    --> mean number of infections

    '''
    targets = vars(args)

    # 1. load user input using argparse
    # print('args are:')
    # for key, value in targets.items():
    #     print(key, ':', value)

    # 3. generate model instance
    model_instance = user_viz(targets=targets)
    print('model instance generated')

    # 4. run model
    model_instance.model_run()

    print('Simulation complete! Please check /output folder for output.')

    return

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
    parser.add_argument('-r', '--room_type', type=str, default='small', help='Room Type: large or small')
    parser.add_argument('-ns', '--num_students', type=int, default=25, help='Number of Students in Class')
    parser.add_argument('-ni', '--num_initial', type=int, default=1, help='Number of Initial Infected')
    parser.add_argument('-na', '--num_adults', type=int, default=1, help='Number of Adults in Class')
    # make this 100
    parser.add_argument('-n', '--n_sims', type=int, default=1, help='Duration of Simulation')
    parser.add_argument('-a', '--age_group',            type=str, default='<15 years', help='Age Group of students in simulation')
    parser.add_argument('-mpc', '--mins_per_class',     type=int, default=60, help='Number of minutes per class')
    parser.add_argument('-cpd', '--classes_per_day',    type=int, default=3, help='Number of classes per day')
    parser.add_argument('-dps', '--days_per_simulation',type=int, default=5, help='Number of days per simulation')
    parser.add_argument('-wmr', '--well_mixed_room',    type=bool, default=False, help='True: Use Wells-Riley Well-Mixed Room Model; False: Include airflow proxy')
    parser.add_argument('-v', '--ventilation',          type=str, default='Open Windows', help='Ventilation Options; Closed Windows, Open Windows, Mechanical, Open Windows and Fans, Better Mechanical, Outdoors')

    # Human Arguments
    parser.add_argument('-qb', '--breathing_rate',      type=float, default=0.29, help='Breathing Rate') 
    parser.add_argument('-ra', '--respiratory_activity',type=str, default='speaking', help='Respiratory Activity')
    parser.add_argument('-sm', '--student_mask_percent',type=int, default=100, help='Student mask wearing %')
    parser.add_argument('-am', '--adult_mask_percent',  type=int, default=100, help='Adult mask wearing %')
    parser.add_argument('-pm', '--mask_protection_rate',type=float, default=.1, help='Mask Protection %')

    # Room Arguments
    parser.add_argument('-fa', '--floor_area',      type=int, default=900, help='Floor Area') #  area in ft^2
    parser.add_argument('-rh', '--room_height',     type=int, default=12.0, help='Height') # height in feet
    parser.add_argument('-vl', '--vent_locations',  type=list, default=[(50, 0)], help='optional vent location specifics') # x, y location
    parser.add_argument('-vs', '--vent_size',       type=int, default=10, help='Surface Area of the vents')
    parser.add_argument('-wl', '--window_locations', type=list, default=[(25, 0), (75, 0)], help='optional window location specifics') # x, y location
    parser.add_argument('-ws', '--window_size',     type=int, default=20, help='Surface area of the windows')
    parser.add_argument('-dl', '--door_locations',  type=list, default=[(20, 99)], help='optional door location specifics') # x, y location
    parser.add_argument('-ds', '--door_size',       type=int, default=80, help='Surface area of doors')
    parser.add_argument('-sc', '--seating_chart',   type=str, default='grid', help='Seating Chart')

    # Vent Arguments
    parser.add_argument('-ach', '--ach_level',      type=int, default=0, help='ACH level')
    parser.add_argument('-mr', '--merv_level',      type=int, default=0, help='MERV level')
    parser.add_argument('-qr', '--recirc_rate',     type=float, default=1, help='Recirculation Rate')
    parser.add_argument('-hum', '--relative_humidity',type=float, default=0.69, help='Relative Humidity')
    parser.add_argument('-paf', '--primary_outdoor_air_fraction', type=float, default=0.75, help='Zp Outdoor air fraction')

    # Advanced Arguments
    parser.add_argument('-st', '--strain',              type=str, default='B.1.427/429 (California)', help='Strain')
    parser.add_argument('-cdr', '--crit_droplet_radius',type=int, default=2, help='cutoff radius for more infectious particles: currently 2 microns')
    parser.add_argument('-vdr', '--viral_deact_rate',   type=float, default=0.3, help='Viral Deactivation Rate')
    parser.add_argument('-i', '--immunity_rate',        type=float, default=0, help='Vaccinated/Immune')
    parser.add_argument('-cvr', '--child_vax_rate',     type=float, default=0, help='Child Vaccination Rate')
    parser.add_argument('-avr', '--adult_vax_rate',     type=float, default=100, help='Adult Vaccination Rate')
    parser.add_argument('-vi', '--viral_infectivity',   type=float, default=30, help='Relation between mass of Virions and infection rate')
    parser.add_argument('-af', '--aerosol_filtration_eff', type=float, default=0.0, help='Aerosol filtration efficiency')

    args = parser.parse_args()
    print('arguments received')
    main(args)
