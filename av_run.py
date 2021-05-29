# imports
import argparse
import sys
sys.path.insert(0, 'src')
from av import *
import json

def load_everything(default_data, aerosol_data, args):
    room_type = args.room_type # string
    duration = args.duration # in minutes
    number_of_students = args.num_students # total occupancy
    mask = args.mask_wearing # in %
    windows = args.windows # string : see below
    indoors = args.indoors # 1 if yes, 0 if no
    vent_loc = args.vent_locations # list of tuples
    window_loc = args.window_locations # list of xy tuples
    room_size = args.room_type
    num_sims_ = 100 # default
    mask_eff_ = .1 # default
    air_exchange_rate = aerosol_data['air_exchange_rate']

    floor_areas_by_room = {"small": 1000, "large": 1500, "library": 3000, "gym": 5000} # square meters
    floor_area = floor_areas_by_room[args.room_type] * 10.7639 # Convert to square feet


    if windows == 'closed':
        air_exchange_rate = 2 # TESTING for windows closed
    elif windows== 'mechanical':
        air_exchange_rate = 4
    elif windows=='open':
        air_exchange_rate= 5
    elif windows=='outdoors' or indoors == 0:
        air_exchange_rate = 20
    else:
        print('inputs gone wrong, please advise')
        air_exchange_rate = 6

    room_type_arg = '-r' #type of classroom
    duration_arg = '-c'
    # seating_chart_type_arg = '-s'
    num_students_in_class = '-n'
    mask_wearing_arg = '-m' # likelihood
    windows_arg = '-w'

    # update aerosol to results
    aerosol_data["floor_area"] = floor_area
    aerosol_data["mask_passage_prob"] = mask_eff_
    aerosol_data["mean_ceiling_height"] = 6.07
    aerosol_data["air_exchange_rate"] = air_exchange_rate
    aerosol_data["primary_outdoor_air_fraction"] = 0.2
    aerosol_data["aerosol_filtration_eff"] = 0
    aerosol_data["relative_humidity"] = 0.69
    aerosol_data["breathing_flow_rate"] = 0.5
    aerosol_data["max_aerosol_radius"] = 2
    aerosol_data["exhaled_air_inf"] = 0
    aerosol_data["max_viral_deact_rate"] = 0.3

    # update default to results
    default_data["class_length"] = int(duration)
    default_data["number_students"] = int(number_of_students)
    default_data["mask_var"] = int(mask)
    default_data["window_var"] = windows
    default_data["number_simulations"] = int(num_sims_)
    aerosol_data["room_type"] = room_type
    aerosol_data["room_size"] = floor_areas_by_room[args.room_type]
    # default_data["seating_choice"] = seating_

    with open('results/default_data_.json', 'w') as f:
        json.dump(default_data, f)
    with open('results/aerosol_data_.json', 'w') as g:
        json.dump(aerosol_data, g)
    print('json loading done')

def main(args):
    '''
    args = airavata inputs
    '''
    with open('config/default.json') as f:
        default_data = json.load(f)
    with open('config/aerosol.json') as g:
        aerosol_data = json.load(g)

    load_everything(default_data, aerosol_data, args)
    print('user viz start')
    airavata_output = user_viz()
    # run all functions
    airavata_output.plot_class_seating() # swap this to new viz for ventilation

    airavata_output.model_run(args)

    print('finis')

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--room_type', required=True)
    parser.add_argument('-c', '--duration', required=True)
    parser.add_argument('-n', '--num_students', required=True)
    parser.add_argument('-m', '--mask_wearing', required=True)
    parser.add_argument('-w', '--windows', required=True)

    ### Add to AV
    parser.add_argument('-i', '--indoors', required=True) # is the activity indoors?

    # These are experimental- add sliders in future to allow for easy custom rooms
    parser.add_argument('-v', '--vent_locations') # x, y location
    parser.add_argument('-l', '--window_locations') # x, y location
    parser.add_argument('-s', '--room_size') # Length x Width in M

    args = parser.parse_args()
    main(args)
