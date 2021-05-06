# imports
import argparse
import sys
sys.path.insert(0, 'src')
from av import *
import json


    #
    # for item in data['maps']:
    #     item['iscategorical'] = item['iscategorical'].replace('$home', item['id'])
    #
    # with open('new_data.json', 'w') as f:
    #     json.dump(data, f)
    # return
# def replacer(data, new_val):
#     for item in data:
#         data[item] = data[item].replace("test", new_val)
#
#     return data

def main(targets):
    '''
    targets = airavata inputs
    '''
    with open('config/default.json') as f:
        default_data = json.load(f)
    with open('config/aerosol.json') as g:
        aerosol_data = json.load(g)
    print(targets)

    # for i in sys.argv:
    # print('t', targets, '0', args.0])
    # number each of these:
    floor_area = float(args.class_type) * 2.3 * 3.28084 # square feet
    trip_duration = args.trip_duration
    number_of_students = args.num_students
    mask = args.mask_wearing
    windows = args.windows
    num_sims_ = 100 # default
    mask_eff_ = .1 # default
    seating_ = args.seating_type
    air_exchange_rate = 20 # default for a class

    class_type_arg = '-l' #length of class
    trip_duration_arg = '-t'
    seating_chart_type_arg = '-s'
    num_students_on_class = '-n'
    mask_wearing_arg = '-m' # likelihood
    windows_arg = '-w'





        # ask kaushik about how inputs work
        # replace values with airavata inputs (command line?)

        # if this causes an error, change logic to alter default json file
        # and/or create a new file w/ their parameter selection

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
    default_data["trip_length"] = int(trip_duration)
    default_data["number_students"] = int(number_of_students)
    default_data["mask_var"] = int(mask)
    default_data["window_var"] = int(windows)
    default_data["number_simulations"] = int(num_sims_)
    default_data["seating_choice"] = seating_

    with open('results/default_data_.json', 'w') as f:
        json.dump(default_data, f)
    with open('results/aerosol_data_.json', 'w') as g:
        json.dump(aerosol_data, g)

    airavata_output = user_viz()
    # run all functions
    airavata_output.plot_class_seating()

    airavata_output.conc_heat()

    airavata_output.model_run()

    print('finis')

    return

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--class_type', required=True)
    parser.add_argument('-t', '--trip_duration', required=True)
    parser.add_argument('-s', '--seating_type', required=True)
    parser.add_argument('-n', '--num_students', required=True)
    parser.add_argument('-m', '--mask_wearing', required=True)
    parser.add_argument('-w', '--windows', required=True)
    args = parser.parse_args()
    print(args)


    targets = args
    # parser = argparse.ArgumentParser()
    # targets = parser.parse_args()
    # print(targets, 'targe')
    main(targets)
