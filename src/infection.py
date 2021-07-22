# imports
import math
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import json
import os

def generate_infectivity_curves():
    '''
    Citation: He et al. https://www.nature.com/articles/s41591-020-0869-5
    Temporal Dynamics in Viral Shedding

    Returns a list of useful variables:
   
    :return s_l_s_symptom_countdown: 
    :return x_symp_countdown:
    :return s_l_s_infectivity_density:
    :return x_infectivity:
    '''
    s_l_s_symptom_countdown = [0.6432659248014824, -0.07787673726582335, 4.2489459496009125]
    x_symptom_countdown = np.linspace(0, 17, 1000)

    s_l_s_infectivity_density = [20.16693271833812, -12.132674385322815, 0.6322296057082886]
    x_infectivity = np.linspace(-10, 8, 19)

    inf_params = {
        's_l_s_symptom_countdown': s_l_s_symptom_countdown, 
        'x_symptom_countdown': x_symptom_countdown, 
        's_l_s_infectivity_density': s_l_s_infectivity_density, 
        'x_infectivity': x_infectivity
    }
    return inf_params

def plot_infectivity_curves(inf_params, plot=True):
    '''
    plot our sampling process with 2 vertically stacked plots

    with titles and labels


    Density plot of # individuals who become infective that day
    - proxy for infectivity

    Lognorm plot of estimated days until symptoms appear

    '''
    if plot:
        fig, ax = plt.subplots(1)
        fig2, ax2 = plt.subplots(1)
    sls_symp_count = inf_params['s_l_s_symptom_countdown']
    x_symp_count = inf_params['x_symptom_countdown']
    s_l_s_infectivity_d = inf_params['s_l_s_infectivity_density']
    x_infectivity = inf_params['x_infectivity']
    l_shape, l_loc, l_scale = sls_symp_count
    g_shape, g_loc, g_scale = s_l_s_infectivity_d
    countdown_curve = stats.lognorm(s=l_shape, loc=l_loc, scale=l_scale)

    infective_df = pd.DataFrame({'x': list(x_infectivity), 'gamma': list(stats.gamma.pdf(x_infectivity, a=g_shape, loc=g_loc, scale=g_scale))})

    if plot:
        ax.plot(x_symp_count, countdown_curve.pdf(x_symp_count), 'k-', lw=2)
        ax.set_title('Expected days until symptoms appear')
        ax.set_xlabel('Number of days')
        ax.set_ylabel('Proportion of infected individuals')

                # Save just the portion _inside_ the first  axis's boundaries
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig('Days_to_Symp.png', dpi=300)

        ax2.plot(x_infectivity, infective_df.gamma)
        ax2.set_title('Expected infectivity before symptoms appear')
        ax2.set_xlabel('Number of days')
        ax2.set_ylabel('Initial Infectivity')

                # Save just the portion _inside_ the second  axis's boundaries
        extent2 = ax2.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
        # fig2.savefig('Infectivity_Dynamics.png', dpi=300)

        return infective_df
    else:
        return infective_df

def return_aerosol_transmission_rate(asol_args):
    '''
    
    
    :param floor_area: 
    :param room_height:
    :param air_exchange_rate:
    :param aerosol_filtration_eff:
    :param relative_humidity:
    :param breathing_flow_rate:
    :param exhaled_air_inf:
    :param max_viral_deact_rate:
    :param mask_passage_prob:
    :param max_aerosol_radius:
    :param primary_outdoor_air_fraction:

    :methods as follows:


    :return:
    '''

    floor_area = asol_args['floor_area']
    mean_ceiling_height = asol_args['room_height']
    air_exchange_rate = asol_args['air_exchange_rate']
    aerosol_filtration_eff = asol_args['aerosol_filtration_eff']
    relative_humidity = asol_args['relative_humidity']
    breathing_flow_rate = asol_args['breathing_flow_rate']
    exhaled_air_inf = asol_args['exhaled_air_inf']
    max_viral_deact_rate = asol_args['max_viral_deact_rate']
    mask_passage_prob = asol_args['mask_passage_prob']
    max_aerosol_radius = asol_args['max_aerosol_radius']
    primary_outdoor_air_fraction = asol_args['primary_outdoor_air_fraction']

    mean_ceiling_height_m = mean_ceiling_height * 0.3048 #m3
    room_vol = floor_area * mean_ceiling_height  # ft3
    room_vol_m = 0.0283168 * room_vol  # m3
    fresh_rate = room_vol * int(air_exchange_rate) / 60  # ft3/min
    recirc_rate = fresh_rate * (1/primary_outdoor_air_fraction - 1)  # ft3/min
    air_filt_rate = aerosol_filtration_eff * recirc_rate * 60 / room_vol  # /hr
    eff_aerosol_radius = ((0.4 / (1 - relative_humidity)) ** (1 / 3)) * max_aerosol_radius
    viral_deact_rate = max_viral_deact_rate * relative_humidity
    sett_speed = 3 * (eff_aerosol_radius / 5) ** 2  # mm/s
    sett_speed = sett_speed * 60 * 60 / 1000  # m/hr
    conc_relax_rate = int(air_exchange_rate) + air_filt_rate + viral_deact_rate + sett_speed / mean_ceiling_height_m  # /hr
    airb_trans_rate = ((breathing_flow_rate * float(mask_passage_prob)) ** 2) * exhaled_air_inf / (room_vol_m * conc_relax_rate)

    return airb_trans_rate #This is mean number of transmissions per hour between pair of infected / healthy individuals

def fast_root(num, n):
    '''
    helper function to get nth root of a number

    :param num: number to get root of
    :param n: root to get

    :return: nth root of num
    '''
    return (num ** (1 / n))

def minute_transmission(inf_id, other_id, base_srt, airflow_proxy, wmr, seats, mask_var, floor_area, initial_infect, distance):
    '''
    Primary Infection Function at each step

    :param inf_id:          ID of infected individual
    :param other_id:        student id of uninfected individual
    :param base_srt:        Well Mixed Room shared room transmission
    :param airflow_proxy:   Optional airflow proxy for uneven concentration of viral particles
    :param wmr:             Well Mixed Room ?
    :param seats:           dict of studentid: (x, y) coordinates of each student
    :param mask_var:        Likelihood of student wearing mask

    :var t_rate_by_hour: average # of infections per hour per infectious per susceptible
    :var close_range_transmission:    transmission calculations for 'ballistic droplets' >= 100 microns
    :var shared_room_transmission:    transmission calculations for 'aerosol droplets' < 100 microns
    :var long_range_transmission:     transmission calculations for 'ballistic droplets >= 300 microns

    :method as follows: 
    crt = infectivity % * chu_proxy
    srt = (mask pen ^ 2 * breathing rate * quanta emission rate) / 
    lrt = 


    t_rate_by_hour = crt + srt + lrt  
    t_rate_by_minute = (1 - (1 - 60thRoot(t_rate_by_hour))) 

    :return infection_occurs:   other_id is exposed to an infectious dose at this step
    :return caused_by:          string of cause of infection
    '''
    # MASKS

    # WMR  # function to find distance between two points


    infect_baseline = initial_infect # get BaseStudent with id = inf_id: return 

    cause = '' # string to hold cause of infection
    infection_occurs = False # set to True if infected


    # close_range = chu proxy (eventually chen proxy)
    chu_proxy = 1/(2.02 ** distance)
    crt = chu_proxy * infect_baseline
    crt_minute = (1 - fast_root(1-crt, 60))
    crt_infect = np.random.choice([False, True], p=[1 - crt_minute, crt_minute])

    # shared_room = bazant and bush aerosol t rate + vent proxy
    if airflow_proxy == 0:
        srt = base_srt
    # else:
    #     srt = base_srt * airflow_proxy[][] # get other_id's airflow_proxy

    srt_minute = (1 - fast_root(1-srt, 60))
    srt_infect = np.random.choice([False, True], p=[1 - srt_minute, srt_minute])

    # long_range = chu proxy (eventually chen proxy)
    lrt = chu_proxy * infect_baseline * .1 # scaling factor to account for larger particle radius
    lrt_minute = (1 - fast_root(1-lrt, 60))
    lrt_infect = np.random.choice([False, True], p=[1 - lrt_minute, lrt_minute])

    # total transmission likelihood: 
    t_rate_by_hour = crt + srt + lrt
    t_rate_by_minute = crt_minute + srt_minute + lrt_minute

    ###### Experiment with order of if statements
    if crt_infect:
        cause = 'close range transmission'
        infection_occurs = True
    elif infection_occurs == False and srt_infect:
        cause = 'shared room transmission'
        infection_occurs = True
    elif infection_occurs == False and lrt_infect:
        cause = 'long range transmission'
        infection_occurs = True
    else:
        cause = 'no transmission'

    t_rates = {
        'crt_minute': crt_minute, 
        'srt_minute': srt_minute, 
        'lrt_minute': lrt_minute, 
        't_rate_by_hour': t_rate_by_hour,
        't_rate_by_minute': t_rate_by_minute
    }
    return infection_occurs, cause, t_rates