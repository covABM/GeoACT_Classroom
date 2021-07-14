# imports
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
    s_l_s_symptom_countdown   | Symptom countdown: days until symptoms are expected
    x_symptom_countdown       | X range for ^

    s_l_s_infectivity_density |
    x_infectivity             |

    TODO:
    chu_distance              | replaced by self.chu_proxy_vars

    '''
    s_l_s_symptom_countdown = [0.6432659248014824, -0.07787673726582335, 4.2489459496009125]
    x_symptom_countdown = np.linspace(0, 17, 1000)

    s_l_s_infectivity_density = [20.16693271833812, -12.132674385322815, 0.6322296057082886]
    x_infectivity = np.linspace(-10, 8, 19)

    return [s_l_s_symptom_countdown, x_symptom_countdown, s_l_s_infectivity_density, x_infectivity, chu_distance]

def plot_infectivity_curves(in_array, plot=True):
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
    sls_symp_count, x_symp_count, s_l_s_infectivity_density, x_infectivity, distance_multiplier = in_array
    l_shape, l_loc, l_scale = sls_symp_count
    g_shape, g_loc, g_scale = s_l_s_infectivity_density
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
        fig2.savefig('Infectivity_Dynamics.png', dpi=300)

        return infective_df
    else:
        return infective_df

def return_aerosol_transmission_rate(floor_area, room_height, air_exchange_rate, aerosol_filtration_eff, relative_humidity, breathing_flow_rate, exhaled_air_inf, max_viral_deact_rate, mask_passage_prob, max_aerosol_radius, primary_outdoor_air_fraction):
    '''
    Room
    floor_area, 
    room_height, 

    Human


    Simulation

    Advanced


    air_exchange_rate, 
    aerosol_filtration_eff,   
    relative_humidity, 

    breathing_flow_rate, 
    exhaled_air_inf, 
    max_viral_deact_rate, 
    mask_passage_prob, 
    max_aerosol_radius, 
    primary_outdoor_air_fraction
    '''

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
