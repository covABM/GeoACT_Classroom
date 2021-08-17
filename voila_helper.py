# imports
import ipywidgets as wg
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd

# Define the sample function
def sample_initial(n_initial=3, plot=False):
    """
    Takes n initial samples of the 
    - serial interval,
    - infectious_after_symp,
    - and symp_after_infected

    :Source: https://www.nature.com/articles/s41591-020-0869-5
    """
    # serial interval spline curve


    # countdown interval spline curve
    s_l_s_symptom_countdown = [0.6432659248014824, -0.07787673726582335, 4.2489459496009125]
    x_symptom_countdown = np.linspace(0, 17, 1000)

    # infectivity based on number of days after symptom onset spline curve
    s_l_s_infectivity_density = [20.16693271833812, -12.132674385322815, 0.6322296057082886]
    x_infectivity = np.linspace(-10, 8, 19)

    
    l_shape, l_loc, l_scale = s_l_s_symptom_countdown
    g_shape, g_loc, g_scale = s_l_s_infectivity_density
    countdown_curve = stats.lognorm(s=l_shape, loc=l_loc, scale=l_scale)
    infective_curve = stats.gamma.pdf(x_infectivity, a=g_shape, loc=g_loc, scale=g_scale)

    # Generate 1 for each initial infected agent
    i_a_s = [int(np.round(i)) for i in stats.lognorm.rvs(l_shape, l_loc, l_scale, size=n_initial)]
    # density function: % chance of becoming infectious after showing symptoms
    # infectious_after_symp.append(i_a_s)
    
    infective_df = pd.DataFrame({'x': list(x_infectivity), 'gamma': list(stats.gamma.pdf(x_infectivity, a=g_shape, loc=g_loc, scale=g_scale))})


    # Return N RANDOM NUMBERS HOW HARD IS THAT i'm so tired



    # s_a_i = [int(np.round(j)) for j in stats.gamma.rvs(a=g_shape, loc=g_loc, scale=g_scale, size=n_initial)]
    # density function: % chance of developing symptoms after being infected
    # symp_after_infected.append(s_a_i)

    # Define the plot logic
    # def plot_spare(countdown_curve, infective_df, x_symptom_countdown, x_infectivity):
    #     fig, ax=plt.subplots()
    #     fig2, ax2=plt.subplots()
    #     ax.plot(x_symptom_countdown, countdown_curve.pdf(x_symptom_countdown), 'k-', lw=2)
    #     ax.set_title('Expected days until symptoms appear')
    #     ax.set_xlabel('Number of days')
    #     ax.set_ylabel('Proportion of infected individuals')

    #             # Save just the portion _inside_ the first  axis's boundaries
    #     extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    #     # fig.savefig('Days_to_Symp.png', dpi=300)

    #     ax2.plot(x_infectivity, infective_df.gamma)
    #     ax2.set_title('Expected infectivity before symptoms appear')
    #     ax2.set_xlabel('Number of days')
    #     ax2.set_ylabel('Initial Infectivity')



    # return the sampled variables
    return x_symptom_countdown, countdown_curve, x_infectivity, infective_df#initial_i_a_s #serial_intervals, infectious_after_symp, symp_after_infected 





    
# out = wg.Output(width='70%')
# def on_value_change(change):
#     num_init = num_init_slider.value
#     asymp_rate = asymp_slider.value

#     with out:
#         fig, ax = plt.subplots()

        # the histogram of the data
        # x = mu + sigma * np.random.randn(437)
        # n, bins, patches = ax.hist(x, num_bins, density=1)

        # # add a 'best fit' line
        # y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
        # ax.plot(bins, y, '--')
        
        # ax.set_xlabel('X')
        # ax.set_ylabel('Probability density')
        # ax.set_title(f'Histogram with: mu={mu}, sigma={sigma}, bins={num_bins}')

        # clear_output(wait=True)
        # plt.show(fig)

# num_init_slider.observe(on_value_change, names="value")
# asymp_slider.observe(on_value_change, names="value")
# on_value_change(None)
# input_rows = [num_init_slider, asymp_slider, init_sample_button]