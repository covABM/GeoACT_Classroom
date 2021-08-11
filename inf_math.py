# This file contains all functions necessary for using the test notebook

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import panel as pn
import param 

class Simulation(param.Parameterized):
    """
    Base class for simulation parameters:



    """
    # sim arguments
    room_type = param.ObjectSelector(default='Classroom',objects=['small','large'])
    num_students = param.Integer(default=10,bounds=(0,None))
    num_teachers = param.Integer(default=1,bounds=(0,None))
    num_initial_infections = param.Integer(default=1,bounds=(0,None))

    # human arguments

    # room arguments

    # vent arguments

    # advanced args
    



    floor_area = param.Number(default=1000, bounds=(0,None), doc="Floor area of the building")
    room_height = param.Number(default=3, bounds=(0,None), doc="Height of the room")
    air_change_per_hour = param.Number(default=0.7, bounds=(0,None), doc="Air changes per hour")
    breathing_flow_rate = param.Number(default=0.5, bounds=(0,None), doc="Agent breathing rate")
    aerosol_filtration_eff = param.Number(default=0.5, bounds=(0,None), doc="Aerosol filtration efficiency")


