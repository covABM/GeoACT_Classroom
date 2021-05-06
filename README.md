OneRoomABM: A scale model for simulating transmission for an Agent-Based modeling system.
By scaling down the parameters to a single room of students, effects of elements such as distance and ventilation can be seen more easily. This also provides a testing environment that has a significantly faster runtime and has been useful in verifying our math for transmission both through droplets and aerosols

Installation instructions:
---

clone repository (use git bash or Github Desktop)
ensure package installations

run python -m verify to test that repository is downloaded fully and correctly


Usage instructions:

Open GUI: python -m run gui

Select Parameter inputs:

Model Type: [Bus, Classroom]

Number of Students: [Depends on Model type]

Percent wearing masks: [100%, 90%, 80%]

Windows Open: inches

Bus Route: dropdown of potential Bus Routes (and thus the bus size/type)

Seating Chart Type:


[Full, zigag, windows_only]

Concentration_average = Proxy for calculating uneven viral spread within the room/bus

Seating Chart Viz = hover

Model Run?

Previous Runs


Advanced Parameters



TODO:

Prettify: Heatmaps, plots / time
Methods according to SEIR-P
Verify.py: make sure download went correctly
