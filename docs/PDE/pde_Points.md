# pde_Points

Functions for generating points for training model.

While available to user, not meant to be used. Instead interface through
N_pde, N_iv, N_sensors, sensor_range, and N_bc parameters of solving and boundary 
functions detailed in "Main User Functions".

::: src.pinnDE.PDE.pde_Points
    options:
        members_order: source
    rendering:
      show_root_heading: yes
    selection:
      members:
        - defineCollocationPoints_tx
        - defineCollocationPoints_xy
        - defineCollocationPoints_DON_tx
        - defineCollocationPoints_DON_xy