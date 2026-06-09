from time import perf_counter
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import torch
from torch.profiler import profile, ProfilerActivity, record_function

def plot_tire_characteristics(dynamics_model, compile = False):
    tire_model = dynamics_model.tire_model

    SAN, SRN = 100, 100
    #slip_angles = torch.linspace(-np.pi, np.pi, 100)  # radians
    #slip_angles = torch.linspace(-1.5, 1.5, 100)  # radians
    #slip_angles = torch.linspace(-1.7, 1.7, 100)  # radians
    #slip_angles = torch.linspace(-1.0, 1.0, SAN)  # radians
    #slip_ratios = torch.linspace(-1.0, 1.0, SRN)  # unitless
    slip_angles = torch.linspace(-0.2, 0.2, SAN)  # radians
    slip_ratios = torch.linspace(-0.2, 0.2, SRN)  # unitless
    #slip_ratios = torch.linspace(0.0, 0.0, 1)  # unitless
    #slip_angles = torch.linspace(-1.5, 1.5, 1)  # radians
    #slip_ratios = torch.linspace(-1.0, 1.0, 1)  # unitless

    # Create a meshgrid for slip angles and slip ratios
    SA, SR = torch.meshgrid(slip_angles, slip_ratios, indexing='ij')
    SA = SA.reshape(-1)
    SR = SR.reshape(-1)

    with torch.no_grad():
        #if hasattr(tire_model, 'tire_forces_model'):
        #    # Using the old tire model interface
        #    tfm = torch.compile(tire_model.tire_forces_model) if compile else tire_model.tire_forces_model
        #    Fx_f, Fy_f = tfm(SA, SR, tire_model.front_tire_model_parameters())
        #    Fx_r, Fy_r = tfm(SA, SR, tire_model.rear_tire_model_parameters())
        #    t0 = perf_counter()
        #    #for i in range(100):
        #    #    Fx_f, Fy_f = tfm(SA, SR, tire_model.front_tire_model_parameters())
        #    #    Fx_r, Fy_r = tfm(SA, SR, tire_model.rear_tire_model_parameters())
        #    #with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        #    #    with record_function("tfm"):
        #    #        #for i in range(10_000):
        #    #        for i in range(1):
        #    #            SA, SR = torch.randn(2)
        #    #            Fx_f, Fy_f = tfm(SA, SR, tire_model.front_tire_model_parameters())
        #    #            Fx_r, Fy_r = tfm(SA, SR, tire_model.rear_tire_model_parameters())
        #    #print(prof.key_averages().table(sort_by="cpu_time_total"))
        if hasattr(tire_model, 'tire_forces'):
            # Using the new tire model interface
            tf = torch.compile(tire_model.tire_forces) if compile else tire_model.tire_forces
            t0 = perf_counter()
            Fy_f, Fy_r, Fx_f, Fx_r = tf(SR, SA, SR, SA, dynamics_model.vehicle_parameters())
            Fy_f = Fy_f * dynamics_model.log_friction.exp()
            Fy_r = Fy_r * dynamics_model.log_friction.exp()
            Fx_f = Fx_f * dynamics_model.log_friction.exp()
            Fx_r = Fx_r * dynamics_model.log_friction.exp()
            #for i in range(100):
            #    Fy_f, Fy_r, Fx_f, Fx_r = tfs(SR, SA, SR, SA)
            #with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            #    with record_function("tfs"):
            #        #for i in range(10_000):
            #        for i in range(1):
            #            SA1, SR1, SA2, SR2 = torch.randn(4)
            #            Fy_f, Fy_r, Fx_f, Fx_r = tfs(SR1, SA1, SR2, SA2)
            #print(prof.key_averages().table(sort_by="cpu_time_total"))
        else:
            raise ValueError("Tire model does not have a recognized interface.")
        t1 = perf_counter()
        print(f"Tire model evaluation time: {t1 - t0:.6f} seconds")
        #assert False

    # Reshape forces back to grid shape for plotting
    Fx_f_grid = Fx_f.reshape(SAN, SRN).detach().numpy()
    Fy_f_grid = Fy_f.reshape(SAN, SRN).detach().numpy()
    Fx_r_grid = Fx_r.reshape(SAN, SRN).detach().numpy()
    Fy_r_grid = Fy_r.reshape(SAN, SRN).detach().numpy()
    SA_grid = SA.reshape(SAN, SRN).detach().numpy()
    SR_grid = SR.reshape(SAN, SRN).detach().numpy()

    plt.subplot(121)
    plt.plot(slip_angles.detach().numpy(), Fy_f_grid[:, 0], label='Fy at SR=-1.0')
    plt.plot(slip_angles.detach().numpy(), Fy_f_grid[:, slip_ratios.shape[0] // 4], label='Fy at SR=-0.5')
    plt.plot(slip_angles.detach().numpy(), Fy_f_grid[:, slip_ratios.shape[0] // 2], label='Fy at SR=0')
    plt.plot(slip_angles.detach().numpy(), Fy_f_grid[:, 3 * slip_ratios.shape[0] // 4], label='Fy at SR=0.5')
    plt.plot(slip_angles.detach().numpy(), Fy_f_grid[:, -1], label='Fy at SR=1.0')
    plt.xlabel('Slip Angle (rad)')
    plt.ylabel('Fy front Force (N)')
    plt.grid()
    plt.legend()

    plt.subplot(122)
    plt.plot(slip_angles.detach().numpy(), Fy_r_grid[:, 0], label='Fy at SR=-1.0')
    plt.plot(slip_angles.detach().numpy(), Fy_r_grid[:, slip_ratios.shape[0] // 4], label='Fy at SR=-0.5')
    plt.plot(slip_angles.detach().numpy(), Fy_r_grid[:, slip_ratios.shape[0] // 2], label='Fy at SR=0')
    plt.plot(slip_angles.detach().numpy(), Fy_r_grid[:, 3 * slip_ratios.shape[0] // 4], label='Fy at SR=0.5')
    plt.plot(slip_angles.detach().numpy(), Fy_r_grid[:, -1], label='Fy at SR=1.0')
    plt.xlabel('Slip Angle (rad)')
    plt.ylabel('Fy rear Force (N)')
    plt.legend()
    plt.grid()
    plt.show()

    plt.subplot(121)
    plt.plot(slip_ratios.detach().numpy(), Fx_f_grid[0, :], label='Fx at SA=-1.5')
    plt.plot(slip_ratios.detach().numpy(), Fx_f_grid[slip_angles.shape[0] // 4, :], label='Fx at SA=-0.75')
    plt.plot(slip_ratios.detach().numpy(), Fx_f_grid[slip_angles.shape[0] // 2, :], label='Fx at SA=0')
    plt.plot(slip_ratios.detach().numpy(), Fx_f_grid[3 * slip_angles.shape[0] // 4, :], label='Fx at SA=0.75')
    plt.plot(slip_ratios.detach().numpy(), Fx_f_grid[-1, :], label='Fx at SA=1.5')
    plt.xlabel('Slip Ratio')
    plt.ylabel('Fx front Force (N)')
    plt.legend()
    plt.grid()

    plt.subplot(122)
    plt.plot(slip_ratios.detach().numpy(), Fx_r_grid[0, :], label='Fx at SA=-1.5')
    plt.plot(slip_ratios.detach().numpy(), Fx_r_grid[slip_angles.shape[0] // 4, :], label='Fx at SA=-0.75')
    plt.plot(slip_ratios.detach().numpy(), Fx_r_grid[slip_angles.shape[0] // 2, :], label='Fx at SA=0')
    plt.plot(slip_ratios.detach().numpy(), Fx_r_grid[3 * slip_angles.shape[0] // 4, :], label='Fx at    SA=0.75')
    plt.plot(slip_ratios.detach().numpy(), Fx_r_grid[-1, :], label='Fx at SA=1.5')
    plt.xlabel('Slip Ratio')
    plt.ylabel('Fx rear Force (N)')
    plt.legend()
    plt.grid()
    plt.show()

    ## --- 3D Surface Plots ---
    #fig_3d = plt.figure(figsize=(16, 10))
    #fig_3d.suptitle("Tire Forces 3D Characteristics")

    ## Front Fx
    #ax1 = fig_3d.add_subplot(2, 2, 1, projection='3d')
    #surf1 = ax1.plot_surface(SR_grid, SA_grid, Fx_f_grid, cmap=cm.viridis, linewidth=0, antialiased=False)
    #ax1.set_xlabel('Slip Ratio')
    #ax1.set_ylabel('Slip Angle (rad)')
    #ax1.set_zlabel('Fx Front (N)')
    #ax1.set_title('Front Longitudinal Force (Fx)')

    ## Front Fy
    #ax2 = fig_3d.add_subplot(2, 2, 2, projection='3d')
    #surf2 = ax2.plot_surface(SR_grid, SA_grid, Fy_f_grid, cmap=cm.viridis, linewidth=0, antialiased=False)
    #ax2.set_xlabel('Slip Ratio')
    #ax2.set_ylabel('Slip Angle (rad)')
    #ax2.set_zlabel('Fy Front (N)')
    #ax2.set_title('Front Lateral Force (Fy)')

    ## Rear Fx
    #ax3 = fig_3d.add_subplot(2, 2, 3, projection='3d')
    #surf3 = ax3.plot_surface(SR_grid, SA_grid, Fx_r_grid, cmap=cm.viridis, linewidth=0, antialiased=False)
    #ax3.set_xlabel('Slip Ratio')
    #ax3.set_ylabel('Slip Angle (rad)')
    #ax3.set_zlabel('Fx Rear (N)')
    #ax3.set_title('Rear Longitudinal Force (Fx)')

    ## Rear Fy
    #ax4 = fig_3d.add_subplot(2, 2, 4, projection='3d')
    #surf4 = ax4.plot_surface(SR_grid, SA_grid, Fy_r_grid, cmap=cm.viridis, linewidth=0, antialiased=False)
    #ax4.set_xlabel('Slip Ratio')
    #ax4.set_ylabel('Slip Angle (rad)')
    #ax4.set_zlabel('Fy Rear (N)')
    #ax4.set_title('Rear Lateral Force (Fy)')

    #plt.tight_layout()
    #plt.show()