import numpy as np
import matplotlib.pyplot as plt

def plot_ego_transformed_boundaries(ego_pose, track_x, track_y, ego_transformed_boundaries):
    car_length = 0.4
    car_width = 0.2

    plt.subplot(121)
    plt.plot(track_x.cpu().numpy(),
             track_y.cpu().numpy(), 'k-', label='Track Centerline')
    # Plot the vehicle as a rotated rectangle in world frame

    # Get current position and orientation
    x_pos = ego_pose[0].cpu().numpy()
    y_pos = ego_pose[1].cpu().numpy()
    yaw = ego_pose[2].cpu().numpy()

    # Create rectangle vertices (centered at origin, facing along x-axis)
    corners = np.array([
        [-car_length/2, -car_width/2],
        [car_length/2, -car_width/2],
        [car_length/2, car_width/2],
        [-car_length/2, car_width/2]
    ])

    # Rotation matrix
    R = np.array([
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw), np.cos(yaw)]
    ])

    # Rotate and translate the corners
    rotated_corners = np.dot(corners, R.T) + np.array([x_pos, y_pos])

    # Plot the vehicle
    plt.plot(rotated_corners[:, 0], rotated_corners[:, 1], 'r-')
    plt.plot(np.append(rotated_corners[:, 0], rotated_corners[0, 0]), 
                np.append(rotated_corners[:, 1], rotated_corners[0, 1]), 'r-', label='Vehicle')

    # Add an arrow to show heading direction
    plt.arrow(x_pos, y_pos, 
                car_length/2 * np.cos(yaw), 
                car_length/2 * np.sin(yaw), 
                head_width=car_width/2, head_length=car_length/3, 
                fc='r', ec='r')

    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.legend()
    plt.axis('equal') 

    plt.subplot(122)
    ax = plt.gca()
    
    # Get transformed boundaries
    
    # Extract left and right boundary points
    left_x = ego_transformed_boundaries[:, 0].cpu().numpy()
    left_y = ego_transformed_boundaries[:, 1].cpu().numpy()
    right_x = ego_transformed_boundaries[:, 2].cpu().numpy()
    right_y = ego_transformed_boundaries[:, 3].cpu().numpy()
    
    # Calculate centerline as average of left and right boundaries
    center_x = (left_x + right_x) / 2
    center_y = (left_y + right_y) / 2
    
    # Plot boundaries and centerline
    ax.plot(left_x, left_y, 'b-', linewidth=2, label='Left Boundary')
    ax.plot(right_x, right_y, 'r-', linewidth=2, label='Right Boundary')
    ax.plot(center_x, center_y, 'g--', linewidth=1, label='Centerline')
    
    # Ego is at origin facing toward positive x-axis
    ego_rect = plt.Rectangle((-car_length/2, -car_width/2), car_length, car_width, 
                            color='black', alpha=0.5, label='Ego Vehicle')
    ax.add_patch(ego_rect)
    
    # Add arrow to show vehicle heading
    ax.arrow(0, 0, car_length/2, 0, head_width=0.3, head_length=0.3, fc='k', ec='k')
    
    # Set equal aspect ratio and labels
    ax.set_aspect('equal')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title('Ego-Centric View of Track')
    ax.legend()
    ax.grid(True)
    
    # Set reasonable axis limits
    max_range = max(np.max(np.abs([left_x, left_y, right_x, right_y])), 10)
    ax.set_xlim([-2, max_range])
    ax.set_ylim([-max_range/2, max_range/2])

    plt.show()