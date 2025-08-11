import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter


# Generate synthetic noisy data for demonstration
# np.random.seed(0)
# t = np.linspace(0.1, 2*np.pi, 50)
# x = np.log(t) + 0.1*np.random.randn(len(t))
# y = np.sin(t) + 0.1*np.random.randn(len(t))
# z = t + 0.1*np.random.randn(len(t))

def Curve_fit2(i):

    data = np.loadtxt(f'/home/radlab/ros2_ws/src/azure_listener/azure_listener/filtered_data{i}.txt', delimiter = ',')


    x = []
    y = []
    z = []

    for point in data:
        x1, y1, z1 = point
        if z1 < 1:
            x.append(x1)
            y.append(y1)
            z.append(z1)

    x = np.array(x); y = np.array(y); z = np.array(z)

    # x = savgol_filter(x, window_length=9, polyorder=2)
    # y = savgol_filter(y, window_length=9, polyorder=2)
    # z = savgol_filter(z, window_length=9, polyorder=2)


    t = np.linspace(0,10,len(x))

    k = 4
    s = None

    # Fit splines to x(t), y(t), z(t)
    spl_x = UnivariateSpline(t, x, k=1, s=s)
    spl_y = UnivariateSpline(t, y, k=k, s=s)
    spl_z = UnivariateSpline(t, z, k=k, s=s)

    # Evaluate fitted curve at fine resolution
    t_fine = np.linspace(t.min(), t.max(), 200)

    # Trim fraction of time from both ends
    trim_fraction = 0.10  # Trim 5% from each end
    t_start = t_fine[int(len(t_fine) * trim_fraction)]
    t_end = t_fine[int(len(t_fine) * (1 - trim_fraction))]
    t_fine_trimmed = t_fine[(t_fine >= t_start) & (t_fine <= t_end)]

    trimmed_or_not = t_fine_trimmed


    x_fitted = spl_x(trimmed_or_not)
    y_fitted = spl_y(trimmed_or_not)
    z_fitted = spl_z(trimmed_or_not)

    # First and second derivatives
    dx = spl_x.derivative(1)(trimmed_or_not)
    dy = spl_y.derivative(1)(trimmed_or_not)
    dz = spl_z.derivative(1)(trimmed_or_not)

    # ddx = spl_x.derivative(2)(trimmed_or_not)
    ddx = np.linspace(0,1,161)*0
    ddy = spl_y.derivative(2)(trimmed_or_not)
    ddz = spl_z.derivative(2)(trimmed_or_not)

    # Compute curvature
    v = np.stack([dx, dy, dz], axis=1)     # velocity
    a = np.stack([ddx, ddy, ddz], axis=1)  # acceleration
    cross = np.cross(v, a)
    numerator = np.linalg.norm(cross, axis=1)
    denominator = np.linalg.norm(v, axis=1) ** 3
    curvature = numerator / denominator

    def reject_outliers(data, m = 2.):
        d = np.abs(data - np.median(data))
        mdev = np.median(d)
        s = d/mdev if mdev else np.zeros(len(d))
        return data[s<m]
    
    filtered_curvature = reject_outliers(curvature)

    # Find max and min curvature
    t_max = trimmed_or_not[np.argmax(curvature)]
    t_min = trimmed_or_not[np.argmin(curvature)]
    kappa_max = curvature.max()
    kappa_min = curvature.min()
    
    kappa_ave = np.average(curvature)

    print(f"Max curvature: {kappa_max:.4f} at t = {t_max:.4f}")
    print(f"Min curvature: {kappa_min:.4f} at t = {t_min:.4f}")
    print(f"Max Radius of Curvature = {1/kappa_min}")
    print(f"Min Radius of curvature = {1/kappa_max}")
    print(f"Average Curvature = {kappa_ave}")
    print(f"Average Radius of curvature = {1/kappa_ave}")
    print(f'Filtered ave Curvature = {np.mean(filtered_curvature)}')
    print(f'Filtered ave Radius of Curvature = {1/np.mean(filtered_curvature)}')


    def set_axes_equal(ax):
        '''Set equal scale for all 3 axes of a 3D plot.'''
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = x_limits[1] - x_limits[0]
        y_range = y_limits[1] - y_limits[0]
        z_range = z_limits[1] - z_limits[0]
        max_range = max([x_range, y_range, z_range]) / 2.0

        x_middle = np.mean(x_limits)
        y_middle = np.mean(y_limits)
        z_middle = np.mean(z_limits)

        ax.set_xlim3d([x_middle - max_range, x_middle + max_range])
        ax.set_ylim3d([y_middle - max_range, y_middle + max_range])
        ax.set_zlim3d([z_middle - max_range, z_middle + max_range])


    # Plot original points and fitted curve
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_fitted, y_fitted, z_fitted, label='Fitted Curve', color='blue')
    ax.scatter(x, y, z, label='Data Points', color='red', s=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    set_axes_equal(ax)
    # ax.set_xlim(-0.2, 0.2)
    # ax.set_ylim(-0.2, 0.2)
    # ax.set_zlim(0, 0.4)
    ax.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    Curve_fit2(1)