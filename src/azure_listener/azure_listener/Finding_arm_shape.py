import numpy as np
from scipy.linalg import lstsq
from scipy.interpolate import make_smoothing_spline
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from matplotlib.animation import FuncAnimation


def evaluate_surface(x_vals, y_vals, coeffs, eX, eY):
    x_vals = np.array(x_vals).reshape(-1, 1)
    y_vals = np.array(y_vals).reshape(-1, 1)
    A = (x_vals ** eX) * (y_vals ** eY)
    z_vals = np.dot(A, coeffs).flatten()
    return z_vals
def calculate_3d_curvature(x, y, z):
    """
    Calculates the curvature of a 3D curve given x, y, and z arrays of points.

    Args:
        x (np.ndarray): Array of x-coordinates.
        y (np.ndarray): Array of y-coordinates.
        z (np.ndarray): Array of z-coordinates.

    Returns:
        np.ndarray: Array of curvature values for each point (excluding endpoints).
    """
    if not (len(x) == len(y) == len(z)):
        raise ValueError("Input arrays x, y, and z must have the same length.")
    if len(x) < 3:
        # Curvature requires at least 3 points for meaningful calculation
        return np.array([])

    # Calculate first derivatives (velocity vector components)
    dx = np.gradient(x)
    dy = np.gradient(y)
    dz = np.gradient(z)

    # Calculate second derivatives (acceleration vector components)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    ddz = np.gradient(dz)

    # Calculate cross product of r' and r''
    # r_prime = [dx, dy, dz]
    # r_double_prime = [ddx, ddy, ddz]
    cross_product_x = dy * ddz - dz * ddy
    cross_product_y = dz * ddx - dx * ddz
    cross_product_z = dx * ddy - dy * ddx

    # Magnitude of the cross product
    magnitude_cross_product = np.sqrt(cross_product_x**2 + cross_product_y**2 + cross_product_z**2)

    # Magnitude of the first derivative (speed)
    magnitude_r_prime = np.sqrt(dx**2 + dy**2 + dz**2)

    # Curvature
    # Avoid division by zero for points where speed is zero
    curvature = np.zeros_like(x, dtype=float)
    non_zero_speed_indices = magnitude_r_prime != 0
    curvature[non_zero_speed_indices] = magnitude_cross_product[non_zero_speed_indices] / \
                                        (magnitude_r_prime[non_zero_speed_indices]**3)

    return curvature
def reject_outliers(data, m = 2.):
        d = np.abs(data - np.median(data))
        mdev = np.median(d)
        s = d/mdev if mdev else np.zeros(len(d))
        return data[s<m]
def sort_xy_nearest_neighbor(x, y):
    """
    Sorts scattered 2D points (x, y) so that the resulting points are ordered
    along a nearest-neighbor path.

    Parameters:
        x (np.ndarray): 1D array of x-coordinates.
        y (np.ndarray): 1D array of y-coordinates.

    Returns:
        x_sorted (np.ndarray): x-values sorted in path order.
        y_sorted (np.ndarray): y-values sorted in path order.
    """
    points = np.column_stack((x, y))
    n = len(points)
    visited = np.zeros(n, dtype=bool)
    path = []
    current_idx = 0
    path.append(points[current_idx])
    visited[current_idx] = True

    for _ in range(1, n):
        dists = np.linalg.norm(points - points[current_idx], axis=1)
        dists[visited] = np.inf
        next_idx = np.argmin(dists)
        visited[next_idx] = True
        current_idx = next_idx
        path.append(points[current_idx])

    path = np.array(path)
    return path[:, 0], path[:, 1]
def average_duplicates(x, z):
    
    x = np.asarray(x)
    z = np.asarray(z)
    x_unique, indices = np.unique(x, return_inverse=True)
    z_avg = np.zeros_like(x_unique, dtype=np.float64)
    counts = np.zeros_like(x_unique, dtype=np.int32)
    
    for i, idx in enumerate(indices):
        z_avg[idx] += z[i]
        counts[idx] += 1

    z_avg /= counts
    return x_unique, z_avg
def surface_normal(x, y, C, e):
    x = float(x); y = float(y)
    fx = 0.0
    fy = 0.0
    for (i, (ex, ey)) in enumerate(e):
        coeff = C[i]
        if ex > 0:
            fx += coeff * ex * (x**(ex - 1)) * (y**ey)
        if ey > 0:
            fy += coeff * ey * (x**ex) * (y**(ey - 1))

    # Ensure fx and fy are scalars
    fx = fx[0] if isinstance(fx, np.ndarray) else fx
    fy = fy[0] if isinstance(fy, np.ndarray) else fy

    n = np.array([-fx, -fy, 1.0])
    n_hat = n / np.linalg.norm(n)
    return n_hat









def find_arm_shape(data, ax):
    
    # # Load data
    # data = np.loadtxt(file, delimiter=',')

    # Get data into array, sorting out points that are greater than a meter away.
    x, y, z = [], [], []
    for point in data:
        x1, y1, z1 = point
        if z1 < 1:
            x.append(x1)
            y.append(y1)
            z.append(z1)
    X = np.array(x).reshape(-1, 1)
    Y = np.array(y).reshape(-1, 1)
    Z = np.array(z).reshape(-1, 1)


    # Make a smoothing spline of the points projected onto the xy plane
    x_spln, y_spln1 = average_duplicates(x, y)
    sort_xy_nearest_neighbor(x_spln, y_spln1)
    z_spln = x_spln*0
    xnew = np.linspace(min(x_spln),max(x_spln),len(y_spln1))
    lam = 300
    spl = make_smoothing_spline(x_spln, y_spln1, lam=lam)


    # Fit a surface to the points
    order = 3
    e = [(x, y) for n in range(order + 1) # e is the exponent matrix 
        for y in range(n + 1)
        for x in range(n + 1) if x + y == n]
    eX = np.array([[x] for x, _ in e]).T
    eY = np.array([[y] for _, y in e]).T
    A = (X ** eX) * (Y ** eY)
    C, resid, _, _ = lstsq(A, Z) # C is the coefficient matrix
    XX, YY = np.meshgrid(np.linspace(X.min(), X.max(), 20),
                        np.linspace(Y.min(), Y.max(), 20))
    A_grid = (XX.reshape(-1, 1) ** eX) * (YY.reshape(-1, 1) ** eY)
    ZZ = np.dot(A_grid, C).reshape(XX.shape)


    ## The following evaluates each point of the spline at the surface in the z-direction.
    y_spln = spl.__call__(x_spln)
    z_spln = evaluate_surface(x_spln, y_spln, C, eX, eY)

    # The following computes the radius of curvature of the spline
    # curvature = calculate_3d_curvature(x_spln,y_spln,z_spln)
    # curvature = reject_outliers(curvature)
    # rad_curvature = 1/curvature
    # print(f"Radius of curvature stats:\n  Min:  {rad_curvature.min():.6f}\n  Max:  {rad_curvature.max():.6f}\n  Mean: {rad_curvature.mean():.6f}")


    # The following plots a new spline that is where the center of the robot arm is. 
    spln_points = np.array([x_spln,y_spln,z_spln]).T
    centerline_spln = np.empty([0,3])
    for i in range(len(x_spln)):
        n1 = surface_normal(x_spln[i], y_spln[i], C, e)
        # ax.quiver(x_spln[i], y_spln[i], z_spln[i], n1[0], n1[1], n1[2], length=.0125, color='red', normalize=True)
        centerline_spln = np.vstack([centerline_spln,spln_points[i]+n1*.0125])
    centerline_x, centerline_y, centerline_z = centerline_spln.T
    ax.plot(centerline_x,centerline_y, centerline_z, c = 'g', label = f'Centerline of arm')
    

    # --- Plot ---
    # ax.scatter(X, Y, Z, c='r', s=2)
    # ax.plot_surface(XX, YY, ZZ, rstride=1, cstride=1, alpha=0.2,
    #                 linewidth=0.5, edgecolor='b')
    # ax.plot(x_spln, y_spln, z_spln, label=fr'$\lambda=${lam}', c = 'g')

    

        # return rad_curvature.mean()


def plot_real_time(frame, ax):
    
    # Load data
    try:
        data = np.loadtxt(f'filtered_data_active.txt', delimiter=',')
        if data.ndim != 2 or data.shape[1] !=3:
            print("Skip!")
            return
    except Exception as e:
        print('Skipping frame due to error: ', e)
        return
    if not len(data) < 10:
        ax.cla()
        find_arm_shape(data, ax)
        ax.set_zlim(.2,.8)
        ax.set_ylim(-.5,.2)
        ax.set_xlim(-.3,.3)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()



def main():
    # create plot object
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ani = FuncAnimation(fig, plot_real_time, fargs = (ax,), interval = 100)
    plt.show()

if __name__ == '__main__':
    main()