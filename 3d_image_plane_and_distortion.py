import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(15,5))

def plot_image_plane(x: np.ndarray, y: np.ndarray) -> None: 

    #first plot - Image plane
    ax = fig.add_subplot(131, projection='3d')

    X, Y = np.meshgrid(x, y)

    Z = np.zeros((np.max(x)+1, np.max(y)+1))
    ax.plot_surface(X, Y, Z,cmap='magma')
    plt.colorbar(ax.plot_surface(X, Y, Z, cmap='magma'), ax=ax, orientation='vertical',shrink=0.5, aspect=15, pad = 0.2)

    

def plot_r_sqr_plane(x: np.ndarray, y: np.ndarray, cx: int, cy: int) -> None: 
    #second Plot - r sqr plane

    
    ay = fig.add_subplot(132, projection='3d')

    nx, ny = (x-cx)/cx, (y-cy)/cx
    r = np.sqrt((nx)**2 + (ny)**2)

    NX,NY = np.meshgrid(nx, ny)
    Z = np.sqrt((NX)**2 + (NY)**2)
    Z2 = Z**2
    #display the surface plot
    ay.plot_surface(NX, NY, Z2,cmap='magma')
    plt.colorbar(ay.plot_surface(NX, NY, Z2, cmap='magma'), ax=ay, orientation='vertical',shrink=0.5, aspect=15, pad = 0.2)
    ay.set_xlabel('x\'')
    ay.set_ylabel('y\'')
    ay.set_zlabel('r2')


def plot_distortion_image_plane(x: np.ndarray, y: np.ndarray, cx: int, cy: int):

    #thrid plot - distorted image plane
    
    az = fig.add_subplot(133, projection='3d')

    nx, ny  = (x-cx)/cx, (y-cy)/cx
    r       = np.sqrt((nx)**2 + (ny)**2)

    NX,NY = np.meshgrid(nx, ny)

    k1 = 0.1
    k2 = 0.01 
    k3 = 0.01 

    NX2, NY2 = NX * (1 + k1 * r**2), NY * (1 + k1 * r**2)

    #NX2, NY2 = NX * (1 + k1 * r**2 + k2 * r**4  + k3 * r**6), NY * (1 + k1 * r**2 + k2 * r**4 + k3 * r**6)

    Z = np.abs(np.sqrt((NX2 - NX)**2 + (NY2 - NY)**2))

    print(np.max(Z))
    print(np.min(Z))

    # Label the axes
    az.set_xlabel('x\'')
    az.set_ylabel('y\'')
    az.set_zlabel('|M|')

    az.plot_surface(NX, NY, Z, cmap='magma')
    plt.colorbar(az.plot_surface(NX, NY, Z, cmap='magma'), ax=az, orientation='vertical',shrink=0.5, aspect=15, pad = 0.2)
    plt.show()

def main() -> None:

    image_width : int = 4096
    image_height : int = 4096

    cx : int = image_width//2 
    cy : int = image_height//2

    x = np.arange(0,image_width)
    y = np.arange(0,image_height)

    plot_image_plane(x,y)
    plot_r_sqr_plane(x,y,cx,cy)
    plot_distortion_image_plane(x,y,cx,cy)

if __name__ == "__main__": 
    main()













