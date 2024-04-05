import numpy as np
import matplotlib.pyplot as plt

def plot_image_plane(x: np.ndarray, y: np.ndarray) -> None: 

    #first plot - Image plane
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    X, Y = np.meshgrid(x, y)

    Z = np.zeros((300, 300))
    ax.plot_surface(X, Y, Z)

    plt.show()

def plot_r_sqr_plane(x: np.ndarray, y: np.ndarray, cx: int, cy: int) -> None: 
    #second Plot - r sqr plane

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    nx, ny = (x-cx)/cx, (y-cy)/cx
    r = np.sqrt((nx)**2 + (ny)**2)

    NX,NY = np.meshgrid(nx, ny)
    Z = np.sqrt((NX)**2 + (NY)**2)

    #display the surface plot
    ax.plot_surface(NX, NY, Z)
    plt.show()


def plot_distortion_image_plane(x: np.ndarray, y: np.ndarray, cx: int, cy: int):

    #thrid plot - distorted image plane
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    nx, ny  = (x-cx)/cx, (y-cy)/cx
    r       = np.sqrt((nx)**2 + (ny)**2)

    NX,NY = np.meshgrid(nx, ny)

    k1 = 0.1
    k2 = 0.1 
    k3 = 0.1 

    NX2, NY2 = NX * (1 + k1 * r**2), NY * (1 + k1 * r**2  )

    #NX2, NY2 = NX * (1 + k1 * r**2 + k2 * r**4  + k3 * r**6), NY * (1 + k1 * r**2 + k2 * r**4 + k3 * r**6)

    Z = np.abs(np.sqrt((NX2 - NX)**2 + (NY2 - NY)**2))

    print(np.max(Z))
    print(np.min(Z))

    # Label the axes
    ax.set_xlabel('Normalized x-coordinates')
    ax.set_ylabel('Normalized y-coordinates')
    ax.set_zlabel('')

    ax.plot_surface(NX, NY, Z, cmap='viridis')
    plt.show()

def main() -> None:

    image_width : int = 300 
    image_height : int = 300

    cx : int = image_width//2 
    cy : int = image_height//2

    x = np.arange(0,image_width)
    y = np.arange(0,image_height)

    plot_image_plane(x,y)
    plot_r_sqr_plane(x,y,cx,cy)
    plot_distortion_image_plane(x,y,cx,cy)

if __name__ == "__main__": 
    main()













