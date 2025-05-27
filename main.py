import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#from scipy.integrate import simps
from scipy.integrate import simpson



class MobiusStrip:
    def __init__(self, R=1.0, w=0.2, n=200):
        self.R = R
        self.w = w
        self.n = n
        self.u = np.linspace(0, 2 * np.pi, n)
        self.v = np.linspace(-w / 2, w / 2, n)
        self.U, self.V = np.meshgrid(self.u, self.v)
        self.X, self.Y, self.Z = self._generate_mesh()

    def _generate_mesh(self):
        U, V = self.U, self.V
        R = self.R

        X = (R + V * np.cos(U / 2)) * np.cos(U)
        Y = (R + V * np.cos(U / 2)) * np.sin(U)
        Z = V * np.sin(U / 2)
        return X, Y, Z

    def compute_surface_area(self):
        # Partial derivatives
        du = 2 * np.pi / (self.n - 1)
        dv = self.w / (self.n - 1)

        Xu = np.gradient(self.X, du, axis=1)
        Yu = np.gradient(self.Y, du, axis=1)
        Zu = np.gradient(self.Z, du, axis=1)

        Xv = np.gradient(self.X, dv, axis=0)
        Yv = np.gradient(self.Y, dv, axis=0)
        Zv = np.gradient(self.Z, dv, axis=0)

        # Cross product magnitude
        cross = np.sqrt(
            (Yu * Zv - Zu * Yv) ** 2 +
            (Zu * Xv - Xu * Zv) ** 2 +
            (Xu * Yv - Yu * Xv) ** 2
        )

        area = simpson(simpson(cross, self.v), self.u)
        return area

    def compute_edge_length(self):
        # Top and bottom edges
        edges = []
        for row in [0, -1]:
            x = self.X[row, :]
            y = self.Y[row, :]
            z = self.Z[row, :]
            dx = np.diff(x)
            dy = np.diff(y)
            dz = np.diff(z)
            ds = np.sqrt(dx**2 + dy**2 + dz**2)
            edges.append(np.sum(ds))
        return np.sum(edges)

    def plot(self):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(self.X, self.Y, self.Z, rstride=5, cstride=5, color='cyan', edgecolor='k', alpha=0.7)
        ax.set_title("Möbius Strip")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    mobius = MobiusStrip(R=1.0, w=0.3, n=300)
    area = mobius.compute_surface_area()
    edge_len = mobius.compute_edge_length()

    print(f"Surface Area ≈ {area:.4f}")
    print(f"Edge Length ≈ {edge_len:.4f}")
    mobius.plot()
