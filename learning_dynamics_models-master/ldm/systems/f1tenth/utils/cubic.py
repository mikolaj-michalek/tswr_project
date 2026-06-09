import torch


def cubic_from_boundary(f0, df0, f1, df1, t):
    """
    Returns f(t) for a cubic polynomial satisfying:
        f(0) = a
        f'(0) = b
        f(1) = c
        f'(1) = d
    """
    A = df1 + df0 - 2*f1 + 2*f0
    B = 3*f1 - 3*f0 - 2*df0 - df1
    C = df0
    D = f0

    return A*t**3 + B*t**2 + C*t + D

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torch
    t = torch.linspace(0, 1.0, 1000)
    y = cubic_from_boundary(f0=2., df0=1.0, f1=2.2, df1=0., t=t)
    plt.plot(t.numpy(), y.numpy())
    plt.xlabel('t')
    plt.ylabel('f(t)')
    plt.title('Cubic Polynomial from Boundary Conditions')
    plt.grid()
    plt.show()