import matplotlib.pyplot as plt

def plot_final(state):
    plt.imshow(state, cmap='hot')
    plt.title("Final Burn Map")
    plt.colorbar()
    plt.show()
