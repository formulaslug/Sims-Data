import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

class SliceViewer:
    def __init__(self, data_3d, title="3D Data Slice Viewer"):
        """
        Initialize the slice viewer with 3D data.
        
        Parameters:
        data_3d: numpy array of shape (X, Y, Z)
        title: window title
        """
        self.data = data_3d
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        plt.subplots_adjust(bottom=0.15)
        
        # Get global min/max for consistent colorbar
        self.vmin = np.min(self.data)
        self.vmax = np.max(self.data)
        
        # Initial slice at Z=0
        self.current_z = 0
        self.im = self.ax.imshow(self.data[:, :, self.current_z], 
                                  cmap='viridis', 
                                  origin='lower',
                                  vmin=self.vmin,
                                  vmax=self.vmax)
        self.ax.set_title(f'{title} - Z slice: {self.current_z}')
        plt.colorbar(self.im, ax=self.ax)
        
        # Create slider
        ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
        self.slider = Slider(ax_slider, 'Z', 0, self.data.shape[2] - 1, 
                             valinit=0, valstep=1)
        self.slider.on_changed(self.update)
        
    def update(self, val):
        """Update the displayed slice when slider changes."""
        self.current_z = int(self.slider.val)
        self.im.set_data(self.data[:, :, self.current_z])
        self.ax.set_title(f'Z slice: {self.current_z}')
        self.fig.canvas.draw_idle()
        
    def show(self):
        """Display the viewer."""
        plt.show()


# Example usage
if __name__ == "__main__":
    # Create sample 3D data
    x, y, z = np.meshgrid(np.linspace(-2, 2, 50),
                          np.linspace(-2, 2, 50),
                          np.linspace(-2, 2, 30))
    data_3d = np.sin(np.sqrt(x**2 + y**2 + z**2))
    
    viewer = SliceViewer(data_3d)
    viewer.show()
