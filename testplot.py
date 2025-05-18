import numpy as np
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# # fig, axes = plt.subplots(nrows=2, ncols=2)

# # # cmap = cm.get_cmap('viridis')

# # normalizer = Normalize(0, 4)
# # im = cm.ScalarMappable(norm=normalizer, cmap=plt.cm.jet)

# # for i, ax in enumerate(axes.flat):

# #     ax.imshow(i + np.random.random((10, 10)), cmap=plt.cm.jet, norm=normalizer)
# #     ax.set_title(str(i))
# #     ax.set_xlabel("eh")

# # fig.colorbar(im, ax=axes.ravel().tolist())

# #--------------

# # Create subplots with adjusted spacing
# fig, axes = plt.subplots(
#     nrows=2, ncols=2,
#     gridspec_kw={'hspace': 0.5, 'wspace': 0.4}  # Vertical/horizontal spacing
# )

# normalizer = Normalize(-5, 11)
# im = cm.ScalarMappable(norm=normalizer, cmap=plt.cm.jet)


# X1, X2 = np.meshgrid(np.linspace(-1, 1, 200), np.linspace(-1, 1, 200), indexing='ij')
# sols0 = np.linspace(10, 11, 200*200)
# sols0 = np.reshape(sols0, (200, 200))
# sols05 = np.linspace(-5, 0, 200*200)
# sols05 = np.reshape(sols05, (200, 200))
# sols1 = np.linspace(0, 8, 200*200)
# sols1 = np.reshape(sols1, (200, 200))
# sols15 = np.linspace(2, 3, 200*200)
# sols15 = np.reshape(sols15, (200, 200))
# plots = [sols0, sols05, sols1, sols15]

# # Plot data and set subplot titles
# for i, ax in enumerate(axes.flat):
#     ax.contourf(X1, X2, plots[i], cmap=plt.cm.jet, norm=normalizer)
#     ax.set_title(str(i))

# # Add shared colorbar with padding
# fig.colorbar(im, ax=axes.ravel().tolist(), pad=0.08)

# # Adjust layout to leave space for the overall title
# fig.tight_layout(rect=[0, 0, 1, 0.92])  # Reserve 8% space at the top

# # Add overall title
# fig.suptitle('Overall Title', fontsize=14, y=0.98)



# plt.savefig("PDE-testplot")

# --------


# Define the grid
x, y = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))

# Create the disk function
z = np.sqrt(x**2 + y**2)

# Mask values outside the disk
z[z > 1] = np.nan

# Create the contour plot
plt.figure(figsize=(6, 6))
contour = plt.contourf(x, y, z, levels=20, cmap='viridis')
plt.colorbar(contour, label='Radius')

# Set plot limits and aspect ratio
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.gca().set_aspect('equal', adjustable='box')
plt.title('Disk Plot with Contourf')
plt.xlabel('x')
plt.ylabel('y')

plt.savefig("test-plot")
