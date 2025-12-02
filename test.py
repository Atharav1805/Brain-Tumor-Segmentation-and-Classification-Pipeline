import numpy as np
import matplotlib.pyplot as plt

# Load the .npy image
img = np.load("/home/atharav/Downloads/CS663_Excluded_From_Presentation/Data/converted_npy/1_image.npy")

# Display & save as PNG
plt.imshow(img, cmap="gray")
plt.axis("off")
plt.savefig("image.png", bbox_inches="tight", pad_inches=0, dpi=300)
plt.show()
