# Astrolab Camera Characterizer
This program is meant to handle any .fits images and process and display statistics relating to noise. Currently it can handle bias images and dark images with planned implementation for pixel by pixel transfer curve imaging. Eventually CCD cameras will hopefully be supported as well.

Note that because these image files are very large this program stores the stack of images as a numpy memorymap in whatever folder this program is stored. For 1000 2048x2048 16 bit images, this takes up 32 GiB of space, and that space should scale linearly with those dimensions. This map is deleted after the program runs, so let it finish or you will have to manually delete this 'stack.memorymap' file.
