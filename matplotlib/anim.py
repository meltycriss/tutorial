import imageio
import os

png_names = os.listdir('./png')
filenames = map(lambda p: os.path.abspath(os.path.join('./png', p)), png_names)


print filenames[:10]

#print os.path.exists('anim000.png')

im = imageio.imread('000000.jpg')

#with imageio.get_writer('cube.gif', mode='I') as writer:
#    for filename in filenames:
#        image = imageio.imread(filename)
#        writer.append_data(image)