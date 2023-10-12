import sys
from os import listdir
from os.path import isfile, join

from fpdf import FPDF

# Parent directories
files_folder = sys.argv[1:][0]
avgret_images_dir = files_folder + '/avgret'
evals_images_dir = files_folder + '/evals'

# Get all image files
image_filenames = [f for f in listdir(avgret_images_dir) if isfile(join(avgret_images_dir, f))]
print('list of images: ', image_filenames)
print('#images: ', len(image_filenames))


# Create and fill in pdf file
FONT = 'Arial'
MARGIN = 10
LINE_HEIGHT = 20
IMG_WIDTH = 200
IMG_HEIGHT = 100



pdf = FPDF()

pdf.set_font(FONT)

current_page_images = 0
cont = 0

for img_file in image_filenames:
    pdf.add_page()
    x = MARGIN
    y = LINE_HEIGHT
    pdf.set_xy(x, y)

    title = img_file.replace('.png', '').replace('_', ' ')
    pdf.cell(len(title) * 2, txt=title, align='L')
    y += LINE_HEIGHT
    pdf.set_y(y)

    pdf.image(avgret_images_dir + '/' + img_file, x=x, y=y, w=IMG_WIDTH, h=IMG_HEIGHT)
    # x = MARGIN + IMG_WIDTH
    y += LINE_HEIGHT + IMG_HEIGHT
    x = MARGIN
    pdf.set_xy(x, y)
    pdf.image(evals_images_dir + '/' + img_file, x=x, y=y, w=IMG_WIDTH, h=IMG_HEIGHT)
    y += IMG_HEIGHT

    # pdf.add_page()
    # if current_page_images == 2:
    #     pdf.add_page()
    #     current_page_images = 0

    # cont += 1
    # if cont == 5:
    #     break

pdf.output('./report.pdf')
