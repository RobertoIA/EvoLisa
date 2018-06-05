import numpy as np

from PIL import Image as plImage
from PIL import ImageDraw as plDraw

from numpy.random import randint

def resizer(original, internal_size):
    factor = max(original.size) / internal_size

    def reduce(image):
        '''Reduces source image to internal resolution'''
        reduced, w, h = image.copy(), original.size[0] / factor, original.size[1] / factor
        reduced.thumbnail((w, h))
        return np.array(reduced), reduced.size[0], reduced.size[1]
    
    def restore(shapes, points, colors):
        '''Restores generated image to original resolution'''
        return draw_image(*original.size, shapes * factor, points, colors, antialiasing=True)
    
    return reduce, restore

def initialize(n_shapes, min_points, max_points, width, height):
    '''Initializes random polygons for target image'''
    shapes = np.empty((n_shapes, max_points * 2), dtype=np.dtype('int'))
    shapes[:,0::2] = randint(0, width, size=(n_shapes, max_points))
    shapes[:,1::2] = randint(0, height, size=(n_shapes, max_points))

    points = np.full(n_shapes, min_points)

    colors = randint(0, 256, size=(n_shapes, 4), dtype=np.dtype('uint8'))
    
    return shapes, points, colors

def draw_image(width, height, shapes, points, colors, antialiasing=False):
    '''Draws image from a set of polygons with or without antialiasing'''
    scale = 4 if antialiasing else 1
    image = plImage.new('RGB', (width * scale, height * scale), (255, 255, 255, 0))
    drawer = plDraw.Draw(image, 'RGBA')

    for shape, point, color in zip(shapes, points, colors):
        drawer.polygon((shape[:point * 2] * scale).tolist(), tuple(color))
    if antialiasing: image.thumbnail((width, height))
    
    return image

def error_abs(a, b):
    '''Calculates difference between two image matrices'''
    return np.abs(np.subtract(a, b, dtype=np.dtype('i4'))).sum()

def error_percent(error, image):
    '''Calculates human-readable % of error from absolute error'''
    return error / (image.shape[0] * image.shape[1] * 255 * 3) * 100

def generate(source, n_shapes, min_points, max_points, internal_res):
    '''Build image. Interrupt program to return current image'''

    def changes(shapes, points, colors):
        '''Selects a polygon and randomly executes a change over it'''

        # Configuration for changes
        point_rng = max(width / 2, height / 2)
        shape_rng = max(width / 2, height / 2)
        color_rng = 100
        alpha_rng = 100

        def point(shapes, points, colors, index):
            '''Random change to one point in xy axis'''
            shapes = shapes.copy()
            change, point = randint(-point_rng, point_rng+1, size=2), np.random.choice(max_points) * 2

            shapes[index, point:point + 2] = np.clip(shapes[index, point:point + 2] + change, 0, [width, height])
            return shapes, points, colors

        def shape(shapes, points, colors, index):
            '''Random change to a polygon in xy axis'''
            shapes = shapes.copy()
            change = np.tile(randint(-shape_rng, shape_rng+1, size=2), max_points)
            
            shapes[index] = np.clip(shapes[index] + change, 0, boundaries)
            return shapes, points, colors

        def order(shapes, points, colors, index):
            '''Random change to drawing order of a polygon's points'''
            shapes = shapes.copy()
            shuffle = np.random.permutation(points[index])
            shapes[index][0:points[index] *2:2] = shapes[index][0:points[index] *2:2][shuffle]
            shapes[index][1:points[index] *2:2] = shapes[index][1:points[index] *2:2][shuffle]
            return shapes, points, colors

        def number(shapes, points, colors, index):
            '''Change the number of sides of a polygon'''
            points = points.copy()
            if points[index] == min_points:
                points[index] = points[index] + 1
            elif points[index] == max_points:
                points[index] = points[index] - 1
            else:
                points[index] = points[index] + np.random.choice([1, -1])
            return shapes, points, colors

        def color(shapes, points, colors, index):
            '''Random change to the color of a polygon in the RGB axis'''
            colors = colors.copy()
            change = randint(-color_rng, color_rng+1, size=3)

            colors[index][:3] = np.clip(colors[index][:3] + change, 0, 256)
            return shapes, points, colors

        def alpha(shapes, points, colors, index):
            '''Random change to the transparency (alpha layer) of a polygon'''
            colors = colors.copy()
            change = randint(-alpha_rng, alpha_rng+1)

            colors[index][3] = np.clip(colors[index][3] + change, 0, 256)
            return shapes, points, colors

        index, func = randint(n_shapes), np.random.choice([point, shape, order, number, color, alpha])
        return func(shapes, points, colors, index)

    def iterate(shapes, points, colors, image, score):
        '''Makes one change to current set of polygons and returns the best one of the two'''
        new_shapes, new_points, new_colors = changes(shapes, points, colors)
        new_image = np.array(draw_image(width, height, new_shapes, new_points, new_colors))
        new_score = error_abs(internal, new_image)

        if new_score <= score:
            return new_shapes, new_points, new_colors, new_image, new_score
        return shapes, points, colors, image, score

    original = plImage.open(source).convert('RGB')
    reduce, restore = resizer(original, internal_res)
    internal, width, height = reduce(original)

    print('Generating {}x{} image with {}x{} internal resolution'.format(*original.size, width, height))

    shapes, points, colors = initialize(n_shapes, min_points, max_points, width, height)

    boundaries = np.tile([width, height], max_points)

    image = np.array(draw_image(width, height, shapes, points, colors))
    score = error_abs(internal, image)

    print('{:>12}  {:>12}  {:>12}  {:>12}'.format('iteration', 'error %', 'error abs', 'avg vert'))

    try:
        it = 0
        while True:
            if not it % 10000: # prints info each 10k iterations
                print('{:>12}  {:>11.4f}%  {:>12}  {:>12.2f}'.format(
                    it, error_percent(score, internal), score, points.mean()))
            shapes, points, colors, image, score = iterate(shapes, points, colors, image, score)
            it += 1
    except:
        print('Generation finished at {} iterations.'.format(it))
        restore(shapes, points, colors).save('evolisa.png', 'PNG')

if __name__ == '__main__':
    from sys import argv

    error_args = (
        "Invalid arguments. Usage:\n"
        "python evolisa.py [source] [# shapes] [min sides] [max sides] [internal res]\n"
        "Example: python evolisa.py source.jpg 50 3 20 160"
        )

    try:
        source = argv[1]
        n_shapes = int(argv[2])
        min_points, max_points = int(argv[3]), int(argv[4])
        internal_res = int(argv[5])
    except:
        print(error_args)

    generate(source, n_shapes, min_points, max_points, internal_res)
        
