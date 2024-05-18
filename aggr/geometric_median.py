import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import torch
# https://github.com/mrwojo/geometric_median?utm_source=pocket_reader
def geometric_median(model_sample_weight_dict, normalize_factor, method='weiszfeld', options={} ):
    """
    Calculates the geometric median of an array of points.
    method specifies which algorithm to use:
        * 'auto' -- uses a heuristic to pick an algorithm
        * 'minimize' -- scipy.optimize the sum of distances
        * 'weiszfeld' -- Weiszfeld's algorithm
    """
    points = []
    for key, value in model_sample_weight_dict.items():
        points.append(torch.flatten(value).cpu().tolist())

    points = np.asarray(points)
    # print("points.shape", points.shape)

    if len(points.shape) == 1:
        # geometric_median((0, 0)) has too much potential for error.
        # Did the user intend a single 2D point or two scalars?
        # Use np.median if you meant the latter.
        raise ValueError("Expected 2D array")

    if method == 'auto':
        if points.shape[1] >= 2:
            # weiszfeld tends to converge faster in higher dimensions
            method = 'weiszfeld'
        else:
            method = 'minimize'

    geo_med = _methods[method](points, options)
    # geo_med = torch.as_tensor(geo_med)
    # geo_med = torch.reshape(geo_med, ())

    client_weights = {}

    for key, value in model_sample_weight_dict.items():
        # print("geo_med", geo_med.size(), geo_med)
        # print("value", value.size(), value)
        client_weights[key] = float(100 / cdist([torch.flatten(value).cpu().tolist()], [geo_med]))

    factor = (100.0 * normalize_factor) / sum(client_weights.values())

    for k in client_weights:
        client_weights[k] = client_weights[k] * factor

    return geo_med, client_weights


def minimize_method(points, options={}):
    """
    Geometric median as a convex optimization problem.
    """

    # objective function
    def aggregate_distance(x):
        return cdist([x], points).sum()

    # initial guess: centroid
    centroid = points.mean(axis=0)

    optimize_result = minimize(aggregate_distance, centroid, method='COBYLA')

    return optimize_result.x


def weiszfeld_method(points, options={}):
    """
    Weiszfeld's algorithm as described on Wikipedia.
    """
    default_options = {'maxiter': 100, 'tol': 1e-7}
    default_options.update(options)
    options = default_options

    def distance_func(x):
        return cdist([x], points)

    # initial guess: centroid
    guess = points.mean(axis=0)

    iters = 0

    while iters < options['maxiter']:
        distances = distance_func(guess).T

        # catch divide by zero
        distances = np.where(distances == 0, 1, distances)

        guess_next = (points/distances).sum(axis=0) / (1./distances).sum(axis=0)

        guess_movement = np.sqrt(((guess - guess_next)**2).sum())

        guess = guess_next

        if guess_movement <= options['tol']:
            break

        iters += 1

    return guess


_methods = {
    'minimize': minimize_method,
    'weiszfeld': weiszfeld_method,
}