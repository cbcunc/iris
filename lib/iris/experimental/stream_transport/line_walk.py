
import numpy as np
import scipy.spatial


def _min_max_edge(lon_tr_bounds, lat_tr_bounds):
    lat_diff = np.diff(lat_tr_bounds)
    lon_diff = np.diff(lon_tr_bounds)
    dist = np.sqrt(lat_diff**2 + lon_diff**2)
    dist = np.ma.masked_equal(dist, 0.0)
    return dist.min(), dist.max()


def _nearest_corner(point, prev_ij, curr_ij, lon_tr_bounds, lat_tr_bounds):
    ij1 = (prev_ij[0], curr_ij[1])
    ij2 = (curr_ij[0], prev_ij[1])

    pnt1 = np.array((lon_tr_bounds[ij1], lat_tr_bounds[ij1]))
    pnt2 = np.array((lon_tr_bounds[ij2], lat_tr_bounds[ij2]))

    dist1 = np.sqrt(np.sum((pnt1 - point)**2))
    dist2 = np.sqrt(np.sum((pnt2 - point)**2))

    return np.array(ij1 if dist1 < dist2 else ij2)


def remove_spike(path):
    if len(path) >= 3:
        if np.array_equal(path[-1], path[-3]):
            path = path[:-2]
    return path


# TODO: Make sure we can handle >2D cubes (currently fails, I think)
def find_path(cube, line):
    start, end = line

    # top right corners
    y_coord = cube.coord(axis="Y")
    x_coord = cube.coord(axis="x")
    lat_tr_bounds = y_coord.bounds[..., 2].squeeze()  # TODO: no squeeze
    lon_tr_bounds = x_coord.bounds[..., 2].squeeze()
    shape = lat_tr_bounds.shape

    # put the corners into a kdtree for fast nearest point searching
    ll_pairs = zip(lon_tr_bounds.flatten(), lat_tr_bounds.flatten())
    kdtree = scipy.spatial.cKDTree(ll_pairs)

    # walk the line
    min_edge, max_edge = _min_max_edge(lon_tr_bounds, lat_tr_bounds)

    # TODO: adaptive step size?
    # for lataitude = 80
    # 0.1:    pointwalk 80 [160, 9, 1, 1, 357, 3, 210]
    # 0.01:   pointwalk 80 [156, 9, 1, 1, 355, 3, 200]
    # 0.001:  pointwalk 80 [156, 9, 1, 1, 355, 3, 200]
    # 0.0001: pointwalk 80 [156, 9, 1, 1, 355, 3, 200]
    step = max(min_edge / 10.0, 0.01)

    dist_vect = end - start
    line_len = np.sqrt(np.sum(dist_vect**2))
    num_steps = int(line_len / step)  # TODO: do the full length of the line
    
    # List of path tuples
    segs = [[]]

    def sanity_check_last_points(segs):
        if len(segs[-1]) >= 2:
            ij_diff = segs[-1][-2] - segs[-1][-1]
            if np.sum(np.abs(ij_diff)) != 1:
                raise Exception(str(segs[-1][-2]) + str(segs[-1][-1]))

    def add_point(segs, new_ij):
        if len(segs[-1]) > 0 and np.all(new_ij == segs[-1][-1]):
            raise ValueError("not adding duplicate" + str(new_ij))
        segs[-1].append(new_ij)
        sanity_check_last_points(segs)
        segs[-1] = remove_spike(segs[-1])
        sanity_check_last_points(segs)

    # Find closest corner points to line
    d = dist_vect / num_steps
    for i in range(num_steps):
        point = start + d*i

        # Get the nearest 3 points and handle co-located points.
        near_dists, near_indices = kdtree.query(point, k=3)#, distance_upper_bound=max_edge)

        # Isolate those that are the same.
        near_dists = np.array(near_dists)
        use_these = np.where(near_dists == near_dists[0])
        near_dists = near_dists[use_these]
        near_indices = near_indices[use_these]
        
        # More than one nearest point?
        if len(near_dists) > 1:
            # Get the ij index for each.
            near_ijs = []
            for near_i in near_indices: 
                near_ij = np.array((int(near_i / shape[1]), near_i % shape[1]))
                near_ijs.append(near_ij)

            # Select nearest ij to last point on the current line.
            if len(segs[-1]) > 0:
                diffs = np.sum(np.abs(near_ijs - segs[-1][-1]), axis=1)
                min_i = np.where(diffs == diffs.min())[0]
                near_ij = near_ijs[min_i]
                near_dist = near_dists[min_i]
            else:
                warnings.warn("can't check last point on a new path")
                near_dist = near_dists[0]
                near_ij = near_ijs[0]
            
        # Just one nearest point.
        else:
            near_i = near_indices[0]
            near_dist = near_dists[0]
            near_ij = np.array((int(near_i / shape[1]), near_i % shape[1]))
        
        # Outside the grid?
        if not np.isfinite(near_dist):
            continue 

        # TODO: I think we need to do this...
#         near_ij[0] -= 1
#         near_ij[1] -= 1

        # Before we add the point...
        if len(segs[-1]) > 0:
            ijdiff = near_ij - segs[-1][-1]

            # Are we still near the same point as last time?
            if np.all(ijdiff == 0):
                continue
            
            # Start a new line segment?
            # TODO: Consider this condition in greater detail.
            if np.all(np.abs(ijdiff) > 1) or np.any(np.abs(ijdiff) > 10):
                segs.append([])

            # Multi cell step? (either i or j diff > 1)
            elif np.any(np.abs(ijdiff) > 1):
                # Add points until we're 1 step away
                walk_back_dim = np.where(np.abs(ijdiff) > 1)[0]
                dir = 1 if ijdiff[walk_back_dim] > 0 else -1
                num_walkback_steps = abs(ijdiff[walk_back_dim]) - 1

                walkback_ij_start = segs[-1][-1].copy()
                for walkback_step in range(num_walkback_steps):
                    walkback_ij = walkback_ij_start.copy()
                    walkback_ij[walk_back_dim] += dir * (walkback_step+1)
                    add_point(segs, walkback_ij)
                    
                # Check we've got a 1,1
                ijdiff = near_ij - segs[-1][-1]
                assert(np.all(np.abs(ijdiff) <= 1))
                
            # Opposite corner step? (i and j diff == 1)
            # Note: no else here, we can go after walkback.
            if np.all(np.abs(ijdiff) == 1):
                # Add nearest corner so we're one step away.
                new_ij = _nearest_corner(point, segs[-1][-1], near_ij,
                                            lon_tr_bounds, lat_tr_bounds)
                add_point(segs, new_ij)
                
        # We should be certain we're one step away now..
        add_point(segs, near_ij)

    # sanity check the segs: ensure single u or v steps
    for path in segs:
        for i in range(len(path) - 1):
            ij_diff = path[i] - path[i+1]
            if np.sum(np.abs(ij_diff)) != 1:
                raise Exception("Line walker error " + str(path[i]) + str(path[i+1]))

    return segs
