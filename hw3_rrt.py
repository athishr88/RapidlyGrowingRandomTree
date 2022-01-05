import numpy as np
import matplotlib.pyplot as plt
import math
import random


name="Athish"
GRAD=True


###############################################################################
## Base Code
###############################################################################
DYNAMICS_MODE = None
class Node:
    def __init__(self, pt, parent=None):
        self.point = pt # n-Dimensional point
        self.parent = parent # Parent node
        self.path_from_parent = [] # List of points along the way from the parent node (for visualization)

def get_nd_obstacle(state_bounds):
    center_vector = []
    for d in range(state_bounds.shape[0]):
        center_vector.append(state_bounds[d][0] + random.random()*(state_bounds[d][1]-state_bounds[d][0]))
    radius = random.random() * 0.6
    return [np.array(center_vector), radius]

def setup_random_2d_world():
    state_bounds = np.array([[0,10],[0,10]])

    obstacles = [] # [pt, radius] circular obstacles
    for n in range(30):
        obstacles.append(get_nd_obstacle(state_bounds))

    def state_is_valid(state):
        for dim in range(state_bounds.shape[0]):
            if state[dim] < state_bounds[dim][0]: return False
            if state[dim] >= state_bounds[dim][1]: return False
        for obs in obstacles:
            if np.linalg.norm(state - obs[0]) <= obs[1]: return False
        return True

    return state_bounds, obstacles, state_is_valid

def setup_fixed_test_2d_world():
    state_bounds = np.array([[0,1],[0,1]])
    obstacles = [] # [pt, radius] circular obstacles
    obstacles.append([[0.5,0.5],0.2])
    obstacles.append([[0.1,0.7],0.1])
    obstacles.append([[0.7,0.2],0.1])

    def state_is_valid(state):
        for dim in range(state_bounds.shape[0]):
            if state[dim] < state_bounds[dim][0]: return False
            if state[dim] >= state_bounds[dim][1]: return False
        for obs in obstacles:
            if np.linalg.norm(state - obs[0]) <= obs[1]: return False
        return True

    return state_bounds, obstacles, state_is_valid


def plot_circle(x, y, radius, color="-k"):
    deg = np.linspace(0,360,50)

    xl = [x + radius * math.cos(np.deg2rad(d)) for d in deg]
    yl = [y + radius * math.sin(np.deg2rad(d)) for d in deg]
    plt.plot(xl, yl, color)

def visualize_2D_graph(state_bounds, obstacles, nodes, goal_point=None, filename=None):
    '''
    @param state_bounds Array of min/max for each dimension
    @param obstacles Locations and radii of spheroid obstacles
    @param nodes List of vertex locations
    @param edges List of vertex connections
    '''

    fig = plt.figure()
    plt.xlim(state_bounds[0,0], state_bounds[0,1])
    plt.ylim(state_bounds[1,0], state_bounds[1,1])

    for obs in obstacles:
        plot_circle(obs[0][0], obs[0][1], obs[1])

    goal_node = None
    for node in nodes:
        if node.parent is not None:
            node_path = np.array(node.path_from_parent)
            plt.plot(node_path[:,0], node_path[:,1], '-b')
        if goal_point is not None and np.linalg.norm(node.point - np.array(goal_point)) <= 1e-5:
            goal_node = node
            plt.plot(node.point[0], node.point[1], 'k^')
        else:
            plt.plot(node.point[0], node.point[1], 'ro')

    plt.plot(nodes[0].point[0], nodes[0].point[1], 'ko')

    if goal_node is not None:
        cur_node = goal_node
        while cur_node is not None: 
            if cur_node.parent is not None:
                node_path = np.array(cur_node.path_from_parent)
                plt.plot(node_path[:,0], node_path[:,1], '--y')
                cur_node = cur_node.parent
            else:
                break

    if goal_point is not None:
        plt.plot(node.point[0], node.point[1], 'gx')


    if filename is not None:
        fig.savefig(filename)
    else:
        plt.show()


def get_random_valid_vertex(state_valid, bounds, obstacles):
    vertex = None
    while vertex is None: # Get starting vertex
        pt = np.random.rand(bounds.shape[0]) * (bounds[:,1]-bounds[:,0]) + bounds[:,0]
        if state_valid(pt):
            vertex = pt
    return vertex



def initialize_non_holonomic_actions():
    actions = np.array([[-1.,-1.], [1.,1.], [-1.,1.], [1.,-1.]])
    action_list = list(actions)
    for action in actions: action_list.append(action*0.4*np.random.random())
    for action in actions: action_list.append(action*0.1*np.random.random())
    return action_list

def simulate_non_holonomic_action(state, action):
    '''
    Returns a discretized path along which the agent moves when performing an action in a state.
    '''
    path = np.linspace(state, state+action, 10)
    return path


#####################################


def get_nearest_vertex(node_list, q_point):
    '''
    @param node_list: List of Node objects
    @param q_point: Query vertex
    @return Node in node_list with closest node.point to query q_point
    '''

    # Your Code Here
    distances = []
    for node in node_list:
        distance = math.sqrt((q_point[0] - node[0]) ** 2 + (q_point[1] - node[1]) ** 2)
        distances.append(distance)
    min_dist_index = np.argmin(distances)
    q_near = node_list[min_dist_index]
    return q_near


    # raise NotImplementedError

def steer(from_point, to_point, delta_q):
    '''
    @param from_point: Point where the path to "to_point" is originating from
    @param to_point: n-Dimensional point (array) indicating destination
    @param delta_q: Max path-length to cover, possibly resulting in changes to "to_point"
    @returns path: list of points leading from "from_node" to "to_point" (inclusive of endpoints)
    '''
    # Don't modify this function, instead change the ones it calls
    if DYNAMICS_MODE == 'holonomic':
        return steer_holonomic(from_point, to_point, delta_q)
    elif DYNAMICS_MODE == 'discrete_non_holonomic':
        return steer_discrete_non_holonomic(from_point, to_point)
    elif DYNAMICS_MODE == 'continuous_non_holonomic':
        return steer_continuous_non_holonomic(from_point, to_point)

def steer_holonomic(from_point, to_point, delta_q):
    '''
    @param from_point: Point where the path to "to_point" is originating from
    @param to_point: n-Dimensional point (array) indicating destination
    @param delta_q: Max path-length to cover, possibly resulting in changes to "to_point"
    @returns path: list of points leading from "from_node" to "to_point" (inclusive of endpoints)
    '''

    # TODO: Use a path resolution of 10 steps for computing a path between points
    
    path = []
    theta = math.atan((to_point[1] - from_point[1]) / (to_point[0] - from_point[0]))
    q_new = [from_point[0] + delta_q * math.cos(theta), from_point[1] + delta_q * math.sin(theta)]
    path.append(from_point)
    path.append(q_new)
    return path

    
def steer_discrete_non_holonomic(from_point, to_point):
    '''
    Given a fixed discrete action space and dynamics model, 
    choose the action that gets you closest to "to_point" when executing it from "from_point"

    @param from_point: Point where the path to "to_point" is originating from
    @param to_point: n-Dimensional point (array) indicating destination
    @returns path: list of points leading from "from_node" to "to_point" (inclusive of endpoints)
    '''
    # Our discrete non-holonomic action space will consist of a limited set of movement primitives
    # your code should choose an action from the actions_list and apply it for this implementation
    # of steer. You can simulate an action with simulate_non_holonomic_action(state_vector, action_vector)
    # which will give you a list of points the agent travels, starting at state_vector. 
    # Index -1 (the last element) is where the agent ends up.
    actions_list = initialize_non_holonomic_actions()
    
    raise NotImplementedError



def rrt(state_bounds, obstacles, state_is_valid, starting_point, goal_point, k, delta_q):
    '''
    @param state_bounds: matrix of min/max values for each dimension (e.g., [[0,1],[0,1]] for a 2D 1m by 1m square)
    @param state_is_valid: function that maps states (N-dimensional Real vectors) to a Boolean (indicating free vs. forbidden space)
    @param starting_point: Point within state_bounds to grow the RRT from
    @param goal_point: Point within state_bounds to target with the RRT. (OPTIONAL, can be None)
    @param k: Number of points to sample
    @param delta_q: Maximum distance allowed between vertices
    @returns List of RRT graph nodes
    '''

    node_list = []
    node_list.append(starting_point)
    for vertex in range(1, k):
        q_rand = get_random_valid_vertex(state_is_valid, state_bounds, obstacles)
        q_near = get_nearest_vertex(node_list, q_rand)
        q_new = steer(q_near, q_rand, delta_q)[1]
        node_list.append(q_new)
    return node_list


def rrt_star(state_bounds, obstacles, state_is_valid, starting_point, goal_point, k, delta_q):
    '''
    @param state_bounds: matrix of min/max values for each dimension (e.g., [[0,1],[0,1]] for a 2D 1m by 1m square)
    @param state_is_valid: function that maps states (N-dimensional Real vectors) to a Boolean (indicating free vs. forbidden space)
    @param k: Number of points to sample
    @param delta_q: Maximum distance allowed between vertices
    @returns List of RRT* graph nodes
    '''

    raise NotImplementedError


if __name__ == "__main__":
    K = 200
    ###############################
    # Problem 1a
    ###############################
    DYNAMICS_MODE = 'holonomic'
    bounds, obstacles, validity_check = setup_fixed_test_2d_world()
    starting_point = None
    starting_point = get_random_valid_vertex(validity_check, bounds, obstacles)
    nodes = rrt(bounds, obstacles, validity_check, starting_point, None, K, np.linalg.norm(bounds/10.))
    visualize_2D_graph(bounds, obstacles, nodes, None, '../figures/rrt_run1.png')

    bounds, obstacles, validity_check = setup_random_2d_world()
    starting_point = get_random_valid_vertex(validity_check, bounds, obstacles)
    nodes = rrt(bounds, obstacles, validity_check, starting_point, None, K, np.linalg.norm(bounds/10.))
    visualize_2D_graph(bounds, obstacles, nodes, None, '../figures/rrt_run2.png')

    bounds, obstacles, validity_check = setup_fixed_test_2d_world()
    starting_point = get_random_valid_vertex(validity_check, bounds, obstacles)
    goal_point = get_random_valid_vertex(validity_check, bounds, obstacles)
    while np.linalg.norm(starting_point - goal_point) < np.linalg.norm(bounds/2.):
        starting_point = get_random_valid_vertex(validity_check, bounds, obstacles)
        goal_point = get_random_valid_vertex(validity_check, bounds, obstacles)
    nodes = rrt(bounds, obstacles, validity_check, starting_point, goal_point, K, np.linalg.norm(bounds/10.))
    visualize_2D_graph(bounds, obstacles, nodes, goal_point, '../figures/rrt_run3.png')

    bounds, obstacles, validity_check = setup_random_2d_world()
    starting_point = get_random_valid_vertex(validity_check, bounds, obstacles)
    goal_point = get_random_valid_vertex(validity_check, bounds, obstacles)
    while np.linalg.norm(starting_point - goal_point) < np.linalg.norm(bounds/2.):
        starting_point = get_random_valid_vertex(validity_check, bounds, obstacles)
        goal_point = get_random_valid_vertex(validity_check, bounds, obstacles)
    nodes = rrt(bounds, obstacles, validity_check, starting_point, goal_point, K, np.linalg.norm(bounds/10.))
    visualize_2D_graph(bounds, obstacles, nodes, goal_point, '../figures/rrt_run4.png')

    ###############################
    # Problem 1b
    ###############################
    DYNAMICS_MODE = 'holonomic'
    bounds, obstacles, validity_check = setup_fixed_test_2d_world()
    starting_point = get_random_valid_vertex(validity_check, bounds, obstacles)
    goal_point = get_random_valid_vertex(validity_check, bounds, obstacles)
    while np.linalg.norm(starting_point - goal_point) < np.linalg.norm(bounds/2.):
        starting_point = get_random_valid_vertex(validity_check, bounds, obstacles)
        goal_point = get_random_valid_vertex(validity_check, bounds, obstacles)
    nodes_rrt = rrt(bounds, obstacles, validity_check, starting_point, goal_point, K, np.linalg.norm(bounds/10.))
    visualize_2D_graph(bounds, obstacles, nodes_rrt, goal_point, '../figures/rrt_comparison_run1.png')
    nodes_rrtstar = rrt_star(bounds, obstacles, validity_check, starting_point, goal_point, K, np.linalg.norm(bounds/10.))
    visualize_2D_graph(bounds, obstacles, nodes_rrtstar, goal_point, '../figures/rrt_star_run1.png')

    bounds, obstacles, validity_check = setup_fixed_test_2d_world()
    starting_point = get_random_valid_vertex(validity_check, bounds, obstacles)
    goal_point = get_random_valid_vertex(validity_check, bounds, obstacles)
    while np.linalg.norm(starting_point - goal_point) < np.linalg.norm(bounds/2.):
        starting_point = get_random_valid_vertex(validity_check, bounds, obstacles)
        goal_point = get_random_valid_vertex(validity_check, bounds, obstacles)
    nodes_rrt = rrt(bounds, obstacles, validity_check, starting_point, goal_point, K, np.linalg.norm(bounds/10.))
    visualize_2D_graph(bounds, obstacles, nodes_rrt, goal_point, '../figures/rrt_comparison_run2.png')
    nodes_rrtstar = rrt_star(bounds, obstacles, validity_check, starting_point, goal_point, K, np.linalg.norm(bounds/10.))
    visualize_2D_graph(bounds, obstacles, nodes_rrtstar, goal_point, '../figures/rrt_star_run2.png')

    ###############################
    # Problem 1c
    ###############################
    if GRAD is True:
        DYNAMICS_MODE = 'discrete_non_holonomic'
        bounds, obstacles, validity_check = setup_fixed_test_2d_world()
        starting_point = None
        starting_point = get_random_valid_vertex(validity_check, bounds, obstacles)
        nodes = rrt(bounds, obstacles, validity_check, starting_point, None, K, np.linalg.norm(bounds/10.))
        visualize_2D_graph(bounds, obstacles, nodes, None, '../figures/rrt_nh_run1.png')

        bounds, obstacles, validity_check = setup_fixed_test_2d_world()
        starting_point = get_random_valid_vertex(validity_check, bounds, obstacles)
        goal_point = get_random_valid_vertex(validity_check, bounds, obstacles)
        while np.linalg.norm(starting_point - goal_point) < np.linalg.norm(bounds/4.):
            starting_point = get_random_valid_vertex(validity_check, bounds, obstacles)
            goal_point = get_random_valid_vertex(validity_check, bounds, obstacles)
        nodes = rrt(bounds, obstacles, validity_check, starting_point, goal_point, K, np.linalg.norm(bounds/10.))
        visualize_2D_graph(bounds, obstacles, nodes, goal_point, '../figures/rrt_nh_run2.png')
