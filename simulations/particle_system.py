#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import logging
import networkx as nx
import random
import  matplotlib.pyplot as plt
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


class Graph(object):
    def __init__(self, nx_graph):
        self.graph = nx_graph
        self.node_size = 800
        self.node_color = '#0D0D0D'
        self.font_color = '#D9D9D9'
        self.edge_color = '#262626'

    def load_graph(self, path="data/graph.edgelist"):
        self.graph = nx.read_edgelist(path,
                                      create_using=nx.DiGraph)

    def add_node_to_graph(self, node):
        logging.debug(f'*** Graph: Adding a node {node}')
        self.graph.add_node(node, value=np.random.randn())

    def add_an_edge_to_graph(self, node_a, node_b, weight=None):
        # edge is a linear function v(node_b) = v(node_a) * _w + _c
        _w = weight if weight else random.choice([1, -1]) * np.random.normal(2, 0.5, 1)
        _c = np.random.normal(0, 1, 1)
        logging.info(f'*** Graph: Adding an edge {node_a}-{node_b}:{weight}')
        self.graph.add_edge(node_a, node_b, color=self.edge_color,
                            weight=_w.item(0),
                            capacity=_c.item(0))

    def remove_an_edge_from_graph(self, node_a, node_b):
        logging.debug(f'*** Graph: Removing an edge {node_a}-{node_b}')
        if self.graph.has_edge(node_a, node_b):
            logging.debug(f'*** Graph: Removed an edge {node_a}-{node_b}')
            self.graph.remove_edge(node_a, node_b)
            return True
        logging.debug(f'*** Graph: No edge {node_a}-{node_b} found')
        return False

    def get_graph(self):
        return self.graph

    def get_total_nodes(self):
        return self.graph.number_of_nodes()

    def get_node_values(self):
        values = nx.get_node_attributes(self.graph, 'value').values()
        return values

    def get_edge_value(self, node_a, node_b):
        return self.graph.get_edge_data(node_a, node_b)

    def get_successors(self, node):
        return list(self.graph.successors(node))

    def get_node_value(self, node):
        _nodes = self.graph.nodes()
        return _nodes[node]['value']

    def set_node_value(self, node, value):
        attrs = {node: {'value': value}}
        nx.set_node_attributes(self.graph, attrs)

    def get_random_node(self):
        node = None
        if self.graph.number_of_nodes() != 0:
            node = random.sample(self.graph.nodes(), 1)[0]
        return node

    def reset_all_nodes(self):
        # TODO:reset all node values and edge values.
        for node in self.get_nodes():
            self.set_node_value(node, np.random.randn()/1000)

    def get_source_nodes(self):
        _nodes = []
        for node in self.graph.nodes():
            if self.graph.in_degree(node) == 0:
                _nodes.append(node)
        return _nodes

    def get_nodes(self):
        return self.graph.nodes()

    def get_all_parents(self):
        _parents = []
        for node in self.graph.nodes():
            if self.graph.out_degree(node) > 0:
                _parents.append(node)
        return _parents

    @classmethod
    def draw_graph(cls, _graph, axes):
        nx.draw(_graph, nx.circular_layout(_graph),
                with_labels=True,
                node_size=500,
                ax=axes)


class ParticleGraph(Graph):
    """
    Keeps track of particles and springs as a graph system.
    :param
    """
    def __init__(self):
        super().__init__(nx.DiGraph())
        self.particle_count = 0
        self.spring_count = 0

    def __repr__(self):
        return self.get_graph()

    def get_node_names(self):
        return self.get_nodes()

    def get_total_number_of_particles(self):
        return self.particle_count

    def get_total_number_of_springs(self):
        return self.spring_count

    def add_particle_node_to_graph(self, name=None):
        _node = name if name else f'p_{self.get_total_nodes()}'
        logging.debug(f'*** ParticleGraph: Adding a node to graph {name}')
        self.add_node_to_graph(_node)
        self.particle_count += 1
        return _node

    def add_spring_to_graph(self, particle_a, particle_b, spring_constant):
        x = particle_a
        y = particle_b
        logging.debug(f'*** ParticleGraph: Adding a spring {particle_a}-{particle_b}:{spring_constant}')
        self.spring_count += 1
        self.add_an_edge_to_graph(x, y, weight=spring_constant)

    def remove_spring_from_graph(self, node_a, node_b):
        if self.remove_an_edge_from_graph(node_a=node_a, node_b=node_b):
            self.spring_count -= 1
            return True
        return False

    def add_springs_to_graph(self, spring_constant_matrix):
        # Since spring constant matrix is symmetric nullify lower half
        spring_constant_matrix = np.tril(spring_constant_matrix, k=0)
        for i in range(self.particle_count):
            for j in range(self.particle_count):
                if spring_constant_matrix[i][j] != 0:
                    logging.info(f'*** ParticleGraph: Adding spring between particle_{i} and particle_{j} with k={spring_constant_matrix[i][j]} ')
                    self.add_spring_to_graph(particle_a=f'p_{i}',
                                             particle_b=f'p_{j}',
                                             spring_constant=spring_constant_matrix[i][j])

    def show(self):
        colors = nx.get_edge_attributes(self.graph, 'color').values()
        weights = nx.get_edge_attributes(self.graph, 'weight').values()
        nx.draw(self.graph,
                pos=nx.circular_layout(self.graph),
                with_labels=True,
                edge_color=list(colors),
                width=list(weights),
                node_size=500)
        plt.show()

    def draw(self, axes):
        colors = nx.get_edge_attributes(self.graph, 'color').values()
        weights = nx.get_edge_attributes(self.graph, 'weight').values()
        nx.draw(self.graph,
                pos=nx.circular_layout(self.graph),
                with_labels=True,
                edge_color=list(colors),
                width=list(weights),
                node_size=500,
                ax=axes)


class Environment(object):
    def __init__(self):
        self.box_size = 5.0
        self._delta_T = 0.01
        self.dimensions = 2
        self._positions = []
        self._velocities = []

    def reset(self):
        self._positions.clear()
        self._velocities.clear()

    def add_a_particle(self):
        pass

    def get_positions(self):
        return self._positions

    def get_velocities(self):
        return self._velocities


class SpringSystem(Environment):
    def __init__(self):
        super().__init__()
        self.p_graph = ParticleGraph()
        self.noise_variance = 0.5
        self.init_velocity_mean_sd = (0.01, 1)
        self.k = np.asarray([])
        self.num_particles = 0
        self.noise = (0.5, 1)
        # In space the periodic interactions between particles go on to infinity
        # In real world however system would loose energy slows down by damp_factor.
        # (rapid) 0.0 < damp < 1.0 (no damp)
        self.damp_factor = 1.0

    def set_initial_velocity_mean_sd(self, m_sd):
        self.init_velocity_mean_sd = m_sd

    def set_damp_factor(self, value):
        self.damp_factor = value

    def set_noise_mean_sd(self, m_sd):
        self.noise = m_sd

    def get_particle_names(self):
        return self.p_graph.get_node_names()

    def get_particles_count(self):
        return self.num_particles

    def get_column_names(self):
        self.p_graph.get_node_names()

    def add_particles(self, num_of_particles=0):
        logging.debug(f'*** SpringSystem: Creating a spring particle system with {num_of_particles} particles')
        for _ in range(num_of_particles):
            self.p_graph.add_particle_node_to_graph()
        self.num_particles = self.p_graph.get_total_number_of_particles()
        logging.debug(f'*** SpringSystem: Initialized springs to 0.0')
        # initialize springs
        self.k = np.zeros((self.num_particles, self.num_particles))
        logging.info(f'*** SpringSystem: Created a spring particle system with {num_of_particles} particles')

    def show_graph(self):
        self.p_graph.show()

    def add_a_spring(self, particle_a, particle_b, spring_constant):
        num_of_particles = self.p_graph.get_total_number_of_particles()
        if num_of_particles == 0:
            logging.error('*** SpringSystem: Environment has no particles to add a spring')
            return

        self.k[particle_a][particle_b] = spring_constant
        self.k[particle_b][particle_a] = spring_constant
        self.p_graph.add_spring_to_graph(particle_a=particle_a,
                                         particle_b=particle_b,
                                         spring_constant=spring_constant)
        logging.info(f'*** SpringSystem: Added spring to a {particle_a} {particle_b} : {spring_constant}')

    def add_springs(self, spring_constants_matrix):

        num_of_particles = self.p_graph.get_total_number_of_particles()

        if num_of_particles == 0:
            logging.error('*** SpringSystem: Environment has no particles to add a spring')
            return

        if spring_constants_matrix.shape != (num_of_particles, num_of_particles):
            logging.error('*** SpringSystem: Shapes of spring constants matrix and number of particles wont match')
            return

        # Establish symmetry
        spring_constants_matrix = np.tril(spring_constants_matrix) + np.tril(spring_constants_matrix, -1).T

        # Nullify self interaction or causality
        np.fill_diagonal(spring_constants_matrix, 0)
        self.k = spring_constants_matrix
        self.p_graph.add_springs_to_graph(spring_constant_matrix=self.k)
        logging.info(f'*** SpringSystem: Added springs to a spring particle system')

    def remove_spring(self, particle_a, particle_b):
        self.k[particle_a][particle_b] = 0.0
        self.p_graph.remove_spring_from_graph(node_a=particle_a, node_b=particle_b)
        logging.info(f'*** SpringSystem: Removed Spring p_{particle_a} p_{particle_b}')

    def remove_all_springs(self):
        num_of_particles = self.p_graph.get_total_number_of_particles()
        self.k = np.zeros(shape=(num_of_particles, num_of_particles))
        particle_names = self.p_graph.get_node_names()
        for particle_a in particle_names:
            for particle_b in particle_names:
                if particle_a != particle_b:
                    self.p_graph.remove_spring_from_graph(node_a=particle_a,
                                                          node_b=particle_b)

    def simulate(self, total_time_steps, period, sample_freq, observations, spring_observations, traj_id):
        num_particles = self.p_graph.get_total_number_of_particles()
        if num_particles == 0:
            logging.warning('*** SpringSystem: Nothing to simulate, add particles')
            return

        def get_init_pos_velocity():
            """
            This function samples position and velocity from a distribution.
            These position and velocity will be used as
            initial position and velocity for all particles.
            :return: initial position and velocity
            """
            vel_norm = 0.5
            loc_std = 0.5
            # start at origin
            _position = np.random.randn(2, num_particles) * loc_std
            # sample initial velocity from normal distribution
            _mv = np.random.normal(self.init_velocity_mean_sd[0], 0.01, 1)
            logging.info(f'*** SpringSystem: xInitial velocity set to {_mv}')
            _velocity = (_mv + np.random.randn(2, num_particles)) * self.init_velocity_mean_sd[1]
            # Compute magnitude of this velocity vector and format to right shape
            #v_norm = np.linalg.norm(_position, axis=0)
            # Scale by magnitude
            #_velocity = _position * vel_norm / v_norm
            return _position, _velocity

        def get_force1(_edges, current_positions):
            """
            :param _edges: Adjacency matrix representing mutual causality
            :param current_positions: current coordinates of all particles
            :return: net forces acting on all particles.
            """
            force_matrix = - 0.5 * _edges
            np.fill_diagonal(force_matrix, 0)
            x_cords, y_cords = current_positions[0, :], current_positions[1, :]

            #x_diffs = np.subtract.outer(x_cords, x_cords)
            #y_diffs = np.subtract.outer(y_cords, y_cords)
            #distance_matrix = np.sqrt(np.square(x_diffs) + np.square(y_diffs))
            # By Hooke's law Force = -k * dx
            #force_matrix = np.multiply(-0.5 * _edges, distance_matrix)

            x_diffs = np.subtract.outer(x_cords, x_cords).reshape(1, self.num_particles, self.num_particles)
            y_diffs = np.subtract.outer(y_cords, y_cords).reshape(1, self.num_particles, self.num_particles)
            force_matrix = force_matrix.reshape(1, self.num_particles, self.num_particles)
            _force = (force_matrix * np.concatenate((x_diffs, y_diffs))).sum(axis=-1)
            return _force

        # Initialize the first position and velocity from a distribution
        init_position, init_velocity = get_init_pos_velocity()

        # Compute initial forces between particles.
        init_force_between_particles = get_force1(self.k, init_position)

        # Compute new velocity.
        '''
        F = m * a, for unit mass force is acceleration
        F = a = dv/dt
        dv = dt * F
        current_velocity - velocity = dt * F
        velocity = current_velocity - (self._delta_T * F)
        '''
        velocity = init_velocity + (self._delta_T * init_force_between_particles)
        current_position = init_position
        step = 0
        original_springs = self.k
        flip_flag = True
        for i in range(total_time_steps):
            # Compute forces between particles
            force_between_particles = get_force1(self.k, current_position)

            # Compute new velocity based on current velocity and forces between particles.
            new_velocity = velocity + (self._delta_T * force_between_particles)

            # Compute new position based on current velocity and positions.
            # dx/dt = v
            # (current_position - new_position) = dt * v
            new_position = current_position + (self._delta_T * new_velocity)

            # Update velocity and position
            velocity = new_velocity
            current_position = new_position
            # Add noise to observations
            #current_position += np.random.randn(2, self.num_particles) * self.noise_variance
            #velocity += np.random.randn(2, self.num_particles) * self.noise_variance
            # add to observations

            if i % sample_freq == 0:
                self.add_observation(observations, spring_observations, traj_id, step, self.k, current_position, velocity)
                step += 1

            if (period != 0) and (i % period == 0):
                if flip_flag:
                    logging.info(f'*** Simulation: Step {i} Flipping edges (Removing)')
                    self.remove_all_springs()
                    flip_flag = False
                else:
                    logging.info(f'*** Simulation: Step {i} Flipping edges (Restoring)')
                    self.add_springs(spring_constants_matrix=original_springs)
                    flip_flag = True

    def add_observation(self, observations, spring_observations, traj_id, step, springs, positions, velocity):
        # local dict to collect readings
        observation = {}
        sp_observation = {}

        # Adding all positions, velocity
        for i in range(len(positions)):
            for j in range(len(positions[0])):
                particle_id = j
                if i == 0:
                    # x axis
                    observation[f'p_{particle_id}_x_position'] = positions[i][j]
                    observation[f'p_{particle_id}_x_velocity'] = velocity[i][j]
                else:
                    # y axis
                    observation[f'p_{particle_id}_y_position'] = positions[i][j]
                    observation[f'p_{particle_id}_y_velocity'] = velocity[i][j]

        # Calculate and store distance
        """
        x_cords, y_cords = positions[0, :], positions[1, :]
        x_diffs = np.subtract.outer(x_cords, x_cords)
        y_diffs = np.subtract.outer(y_cords, y_cords)
        distance_matrix = np.sqrt(np.square(x_diffs) + np.square(y_diffs))
        for i in range(len(distance_matrix)):
            for j in range(len(distance_matrix[0])):
                observation[f'p_{i}_{j}_distance'] = distance_matrix[i][j]
        """
        for i in range(self.num_particles):
            for j in range(self.num_particles):
                if i != j:
                    sp_observation[f's_{i}_{j}'] = springs[i][j]

        observation[f'trajectory_step'] = f'{traj_id}_{step}'
        sp_observation[f'trajectory_step'] = f'{traj_id}_{step}'
        # Add observation to global record
        observations.add_an_observation(observation)
        spring_observations.add_an_observation(sp_observation)

def test():
    sp = SpringSystem()
    sp.add_particles(num_of_particles=3)
    spring_constants_matrix = np.asarray([[0, 0, 1],
                                          [0, 0, 0],
                                          [1, 0, 0]])
    #spring_constants_matrix = np.random.rand(5, 5)
    sp.add_springs(spring_constants_matrix=spring_constants_matrix)
    sp.show_graph()


if __name__ == "__main__":
    test()
