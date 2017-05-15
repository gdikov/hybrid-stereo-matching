import os
import logging

import numpy as np
import scipy.sparse as sps
import itertools as it
import spynnaker.pyNN as pyNN

from utils.config import load_config
from utils.helpers import pairs_of_neighbours

logger = logging.getLogger(__file__)


class TemporalCoincidenceDetectionNetwork:
    def __init__(self, input_sources=None, network_params=None, experiment_params=None, mode='offline'):
        """
        Args:
            input_sources: a dict of left and right view event inputs as SpikeSourceArrays 
            network_params: filename of the parameter yaml file containing 
             the neural and topology settings of the network
            experiment_params: configuration object containing the experiment-specific parameters
             such as maximum and minimum disparity, retina resolution etc.
            mode: the operation mode. Can be `offline` or `online`.
        """
        path_to_params = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "config")
        if network_params is None:
            self.params = load_config(os.path.join(path_to_params, 'default_params.yaml'))
        else:
            self.params = load_config(os.path.join(path_to_params, network_params + '_params.yaml'))
        self.max_disparity = experiment_params['max_disparity']
        self.min_disparity = experiment_params['min_disparity']
        self.retina_n_cols = experiment_params['resolution'][0]
        self.retina_n_rows = experiment_params['resolution'][1]

        if self.min_disparity > 0:
            logger.warning("Detected invalid minimum disparity of {}. "
                           "Reducing to 0, for larger values are not supported yet."
                           .format(self.min_disparity))
            self.min_disparity = 0

        disp_range = self.max_disparity - self.min_disparity
        self.size = (2 * (self.retina_n_cols - self.min_disparity) *
                     (disp_range + 1) - (disp_range + 1) ** 2 + disp_range + 1) / 2

        self._network = self._create_network(record_spikes=(mode == 'offline'),
                                             add_gating=self.params['topology']['add_gating'])

        self._connect_spike_sources(input_sources=input_sources)
        if self.params['topology']['add_uniqueness_constraint']:
            self._apply_uniqueness_constraint()
        if self.params['topology']['add_continuity_constraint']:
            self._apply_continuity_constraint()
        if self.params['topology']['add_ordering_constraint']:
            self._apply_ordering_constraint()

    def _create_network(self, record_spikes=False, add_gating=True):
        """
        Create the temporal event coincidence populations.
        
        Args:
            record_spikes: bool flag whether the spikes of the collector populations should be recorded
            add_gating: bool flag whether the blocking neural gating should be added too 

        Returns:
            a dict with keys 'collectors' containing a list of collector populations and if the blocking gates
            should be added, then a key 'blockers' contains a list of the blocker populations.
        """
        logger.info("Creating temporal coincidence detection network with {0} populations.".format(self.size))

        network = {'blockers': [], 'collectors': []}
        for pop_id in xrange(self.size):
            if add_gating:
                blocker_column = pyNN.Population(self.retina_n_rows * 2,
                                                 pyNN.IF_curr_exp,
                                                 {'tau_syn_E': self.params['neuron']['tau_E'],
                                                  'tau_syn_I': self.params['neuron']['tau_I'],
                                                  'tau_m': self.params['neuron']['tau_mem'],
                                                  'v_reset': self.params['neuron']['v_reset_blocker']},
                                                 label="Blocker_{0}".format(pop_id))
                network['blockers'].append(blocker_column)

            collector_column = pyNN.Population(self.retina_n_rows,
                                               pyNN.IF_curr_exp,
                                               {'tau_syn_E': self.params['neuron']['tau_E'],
                                                'tau_syn_I': self.params['neuron']['tau_I'],
                                                'tau_m': self.params['neuron']['tau_mem'],
                                                'v_reset': self.params['neuron']['v_reset_collector']},
                                               label="Collector_{0}".format(pop_id))
            network['collectors'].append(collector_column)

            if record_spikes:
                collector_column.record()

        if add_gating:
            self._gate_neurons(network)

        # construct network topology by filling in the diagonal in a square matrix
        #   * each diagonal represent the population ids which share the same disparity
        #   * left retina pixels are attached to each network population from the top left to the bottom left
        #       - i.e. a pixel excites all populations along a row in the matrix in which it lies too
        #   * right retina pixels are attached from the top left to the top right
        #       - i.e. a pixel excites populations along the column of the matrix to which it is tied.
        #
        # For example, a network connected to a retina with 4 columns and limited to maximum disparity of 2 would yield:
        #
        #   	   | R0  R1  R2  R3
        #       -------------------
        #       L0 | 0   .   .   .
        #       L1 | 4   1   .   .
        #       L2 | 7   5   2   .
        #       L3 | .   8   6   3
        #                 \   \   \
        #                 d=2 d=1 d=0
        self._network_topology = sps.diags(np.split(np.arange(self.size),
                                                    np.cumsum(np.arange(self.retina_n_cols,
                                                                        self.retina_n_cols - self.max_disparity, -1))),
                                           -1 * np.arange(self.max_disparity + 1)).toarray().astype(np.int)
        # invalidate entries which do not represent population ids and be careful with the 0th population
        self._network_topology[self._network_topology == 0] = -1
        self._network_topology[0, 0] = 0

        # pre-compute some useful population mappings and sets which are used in the neural connections for
        # some of the constraints and the interpreting of the output spikes in the get_output() method.
        self._same_disparity_populations = (np.diagonal(self._network_topology, i)
                                            for i in xrange(self.min_disparity, -self.max_disparity, -1))
        self._id2pixel = [0] * self.size
        self._id2disparity = [0] * self.size
        for i, population_group in enumerate(self._same_disparity_populations):
            for pixel_id, population_id in enumerate(population_group):
                self._id2pixel[population_id] = pixel_id + i
                self._id2disparity[population_id] = i

        return network

    def _gate_neurons(self, network):
        """
        Connect the blocker populations to the collectors according to the scheme described in [cite paper].
        
        Args:
            network: a dict with 'blockers' and 'collectors' lists of populations.

        Returns:
            In-place method
        """
        logger.info("Gating blocker and collector populations.")
        # generate connection lists as follows:
        #   * neurons with id from 0 until the vertical retina resolution -1 (dy - 1) serve as the left blocking neurons
        #   * and neurons with id from dy to the end, i.e. 2dy - 1, serve as the right blocking neurons
        connection_list = []
        for y in range(self.retina_n_rows):
            connection_list.append((y, y, self.params['synaptic']['wBC'], self.params['synaptic']['dBC']))
            connection_list.append((y + self.retina_n_rows, y,
                                    self.params['synaptic']['wBC'], self.params['synaptic']['dBC']))

        logger.debug("Generated gating connection list: {}".format(connection_list))
        # connect the inhibitory neurons (blockers) to the cell output (collector) neurons
        for blockers, collectors in zip(network['blockers'], network['collectors']):
            pyNN.Projection(blockers, collectors, pyNN.FromListConnector(connection_list), target='inhibitory')
        logger.debug("Gating blocker and collector populations completed.")

    def _connect_spike_sources(self, input_sources=None):
        """
        Connect the SpikeSourceArrays to the collector and, if created, to the blocker neural populations.
        
        Args:
            input_sources: a dict with 'left' and 'right' keys containing the left and right retina respectively

        Returns:
            In-place method
        """
        logger.info("Connecting spike sources to the temporal coincidence detection network.")

        add_blockers = len(self._network['blockers']) > 0
        n_rows = self.retina_n_rows
        if add_blockers:
            # left is 0--dimensionRetinaY-1; right is dimensionRetinaY--dimensionRetinaY*2-1
            syn_colateral = [(self.params['synapse']['wSaB'], self.params['synapse']['dSaB'])] * n_rows
            syn_contralateral = [(self.params['synapse']['wSzB'], self.params['synapse']['dSzB'])] * n_rows
            id_map = [(x, x) for x in range(n_rows)]
            offset_map = [(x, x + n_rows) for x in range(n_rows)]
            # unpack the tuples of tuples to tuples of flattened elements
            # NOTE: ret_l_block_l is identical to rel_r_block_l and ret_r_block_r is identical to ret_l_block_r
            # if and only if the contra-lateral and co-lateral synapses have the same weights and delays.
            if self.params['synapse']['wSaB'] == self.params['synapse']['wSzB'] and \
               self.params['synapse']['dSaB'] == self.params['synapse']['dSzB']:
                ret_l_block_l = [(x[0][0], x[0][1], x[1][0], x[1][1]) for x in zip(id_map, syn_colateral)]
                ret_l_block_r = [(x[0][0], x[0][1], x[1][0], x[1][1]) for x in zip(offset_map, syn_contralateral)]
                ret_r_block_l = ret_l_block_l
                ret_r_block_r = ret_l_block_r
            else:
                ret_l_block_l = [(x[0][0], x[0][1], x[1][0], x[1][1]) for x in zip(id_map, syn_colateral)]
                ret_l_block_r = [(x[0][0], x[0][1], x[1][0], x[1][1]) for x in zip(offset_map, syn_contralateral)]
                ret_r_block_l = [(x[0][0], x[0][1], x[1][0], x[1][1]) for x in zip(id_map, syn_contralateral)]
                ret_r_block_r = [(x[0][0], x[0][1], x[1][0], x[1][1]) for x in zip(offset_map, syn_colateral)]
            logger.debug("Generated connection list from left retina to left blockers: {}".format(ret_l_block_l))
            logger.debug("Generated connection list from left retina to right blockers: {}".format(ret_l_block_r))
            logger.debug("Generated connection list from right retina to right blockers: {}".format(ret_r_block_r))
            logger.debug("Generated connection list from right retina to left blockers: {}".format(ret_r_block_l))
        else:
            ret_l_block_l = ret_l_block_r = ret_r_block_l = ret_r_block_r = None

        retina_left = input_sources['left']
        retina_right = input_sources['right']

        def connect_retina(retina, network_topology_matrix, block_l=None, block_r=None):
            # since the network topology is a lower triangular matrix, iterating over the rows and removing the
            # artificially added -1 padding elements, yields the population ids in the network to which
            # a pixel from the left retina should be connected. The same is repeated for the right retina
            # but this time the network topology matrix is iterated along the columns.
            for pixel_id, row in enumerate(network_topology_matrix):
                for population_id in row[row >= 0]:
                    pyNN.Projection(presynaptic_population=retina[pixel_id],
                                    postsynaptic_population=self._network['collectors'][population_id],
                                    connector=pyNN.OneToOneConnector(weights=self.params['synapse']['wSC'],
                                                                     delays=self.params['synapse']['dSC']),
                                    target='excitatory')
                    if block_l is not None and block_r is not None:
                        pyNN.Projection(presynaptic_population=retina[pixel_id],
                                        postsynaptic_population=self._network['blockers'][population_id],
                                        connector=pyNN.FromListConnector(block_l),
                                        target='excitatory')
                        pyNN.Projection(presynaptic_population=retina[pixel_id],
                                        postsynaptic_population=self._network['blockers'][population_id],
                                        connector=pyNN.FromListConnector(block_r),
                                        target='inhibitory')

        connect_retina(retina_left, self._network_topology, ret_l_block_l, ret_l_block_r)
        connect_retina(retina_right, self._network_topology.T, ret_r_block_l, ret_r_block_r)
        logger.debug("Connecting spike sources to the temporal coincidence detection network completed.")

    def _apply_uniqueness_constraint(self):
        """
        Implement the David Marr's uniqueness constraint which prohibits the spiking of multiple 
        disparity sensitive neurons which correspond to the same pixel, i.e. a physical point should
        be assigned to at most one depth value.
        
        Returns:
            In-place method
        """
        logger.info("Applying the uniqueness constraint on the temporal coincidence network.")

        def inhibit_along_eyesight(network_topology):
            for population_ids in network_topology:
                # generate population id pairs from all populations lying along the projection axis
                pairs = filter(lambda x: x[0] != x[1], it.product(population_ids[population_ids >= 0], repeat=2))
                logger.debug("Generated inhibitory connection list for populations {}".format(pairs))
                for presynaptic, postsynaptic in pairs:
                    pyNN.Projection(presynaptic_population=self._network['collectors'][presynaptic],
                                    postsynaptic_population=self._network['collectors'][postsynaptic],
                                    connector=pyNN.OneToOneConnector(weights=self.params['synapse']['wCCi'],
                                                                     delays=self.params['synapse']['dCCi']),
                                    target='inhibitory')

        # connect for inhibition for the left retina
        inhibit_along_eyesight(self._network_topology)
        # and for the right
        inhibit_along_eyesight(self._network_topology.T)
        logger.debug("Applying the uniqueness constraint on the temporal coincidence network completed.")

    def _apply_continuity_constraint(self):
        """
        Implement the David Marr's continuity constraint which encourages the spiking of disparity sensitive neurons 
        which lie in the same disparity map. This is backed by the assumption that physical object are coherent 
        and disparity does not change by much (if at all).
        
        Returns:
            In-place method
        """
        logger.info("Applying the continuity constraint on the temporal coincidence network.")
        logger.warning("The current implementation supports only cross-like connection patterns, "
                       "i.e. a neuron will excite only neurons to the left, right, top and bottom. ")

        if self.params['topology']['radius_continuity'] > self.retina_n_cols:
            new_radius = 1
            logger.warning("Radius of excitation is too big. Setting radius to {}".format(new_radius))
            self.params['topology']['radius_continuity'] = new_radius

        logger.debug("Same-disparity population ids: {}".format(list(self._same_disparity_populations)))

        # iterate over population or neural ids and construct pairs from neighbouring units
        for population_ids in self._same_disparity_populations:
            for presynaptic, postsynaptic in pairs_of_neighbours(population_ids,
                                                                 window_size=self.params['topology']['radius_continuity']+1,
                                                                 add_reciprocal=True):
                pyNN.Projection(presynaptic_population=self._network['collectors'][presynaptic],
                                postsynaptic_population=self._network['collectors'][postsynaptic],
                                connector=pyNN.OneToOneConnector(weights=self.params['synapse']['wCCe'],
                                                                 delays=self.params['synapse']['dCCe']),
                                target='excitatory')

        # construct vertical connections within each neural population
        within_population_neuron_pairs = pairs_of_neighbours(range(self.retina_n_rows),
                                                             window_size=self.params['topology']['radius_continuity']+1,
                                                             add_reciprocal=True)
        logger.debug("Within-population neuron pairs: {}".format(within_population_neuron_pairs))

        connection_list = [(src, dst, self.params['synapse']['wCCe'], self.params['synapse']['dCCe'])
                           for src, dst in within_population_neuron_pairs]
        for population in self._network['collectors']:
            pyNN.Projection(presynaptic_population=population, postsynaptic_population=population,
                            connector=pyNN.FromListConnector(connection_list),
                            target='excitatory')
        logger.debug("Applying the continuity constraint on the temporal coincidence network completed.")

    def _apply_ordering_constraint(self):
        """
        Implement the proposed by Christoph ordering constraint of inhibiting possible *mirror* matches. 
        For example: | |a|b|c| - | |1|2|3|  -->  (a,3), (b,2), (c,1) could be sometimes prevented if (b,2)
        would inhibit (a,1) and (c,3) event types. These lie on the flipped diagonal of each population id.
         
        Returns:
            In-place method
        """
        logger.info("Applying the ordering constraint on the temporal coincidence network.")

        # get all flipped diagonal population groups
        all_mirrored_diags = [self._network_topology[::-1, :].diagonal(i) for i in xrange(-self.max_disparity - 1,
                                                                                          self.max_disparity + 2)]
        valid_mirrored_diags = [group for group in [d[d >= 0] for d in all_mirrored_diags] if len(group) > 1]
        for pair in valid_mirrored_diags:
            for presynaptic, postsynaptic in filter(lambda x: x[0] != x[1], it.product(pair, repeat=2)):
                pyNN.Projection(presynaptic_population=self._network['collectors'][presynaptic],
                                postsynaptic_population=self._network['collectors'][postsynaptic],
                                connector=pyNN.OneToOneConnector(weights=self.params['synapse']['wCCo'],
                                                                 delays=self.params['synapse']['dCCo']),
                                target='inhibitory')
        logger.debug("Applying the ordering constraint on the temporal coincidence network completed.")

    def get_raw_output(self):
        """
        Get the spikes for each neuron and all populations.
        
        Returns:
            A list of population spikes. In each population a neuron id and spiking timestamp are recorded. 
            The population id corresponds to the index in the list.
        """
        logger.debug("Fetching raw output from temporal coincidence network.")
        spikes_per_population = [x.getSpikes() for x in self._network['collectors']]
        return spikes_per_population

    def get_output(self):
        """
        Return a the interpreted raw output (i.e. ocnverted to pixel coordinates and disparity values)

        Returns:
            A numpy array, representing the network activity
        """

        spikes_per_population = self.get_raw_output()
        logger.debug("Converting raw temporal coincidence network output to timestamped "
                     "pixel coordinates and disparities events.")
        spikes_in_pixeldisp_space = {'ts': [], 'xs': [], 'ys': [], 'disps': []}
        for population_id, spikes_in_population in enumerate(spikes_per_population):
            for spike in spikes_in_population:
                spikes_in_pixeldisp_space['ts'].append(spike[1])
                spikes_in_pixeldisp_space['xs'].append(self._id2pixel[population_id])
                spikes_in_pixeldisp_space['ys'].append(int(spike[0]))
                spikes_in_pixeldisp_space['disps'].append(self._id2disparity[population_id])

        return spikes_in_pixeldisp_space
