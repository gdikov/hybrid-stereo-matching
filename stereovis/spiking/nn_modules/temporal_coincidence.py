import os
import logging

import numpy as np
import scipy.sparse as sps
import itertools as it
import spynnaker.pyNN as pyNN

from utils.params import load_params

logger = logging.getLogger(__file__)


class TemporalCoincidenceDetectionNetwork:
    def __init__(self, input_sources=None, params=None, mode='offline'):
        """
        Args:
            input_sources: a dict of left and right view event inputs as SpikeSourceArrays 
            params: filename of the parameter yaml file containing the neural and topological settings of the network
            mode: the operation mode. Can be `offline` or `online`.
        """
        path_to_params = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "config")
        if params is None:
            self.params = load_params(os.path.join(path_to_params, 'default_params.yaml'))
        else:
            self.params = load_params(os.path.join(path_to_params, params + '_params.yaml'))

        if self.params['topological']['min_disparity'] > 0:
            logger.warning("Detected invalid minimum disparity of {}. "
                           "Reducing to 0, for larger values are not supported yet."
                           .format(self.params['topological']['min_disparity']))
            self.params['topological']['min_disparity'] = 0

        disp_range = self.params['topological']['max_disparity'] - self.params['topological']['min_disparity']
        self.size = (2 * (self.params['topological']['n_cols'] - self.params['topological']['min_disparity']) *
                     (disp_range + 1) - (disp_range + 1) ** 2 + disp_range + 1) / 2

        if mode == 'offline':
            self._network = self._create_network(record_spikes=True,
                                                 add_gating=self.params['topological']['add_gating'])
        else:
            self._network = self._create_network(record_spikes=False,
                                                 add_gating=self.params['topological']['add_gating'])

        self._connect_spike_sources(input_sources=input_sources)
        if self.params['topological']['add_uniqueness_constraint']:
            self._apply_uniqueness_constraint()
        if self.params['topological']['add_continuity_constraint']:
            self._apply_continuity_constraint()

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
        for x in range(0, self.size):
            if add_gating:
                blocker_columns = pyNN.Population(self.params['topological']['n_rows'] * 2,
                                                  pyNN.IF_curr_exp,
                                                  {'tau_syn_E': self.params['neural']['tau_E'],
                                                   'tau_syn_I': self.params['neural']['tau_I'],
                                                   'tau_m': self.params['neural']['tau_mem'],
                                                   'v_reset': self.params['neural']['v_reset_blocker']},
                                                  label="Blocker {0}".format(x))
                network['blockers'].append(blocker_columns)

            collector_column = pyNN.Population(self.params['topological']['n_rows'],
                                               pyNN.IF_curr_exp,
                                               {'tau_syn_E': self.params['neural']['tau_E'],
                                                'tau_syn_I': self.params['neural']['tau_I'],
                                                'tau_m': self.params['neural']['tau_mem'],
                                                'v_reset': self.params['neural']['v_reset_collector']},
                                               label="Collector {0}".format(x))
            network['collectors'].append(collector_column)

            if record_spikes:
                collector_column.record()

        if add_gating:
            self._gate_neurons(network)

        self._network_topology = sps.diags([np.arange(k, k + self.params['topological']['n_cols'] - x)
                                            for x, k in enumerate(range(self.params['topological']['max_disparity']))],
                                           [-x for x in range(self.params['topological']['max_disparity'])],
                                           dtype=np.int).todense()
        self._network_topology[self._network_topology == 0] = -1

        return network

    def _gate_neurons(self, network):
        """
        Connect the blocker populations to the collectors according to the scheme described in [cite paper].
        
        Args:
            network: a dict with 'blockers' and 'collectors' lists of populations.

        Returns:
            In-place method
        """
        logger.info("Gating blocker and collector populations initiated.")
        # generate connection lists as follows:
        #   * neurons with id from 0 until the vertical retina resolution -1 (dy - 1) serve as the left blocking neurons
        #   * and neurons with id from dy to the end, i.e. 2dy - 1, serve as the right blocking neurons
        connection_list = []
        for y in range(0, self.params['topological']['n_rows']):
            connection_list.append((y, y, self.params['synaptic']['wBC'], self.params['synaptic']['dBC']))
            connection_list.append((y + self.params['topological']['n_rows'], y,
                                    self.params['synaptic']['wBC'], self.params['synaptic']['dBC']))

        logger.debug("Generated gating connection list: {}".format(connection_list))
        # connect the inhibitory neurons (blockers) to the cell output (collector) neurons
        for blockers, collectors in zip(network['blockers'], network['collectors']):
            pyNN.Projection(blockers, collectors, pyNN.FromListConnector(connection_list), target='inhibitory')
        logger.info("Gating blocker and collector populations completed.")

    def _connect_spike_sources(self, input_sources=None):
        """
        Connect the SpikeSourceArrays to the collector and, if created, to the blocker neural populations.
        
        Args:
            input_sources: a dict with 'left' and 'right' keys containing the left and right retina respectively

        Returns:
            In-place method
        """
        logger.info("Connecting spike sources to the temporal coincidence detection network initiated.")

        add_blockers = 'blockers' in self._network.keys()
        n_rows = self.params['topological']['n_rows']
        if add_blockers:
            # left is 0--dimensionRetinaY-1; right is dimensionRetinaY--dimensionRetinaY*2-1
            syn_colateral = [(self.params['synaptic']['wSaB'], self.params['synaptic']['dSaB'])] * n_rows
            syn_contralateral = [(self.params['synaptic']['wSzB'], self.params['synaptic']['dSzB'])] * n_rows
            id_map = [(x, x) for x in range(n_rows)]
            offset_map = [(x, x + self.params['topological']['n_rows']) for x in range(n_rows)]
            # unpack the tuples of tuples to tuples of flattened elements
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
                                    connector=pyNN.OneToOneConnector(weights=self.params['synaptic']['wSC'],
                                                                     delays=self.params['synaptic']['dSC']),
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
        logger.info("Connecting spike sources to the temporal coincidence detection network completed.")

    def _apply_uniqueness_constraint(self):
        """
        Implement the David Marr's uniqueness constraint which prohibits the spiking of multiple 
        disparity sensitive neurons which correspond to the same pixel, i.e. a physical point should
        be assigned to at most one depth value.
        
        Returns:
            In-place method
        """
        logger.info("Applying the uniqueness constraint on the temporal coincidence network.")

        if self.params['topological']['radius_i'] < self.params['retina']['n_cols']:
            new_radius = self.params['retina']['n_cols']
            logger.warning("Radius of inhibition was too small! Uniquness constraint cannot be satisfied. "
                           "Setting radius to {}".format(new_radius))
            self.params['topological']['radius_i'] = new_radius

        def inhibit_along_eyesight(network_topology):
            for population_ids in network_topology:
                # generate population id pairs from all populations lying along the projection axis
                pairs = filter(lambda x: x[0] != x[1], it.product(population_ids, repeat=2))
                logger.debug("Generated inhibitory connection list for populations {}".format(pairs))
                for presynaptic, postsynaptic in pairs:
                    pyNN.Projection(self._network['collectors'][presynaptic],
                                    self._network['collectors'][postsynaptic],
                                    pyNN.OneToOneConnector(weights=self.params['synaptic']['wCCi'],
                                                           delays=self.params['synaptic']['dCCi']),
                                    target='inhibitory')

        # connect for inhibition for the left retina
        inhibit_along_eyesight(self._network_topology)
        # and for the right
        inhibit_along_eyesight(self._network_topology.T)
        logger.info("Applying the uniqueness constraint on the temporal coincidence network completed.")

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
                       "i.e. a neuron will excite only neurons to the left, right, top and bottom. "
                       "This may be improved in the future.")

        if self.params['topological']['radius_e'] > self.params['retina']['n_cols']:
            new_radius = 1
            logger.warning("Radius of excitation is too big. Setting radius to {}".format(new_radius))
            self.params['topological']['radius_e'] = new_radius

        def nwise(iterable, n=2):
            iters = it.tee(iterable, n)
            for i, obj in enumerate(iters):
                next(it.islice(obj, i, i), None)
            return it.izip(*iters)

        def excite_neighbours(ids):
            # iterate over population or neural ids and construct pairs from neighbouring units
            for unit_id in ids:
                # pairs of population id to be connected for excitation within a disparity map
                pairs = list(set(sum([list(it.combinations(x, 2))
                                      for x in nwise(unit_id, n=self.params['topological']['radius_e'])], [])))
                # add the reciprocal connections too
                pairs = pairs + map(lambda y: (y[1], y[0]), pairs)
                return pairs

        same_disparity_populations = (np.diagonal(self._network_topology, i)
                                      for i in xrange(self.params['topology']['min_disparity'],
                                                      -self.params['topology']['max_disparity'], -1))
        logger.debug("Same-disparity population ids: {}".format(same_disparity_populations))

        for population_ids in same_disparity_populations:
            for presynaptic, postsynaptic in excite_neighbours(population_ids):
                pyNN.Projection(self._network['collectors'][presynaptic],
                                self._network['collectors'][postsynaptic],
                                pyNN.OneToOneConnector(weights=self.params['synaptic']['wCCe'],
                                                       delays=self.params['synaptic']['dCCe']),
                                target='excitatory')

        within_population_neuron_pairs = excite_neighbours(range(self.params['retina']['n_rows']))
        logger.debug("Within-population neuron pairs: {}".format(within_population_neuron_pairs))

        for population in self._network['collectors']:
            pyNN.Projection(presynaptic_population=population, postsynaptic_population=population,
                            connector=pyNN.FromListConnector(within_population_neuron_pairs),
                            target='excitatory')
        logger.info("Applying the continuity constraint on the temporal coincidence network completed.")
