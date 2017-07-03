import time
import numpy as np
import logging

logger = logging.getLogger(__file__)


class StereoMRF(object):
    """
    Markov Random Field with loopy belief propagation (min-sum message passing). 
    """
    def __init__(self, dim, n_levels):
        self.n_levels = n_levels
        self.dimension = (n_levels,) + dim
        self._message_field = {'south': np.zeros(self.dimension, dtype=np.float32),
                               'west': np.zeros(self.dimension, dtype=np.float32),
                               'north': np.zeros(self.dimension, dtype=np.float32),
                               'east': np.zeros(self.dimension, dtype=np.float32),
                               'data': np.zeros(self.dimension, dtype=np.float32)}

    def _init_fields(self, image_left, image_right, prior=None, prior_trust_factor=1.0,
                     prior_influence_mode='adaptive', reinit_messages=True):
        """
        Initialise the message fields -- each hidden variable contains 5 message boxes from the 4 adjacent variables 
        (south, west, north, east) and the observed variable (data).
        
        Args:
            image_left: a numpy array representing the left image (in grayscale with values in [0, 1])
            image_right: a numpy array with the same structure and shape as left image, representing the right image 
            prior: a numpy array with the same shape as left image, providing alternative source of truth (or initial
                belief) about some pixels' corresponding disparity values.
            prior_trust_factor: float in [0, 1] telling how much the prior should be trusted.
            prior_influence_mode: str, how should prior be incorporated: `const` for a constant prior trust factor,
                `adaptive` for one that is proportional to the data score.
            reinit_messages: whether the messages field should be reset to 0

        Returns:
            In-place method
        """
        assert image_left.shape == image_right.shape
        self.reference_image = image_left.astype('float32')
        self.secondary_image = image_right.astype('float32')
        if reinit_messages:
            self._message_field = {'south': np.zeros(self.dimension, dtype=np.float32),
                                   'west': np.zeros(self.dimension, dtype=np.float32),
                                   'north': np.zeros(self.dimension, dtype=np.float32),
                                   'east': np.zeros(self.dimension, dtype=np.float32),
                                   'data': np.zeros(self.dimension, dtype=np.float32)}

        nrow, ncol = self.reference_image.shape
        # crop the border from the high right in the right image as it cannot be matched to the left
        if prior is not None:
            assert image_right.shape == prior.shape and prior.shape == self._message_field['data'].shape[1:]
            for l in xrange(self.n_levels):
                data_contrib = np.abs(self.reference_image[:, l:] - self.secondary_image[:, :ncol - l])

                prior_mask = (prior == l)[:, l:]
                if prior_influence_mode == 'adaptive':
                    prior_trust_factor = (data_contrib / data_contrib.max())[prior_mask]

                # equivalent to linear interpolating between the prior pixels (0 on certain locations only)
                # and the data (weighted by the trust factor = 1 - prior_trust_factor)
                self._message_field['data'][l, :, l:][prior_mask] = (1 - prior_trust_factor) * data_contrib[prior_mask]\
                                                                    + prior_trust_factor * -9.5
                self._message_field['data'][l, :, l:][~prior_mask] = data_contrib[~prior_mask]
        else:
            for l in xrange(self.n_levels):
                self._message_field['data'][l, :, l:] = np.abs(self.reference_image[:, l:]
                                                               - self.secondary_image[:, :ncol - l])

    def _update_message_fields(self):
        """
        Pass messages from each hidden and observable variable to the corresponding adjacent hidden. 
        Since messages get summed up, normalise them to prevent overflows.
        
        Returns:
            In-place method
        """
        for direction in ['south', 'west', 'north', 'east']:
            message_updates = np.sum([field for d, field in self._message_field.iteritems() if d != direction], axis=0)
            if direction == 'south':
                self._message_field[direction][:, 1:, :] = message_updates[:, :-1, :]
            elif direction == 'west':
                self._message_field[direction][:, :, :-1] = message_updates[:, :, 1:]
            elif direction == 'north':
                self._message_field[direction][:, :-1, :] = message_updates[:, 1:, :]
            elif direction == 'east':
                self._message_field[direction][:, :, 1:] = message_updates[:, :, :-1]
            # add normalisation to the message values, as they grow exponentially with the number of iterations
            norm_factor = np.max(self._message_field[direction], axis=0, keepdims=True)
            norm_factor[norm_factor == 0] = 1
            self._message_field[direction] /= norm_factor

    def _update_belief_field(self):
        """
        Find an optimal (Maximum A-Posteriori) assignment for each pixel. 
        
        Returns:
            In-place method
        """
        energy_field = np.sum([field for d, field in self._message_field.items()], axis=0)
        self._belief_field = np.argmin(energy_field, axis=0)

    def loop_belief(self, image_left, image_right, prior=None, prior_trust_factor=1.0,
                    prior_influence_mode='const', n_iter=10, reinit_messages=True):
        """
        Loopy Belief Propagation: initialise messages and pass them around iteratively. 
        Get MAP estimate after some fixed number of iterations.

        Args:
            image_left: the left input image (single channel only!)
            image_right: the right input image (single channel only!)
            prior: prior belief of pixels' disparity referenced to the left image
            prior_trust_factor: float in [0, 1] denoting the noiselessness of the prior.
            prior_influence_mode: str, how should prior be incorporated: `const` for a constant prior trust factor,
                `adaptive` for one that is proportional to the data score.
            n_iter: number of iterations for the belief propagation loop
            reinit_messages: whether the message field should be updated anew. This might be undesired, 
                when for example, the prior is updated every now and then, but the accumulated beliefs should
                be retained.

        Returns:
            In-place method
        """
        self._init_fields(image_left, image_right, prior, prior_trust_factor,
                          prior_influence_mode, reinit_messages=reinit_messages)
        times = []
        for i in xrange(n_iter):
            start_timer = time.time()
            self._update_message_fields()
            times.append(time.time() - start_timer)
        logger.debug("Belief propagation loop took {} sec on average".format(np.mean(times)))

    def get_map_belief(self):
        """
        Return the MAP estimate of the belief field. Look for minima across the disparity (n_levels) axis.
        
        Returns:
            A numpy array with the same shape as the left image, representing the computed disparity map 
        """
        self._update_belief_field()
        return self._belief_field

    def lbp(self, image_left, image_right, prior=None, prior_trust_factor=0.5, prior_influence_mode='const', n_iter=10):
        """
        A wrapper around loop_belief and get_map_belief. See the corresponding docstring for more information.
        
        Args:
            image_left: the left input image (single channel only!)
            image_right: the right input image (single channel only!)
            prior: prior belief of pixels' disparity referenced to the left image
            prior_trust_factor: float in [0, 1] denoting the noiselessness of the prior.
            prior_influence_mode: str, how should prior be incorporated: `const` for a constant prior trust factor,
                `adaptive` for one that is proportional to the data score.
            n_iter: number of iterations for the belief propagation loop

        Returns:
            A numpy array with the same shape as the left image, representing the computed disparity map
        """
        self.loop_belief(image_left=image_left, image_right=image_right,
                         prior=prior, prior_trust_factor=prior_trust_factor,
                         prior_influence_mode=prior_influence_mode,
                         n_iter=n_iter, reinit_messages=True)
        belief = self.get_map_belief()
        return belief
