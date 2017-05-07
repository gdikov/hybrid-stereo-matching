import time
import numpy as np


class StereoMRF(object):
    """
    Markov Random Field with loopy belief propagation (min-sum message passing). 
    """
    def __init__(self, dim, n_levels):
        self.n_levels = n_levels
        self.dimension = (n_levels,) + dim

    def _init_fields(self, image_left, image_right, prior, prior_trust_factor=0.5):
        """
        Initialise the message fields -- each hidden variable contains 5 message boxes from the 4 adjacent variables 
        (south, west, north, east) and the observed variable (data).
        
        Args:
            image_left: a numpy array representing the left image (in grayscale with values in [0, 1])
            image_right: a numpy array with the same structure and shape as left image, representing the right image 
            prior: a numpy array with the same shape as left image, providing alternative source of truth (or initial
             belief) about some pixels' corresponding disparity values. 
            prior_trust_factor: float in [0, 1] telling how much the prior should be trusted.

        Returns:

        """
        self.reference_image = image_left.astype('float32')
        self.secondary_image = image_right.astype('float32')
        self._message_field = {'south': np.zeros(self.dimension, dtype=np.float32),
                               'west': np.zeros(self.dimension, dtype=np.float32),
                               'north': np.zeros(self.dimension, dtype=np.float32),
                               'east': np.zeros(self.dimension, dtype=np.float32),
                               'data': np.zeros(self.dimension, dtype=np.float32)}
        nrow, ncol = self.reference_image.shape
        # crop the border from the high right in the right image as it cannot be mathed to the left
        if prior is not None:
            for l in xrange(self.n_levels):
                data_contrib = np.hstack((np.abs(self.reference_image[:, l:] - self.secondary_image[:, :ncol - l]),
                                          np.ones((nrow, l))))
                # data_contrib /= np.max(data_contrib) * 1/(np.mean(data_contrib))
                # print(data_contrib.min(), data_contrib.max(), data_contrib.mean())
                prior_mask = prior == l

                # equivalent to linear interpolating between the prior regions (having lowest possible value of -1)
                # and the data (weighted by the data_trust_factor = 1 - prior_trust_factor)
                self._message_field['data'][l, prior_mask] = -prior_trust_factor + (1 - prior_trust_factor) * data_contrib[prior_mask]
                self._message_field['data'][l, ~prior_mask] = data_contrib[~prior_mask]
        else:
            for l in xrange(self.n_levels):
                self._message_field['data'][l, :, :ncol - l] = np.abs(self.reference_image[:, l:]
                                                                      - self.secondary_image[:, :ncol - l])

    def _update_message_fields(self):
        """
        Pass messages from each hidden and observable variable to the corresponding adjacent hidden. 
        Since messages get summed up, normalise them to prevent overflows.
        
        Returns:

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

        """
        energy_field = np.sum([field for d, field in self._message_field.items()], axis=0)
        self._belief_field = np.argmin(energy_field, axis=0)

    def lbp(self, image_left, image_right, prior=None, prior_trust_factor=0.5, n_iter=10):
        """
        Loopy Belief Propagation: initialise messages and pass them around iteratively. 
        Get MAP estimate after some fixed number of iterations.
        
        Args:
            image_left: the left input image (single channel only!)
            image_right: the right input image (single channel only!)
            prior: prior belief of pixels' disparity referenced to the left image
            prior_trust_factor: float in [0, 1] denoting the noiselessness of the prior. 
            n_iter: number of iterations for the belief propagation loop

        Returns:
            A numpy array with the same shape as the left image, representing the computed disparity map
        """
        self._init_fields(image_left, image_right, prior, prior_trust_factor)
        times = []
        for i in xrange(n_iter):
            start_timer = time.time()
            self._update_message_fields()
            times.append(time.time() - start_timer)
        print("LBP loop took {} sec on average".format(np.mean(times)))
        self._update_belief_field()
        return self._belief_field
