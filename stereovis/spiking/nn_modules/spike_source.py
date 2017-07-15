import logging
try:
    import spynnaker7.pyNN as pyNN
except ImportError:
    import spynnaker.pyNN as pyNN

logger = logging.getLogger(__file__)


class Retina(object):
    def __init__(self, spike_times=None, label='retina', mode='offline'):
        """
        Args:
            spike_times: list of lists of lists for the spiking of each neuron in each population
            label: label to distinguish between left and right SpikeSourceArray populations
            mode: str, can be 'online' or 'offline' and decides whether the spiking is 
             from pre-recorded input (`spike_times`) or live input stream
        """
        # NOTE: numpy is row-major so the first axis represents the pixel columns (vertical) i.e. fixed x-coordinate
        # and the second the pixels within this column i.e. all y-coordinates
        # here n_cols and n_rows stand for the column_populations and rows within this column (not numpy rows/columns!)
        self.n_cols, self.n_rows = spike_times.shape

        self.pixel_populations = []
        self.label = label

        if mode == 'offline':
            self._init_spiketimes(spike_times)
        else:
            logger.error("Live input streaming is not supported yet. Provide a spike_times array instead.")
            raise NotImplementedError

    def _init_spiketimes(self, spike_times):
        """
        Initialise the spiking times of an offline Retina SpikeSourceArray populations
        
        Args:
            spike_times: a list of lists of lists for the spiking times of all the neurons 

        Returns:
            In-place method
        """
        logger.info("Initialising {} SpikeSourceArray with resolution {}.".format(self.label, (self.n_cols,
                                                                                               self.n_rows)))
        if spike_times is not None:
            for col_id, col in enumerate(spike_times):
                col_of_pixels = pyNN.Population(size=self.n_rows,
                                                cellclass=pyNN.SpikeSourceArray,
                                                cellparams={'spike_times': col},
                                                label="{0}_{1}".format(self.label, col_id), structure=pyNN.Line())

                self.pixel_populations.append(col_of_pixels)


def create_retina(spike_times=None, label='retina', mode='offline'):
    """
    A wrapper around the Retina class.
    
    Args:
        spike_times: see Retina docstring 
        label: see Retina docstring
        mode: see Retina docstring

    Returns:
        A list of SpikeSourceArray populations representing the retina pixel columns.
    """
    retina = Retina(spike_times, label, mode)
    return retina.pixel_populations
