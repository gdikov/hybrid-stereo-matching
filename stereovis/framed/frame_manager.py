import os
import skimage.io as skio


class FrameManager:
    def __init__(self, path_to_data):
        """
        Args:
            path_to_data: a dict with keys 'left' and 'right' containing the full paths to the stereo frames.  
        """
        self.path_to_data = path_to_data
        self.image_paths = {'left': [x for x in os.listdir(path_to_data['left']) if x.endswith(".png")],
                            'right': [x for x in os.listdir(path_to_data['right']) if x.endswith(".png")]}

    def _perform_data_consistency_check(self, sanitise=True):
        """
        Do simple sanity check on the frames dataset. Check if for each left there is a right image 
        and whether the timestamps match. Otherwise perform sanitisation if required.
        Args:
            sanitise: bool flag which triggers the removal of mismatching images from the lists. 
             This may lead to undesired time skips.  

        Returns:
            In-place method
        """
        timings_left = sorted([os.path.splitext(os.path.basename(image))[0] for image in self.image_paths['left']])
        timings_right = sorted([os.path.splitext(os.path.basename(image))[0] for image in self.image_paths['right']])
        if sanitise:
            raise NotImplementedError
        else:
            assert len(timings_left) == len(timings_right)
            assert all([x == y for x, y in zip(timings_left, timings_right)])

    def iter_frames(self):
        """
        Iterate over the stereo pairs of frames. 
        
        Returns:
            Yields a pair of left and right images
        """
        # pre-fetch the images, since loading one at a time during iteration will not be very efficient.
        # In offline mode it shouldn't matter but in online it will slow down the MRF matching significantly.
        frames_left = sorted([(os.path.splitext(os.path.basename(image))[0], skio.imread(image, as_grey=True))
                              for image in self.image_paths['left']], key=lambda x: x[0])
        frames_right = sorted([(os.path.splitext(os.path.basename(image))[0], skio.imread(image, as_grey=True))
                              for image in self.image_paths['right']], key=lambda x: x[0])



        for i in xrange(len(frames_left)):
            timestamp_left = int(frames_left[i][0])
            timestamp_right = int(frames_right[i][0])
            assert timestamp_left - timestamp_right == 0
            yield frames_left[i][1], frames_right[i][1], timestamp_left
