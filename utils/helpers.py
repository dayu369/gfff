class TrimMFCCs():
    """Trims an MFCC sequence by removing the first coefficient."""
    def __call__(self, sequence):
        # Shape: B x D x Tmax
        return sequence[:, 1:, :]

class Standardize():
    """Frame-wise MFCC standardization."""
    def __call__(self, sequence):
        # Shape: B x D x Tmax
        return (sequence - sequence.mean(axis=1)) / sequence.std(axis=1)