class TrimMFCCs():
    """Trims an MFCC sequence by removing the first coefficient."""
    def __call__(self, batch):
        # Shape: B x D x Tmax
        return batch[:, 1:, :]

class Standardize():
    """Frame-wise MFCC standardization."""
    def __call__(self, batch):
        # Shape: B x D x Tmax
        for sequence in batch:
            sequence -= sequence.mean(axis=0)
            sequence /= sequence.std(axis=0)
        return batch