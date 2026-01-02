import numpy as np

def OVL(target, background):
    """
    Calculate the overlap of two distributions.
    
    Parameters:
    target (array-like): Sample values from the target distribution.
    background (array-like): Sample values from the background distribution.
    
    Returns:
    float: Overlap value between the two distributions.
    """
    # Determine the number of bins using Sturges' formula
    nbins = round(1 + np.log2(len(background)))
    
    # Define bin edges
    x = np.linspace(min(min(target), min(background)), max(max(target), max(background)), nbins)
    
    # Estimate probability density function (normalized histogram)
    pdf_t, _ = np.histogram(target, bins=x, density=True)
    pdf_b, _ = np.histogram(background, bins=x, density=True)

    print(pdf_t.shape)
    print(pdf_b.shape)
    
    # Compute overlap
    OVL = np.sum(np.minimum(pdf_t / np.sum(pdf_t), pdf_b / np.sum(pdf_b)))
    print(pdf_t.shape)
    print(pdf_b.shape)
    
    return OVL



# example usage
# overlap = OVL((all_features_pose[test_cat_labels == 1].reshape(3960*1280)).numpy(), all_features_frontal.reshape(3960*1280).numpy())

'''
    features from cnn
'''