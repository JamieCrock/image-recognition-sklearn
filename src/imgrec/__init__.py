"""Image recognition with classical machine learning in scikit-learn."""

from .data import ImageDataset, add_noise, augment, load
from .experiment import ExperimentResult, run
from .models import model_specs

__all__ = ["ExperimentResult", "ImageDataset", "add_noise", "augment", "load", "model_specs", "run"]
__version__ = "1.0.0"
