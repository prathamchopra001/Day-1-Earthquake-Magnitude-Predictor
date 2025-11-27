from sklearn.gaussian_process.kernels import (
    RBF,
    Matern,
    WhiteKernel,
    ConstantKernel,
    RationalQuadratic,
    Sum,
    Product
)
from typing import Optional

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.logger import setup_logger
from src.utils.helpers import load_config

logger = setup_logger('kernel')


def create_rbf_kernel(
    length_scale: float = 1.0,
    length_scale_bounds: tuple = (1e-2, 1e2)
) -> RBF:

    return RBF(
        length_scale=length_scale,
        length_scale_bounds=length_scale_bounds
    )


def create_matern_kernel(
    length_scale: float = 1.0,
    nu: float = 2.5,
    length_scale_bounds: tuple = (1e-2, 1e2)
) -> Matern:
    
    return Matern(
        length_scale=length_scale,
        nu=nu,
        length_scale_bounds=length_scale_bounds
    )


def create_white_kernel(
    noise_level: float = 0.1,
    noise_level_bounds: tuple = (1e-3, 1e1)
) -> WhiteKernel:
    
    return WhiteKernel(
        noise_level=noise_level,
        noise_level_bounds=noise_level_bounds
    )


def create_constant_kernel(
    constant_value: float = 1.0,
    constant_value_bounds: tuple = (1e-3, 1e3)
) -> ConstantKernel:
    
    return ConstantKernel(
        constant_value=constant_value,
        constant_value_bounds=constant_value_bounds
    )


def create_rational_quadratic_kernel(
    length_scale: float = 1.0,
    alpha: float = 1.0,
    length_scale_bounds: tuple = (1e-2, 1e2),
    alpha_bounds: tuple = (1e-2, 1e2)
) -> RationalQuadratic:
    
    return RationalQuadratic(
        length_scale=length_scale,
        alpha=alpha,
        length_scale_bounds=length_scale_bounds,
        alpha_bounds=alpha_bounds
    )


def create_simple_kernel(config_path: str = "config.yaml"):
    
    config = load_config(config_path)
    kernel_config = config['model']['kernel']
    
    length_scale = kernel_config['rbf_length_scale']
    length_scale_bounds = tuple(kernel_config['rbf_length_scale_bounds'])
    noise_level = kernel_config['white_noise_level']
    noise_bounds = tuple(kernel_config['white_noise_bounds'])
    
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3)) *
        RBF(length_scale, length_scale_bounds) +
        WhiteKernel(noise_level, noise_bounds)
    )
    
    logger.info(f"Created simple kernel: {kernel}")
    
    return kernel


def create_composite_kernel(config_path: str = "config.yaml"):
    
    config = load_config(config_path)
    kernel_config = config['model']['kernel']
    
    length_scale = kernel_config['rbf_length_scale']
    length_scale_bounds = tuple(kernel_config['rbf_length_scale_bounds'])
    noise_level = kernel_config['white_noise_level']
    noise_bounds = tuple(kernel_config['white_noise_bounds'])
    
    # RBF for smooth patterns
    rbf = RBF(
        length_scale=length_scale,
        length_scale_bounds=length_scale_bounds
    )
    
    # Matérn for rougher patterns
    matern = Matern(
        length_scale=length_scale,
        nu=2.5,
        length_scale_bounds=length_scale_bounds
    )
    
    # Combine: scale * (rbf + matern) + noise
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3)) *
        (rbf + matern) +
        WhiteKernel(noise_level, noise_bounds)
    )
    
    logger.info(f"Created composite kernel: {kernel}")
    
    return kernel


def create_advanced_kernel(config_path: str = "config.yaml"):
    
    config = load_config(config_path)
    kernel_config = config['model']['kernel']
    
    noise_level = kernel_config['white_noise_level']
    noise_bounds = tuple(kernel_config['white_noise_bounds'])
    
    # Multi-scale component
    rq = RationalQuadratic(
        length_scale=1.0,
        alpha=1.0,
        length_scale_bounds=(1e-2, 1e2),
        alpha_bounds=(1e-2, 1e2)
    )
    
    # Local variation component
    matern = Matern(
        length_scale=1.0,
        nu=1.5,
        length_scale_bounds=(1e-2, 1e2)
    )
    
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3)) * rq +
        ConstantKernel(0.5, (1e-3, 1e3)) * matern +
        WhiteKernel(noise_level, noise_bounds)
    )
    
    logger.info(f"Created advanced kernel: {kernel}")
    
    return kernel


def get_kernel(
    kernel_type: str = "simple",
    config_path: str = "config.yaml"
):
    
    kernel_factories = {
        "simple": create_simple_kernel,
        "composite": create_composite_kernel,
        "advanced": create_advanced_kernel
    }
    
    if kernel_type not in kernel_factories:
        raise ValueError(f"Unknown kernel type: {kernel_type}. Choose from {list(kernel_factories.keys())}")
    
    return kernel_factories[kernel_type](config_path)


# Test function
def test_kernels():
    """Test kernel creation."""
    print("Testing Kernel Creation:")
    print("=" * 60)
    
    # Simple kernel
    print("\n1. Simple Kernel:")
    simple = create_simple_kernel()
    print(f"   {simple}")
    
    # Composite kernel
    print("\n2. Composite Kernel:")
    composite = create_composite_kernel()
    print(f"   {composite}")
    
    # Advanced kernel
    print("\n3. Advanced Kernel:")
    advanced = create_advanced_kernel()
    print(f"   {advanced}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_kernels()