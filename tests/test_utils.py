from functools import partial
import jax.numpy as jnp
import jax.random as random
from jaxtyping import Array, Float, Key, UInt
from jaxtyping import jaxtyped
from beartype import beartype
import numpy as np
import pytest
from beartype.typing import List, Tuple, cast

from renderer.utils import transpose_for_display

# Use seeds instead of directly storing PRNG keys
SEEDS: List[int] = [20230701]

class TestTransposeForDisplay:
    @pytest.mark.parametrize("seed", SEEDS)
    @pytest.mark.parametrize(
        "shape", [(1, 1), (1, 7), (3, 1), (20, 31, 3), (11, 11, 4)]
    )
    @pytest.mark.parametrize("flip_vertical", [True, False])
    @jaxtyped(typechecker=beartype)
    def test_transposed_shape_must_be_flipped_along_first_two_axis(
        self,
        seed: int,
        shape: Tuple[int, ...],
        flip_vertical: bool,
    ) -> None:
        # Create PRNG key inside the test
        prng_key: UInt[Array, "2"] = random.PRNGKey(seed)
        
        # Generate test matrix
        matrix: Float[Array, "fst snd *channel"] = random.uniform(
            prng_key,
            shape,
        )
        
        # Get transposed matrix
        transposed: Float[Array, "snd fst *channel"] = transpose_for_display(
            matrix, 
            flip_vertical=flip_vertical
        )

        # Verify shapes
        assert (
            np.array(matrix.shape) == np.array(shape)
        ).all(), "Matrix shape must not be changed"
        assert (
            np.array(transposed.shape) == np.array([shape[1], shape[0], *shape[2:]])
        ).all(), "Transposed shape must be flipped along first two axises"
        
        # Verify types
        assert isinstance(matrix, Float[Array, "fst snd *channel"])
        assert isinstance(transposed, Float[Array, "snd fst *channel"])

    @pytest.mark.parametrize("seed", SEEDS)
    @pytest.mark.parametrize(
        "shape", [(1, 1), (1, 7), (3, 1), (20, 31, 3), (11, 11, 4)]
    )
    @pytest.mark.parametrize("flip_vertical", [True, False])
    @jaxtyped(typechecker=beartype)
    def test_transposed_unique_values_and_count_must_be_the_same(
        self,
        seed: int,
        shape: Tuple[int, ...],
        flip_vertical: bool,
    ) -> None:
        # Create PRNG key inside the test
        prng_key: UInt[Array, "2"] = random.PRNGKey(seed)
        
        # Generate test matrix
        matrix: Float[Array, "fst snd *channel"] = random.uniform(
            prng_key,
            shape,
        )
        
        # Get transposed matrix
        transposed: Float[Array, "snd fst *channel"] = transpose_for_display(
            matrix, 
            flip_vertical=flip_vertical
        )

        # Get unique values and counts
        m, m_cnt = jnp.unique(
            matrix,
            return_counts=True,
        )
        t, t_cnt = jnp.unique(
            transposed,
            return_counts=True,
        )

        # Verify uniqueness properties
        assert jnp.all(
            m == t
        ), "Unique values must be the same"
        assert jnp.all(
            m_cnt == t_cnt
        ), "Unique values count must be the same"

    @pytest.mark.parametrize("seed", SEEDS)
    @pytest.mark.parametrize("shape", [(5, 3), (20, 31, 3), (11, 11, 4)])
    @jaxtyped(typechecker=beartype)
    def test_flip_vertical(
        self,
        seed: int,
        shape: Tuple[int, ...],
    ) -> None:
        # Create PRNG key inside the test
        prng_key: UInt[Array, "2"] = random.PRNGKey(seed)
        
        # Create partial functions for transform operations
        tf_f = partial(transpose_for_display, flip_vertical=True)
        t_f = partial(transpose_for_display, flip_vertical=False)

        # Generate test matrix
        matrix: Float[Array, "fst snd *channel"] = random.uniform(
            prng_key,
            shape,
        )
        
        # Apply transformations
        tf: Float[Array, "snd fst *channel"] = tf_f(matrix)
        t: Float[Array, "snd fst *channel"] = t_f(matrix)

        # Verify flipping changes the matrix
        assert jnp.any(
            t != tf
        ), "flipped vertical will change the matrix"

        # Test composition of transformations
        ttfttf = t_f(tf_f(t_f(tf_f(matrix))))
        assert jnp.all(
            ttfttf == matrix
        ), "flip twice, transpose 4 times should be identity"

        tt = t_f(t_f(matrix))
        assert jnp.all(
            tt == matrix
        ), "transpose twice should be identity"

        tftftftf = tf_f(tf_f(tf_f(tf_f(matrix))))
        assert jnp.all(
            tftftftf == matrix
        ), "transpose and flip 4 times should be identity"

        tftf = tf_f(tf_f(matrix))
        assert jnp.any(
            tftf != matrix
        ), "transpose and flip twice should not be identity"
