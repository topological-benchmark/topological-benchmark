"""Demo: build a small TDA point-cloud dataset and round-trip it to disk."""

from __future__ import annotations

import numpy as np

from tda_shapes import (
    DEFAULT_SHAPES,
    CompositeDataset,
    ShapeDataset,
    make_composite_dataset,
    make_dataset,
    sample_composite,
)

DENSITY = 20.0
OUT_PATH = "tda_shapes_dataset.npz"
COMPOSITE_PATH = "tda_composite_dataset.npz"


def main() -> None:
    rng = np.random.default_rng(0)

    # Show how size controls the point count at fixed density.
    print(f"Constant density = {DENSITY} points per unit length/area\n")
    print(f"{'shape':<14}{'dim':>4}{'betti':>10}{'N@size=1':>10}{'N@size=2':>10}")
    print("-" * 48)
    for shape in DEFAULT_SHAPES:
        b = "".join(map(str, shape.betti))
        print(
            f"{shape.name:<14}{shape.intrinsic_dim:>4}{b:>10}"
            f"{shape.expected_n(DENSITY, 1.0):>10}"
            f"{shape.expected_n(DENSITY, 2.0):>10}"
        )

    # Anisotropic stretch keeps density constant on the stretched object.
    from tda_shapes import Sphere

    ell = Sphere().sample(density=DENSITY, size=1.5, noise=0.0, stretch=(1, 1, 2), rng=rng)
    print(f"\nSphere stretched to an ellipsoid (1,1,2): {len(ell)} points, uniform by area.")

    # Build, save, and reload a dataset (with random anisotropy for variety).
    ds = make_dataset(
        n_per_class=10, density=DENSITY, noise=0.02, stretch_range=(0.7, 1.5), rng=rng
    )
    print(f"\nGenerated {len(ds)} point clouds across {len(ds.label_names)} classes.")
    sizes = [c.shape[0] for c in ds.clouds]
    print(f"Point counts per cloud: min={min(sizes)}, max={max(sizes)}")

    ds.save(OUT_PATH)
    reloaded = ShapeDataset.load(OUT_PATH)
    assert len(reloaded) == len(ds)
    assert np.array_equal(reloaded.labels, ds.labels)
    assert np.array_equal(reloaded.betti, ds.betti)
    assert all(np.array_equal(a, b) for a, b in zip(reloaded.clouds, ds.clouds))
    print(f"Saved to {OUT_PATH} and verified round-trip.")

    # Composite scenes: k disjoint shapes -> Betti numbers add up.
    k = 3
    scene = sample_composite(k, density=DENSITY, noise=0.02, rng=rng)
    present = ", ".join(
        f"{n}x {scene.shape_names[i]}"
        for i, n in enumerate(scene.counts)
        if n
    )
    print(
        f"\nComposite of k={k} shapes ({present}): "
        f"combined betti={tuple(int(x) for x in scene.betti)}, {len(scene.points)} points."
    )

    cds = make_composite_dataset(30, k=k, density=DENSITY, noise=0.02, rng=rng)
    print(f"Generated {len(cds)} composite clouds; betti rows e.g. {cds.betti[:3].tolist()}")
    cds.save(COMPOSITE_PATH)
    rc = CompositeDataset.load(COMPOSITE_PATH)
    assert np.array_equal(rc.betti, cds.betti)
    assert np.array_equal(rc.shape_counts, cds.shape_counts)
    assert all(np.array_equal(a, b) for a, b in zip(rc.clouds, cds.clouds))
    print(f"Saved to {COMPOSITE_PATH} and verified round-trip.")


if __name__ == "__main__":
    main()
