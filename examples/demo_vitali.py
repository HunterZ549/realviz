"""Demo: the Vitali set — a transversal of Q that cannot be measured.

The figure is a four-panel proof chain: equivalence classes modulo Q are
dense clouds (every window meets every class); the Axiom of Choice picks
one representative per class into ``V``; the rational translates tile the
circle disjointly; and translation invariance then forces the measure of
``V`` to contradict itself — so ``V`` is not Lebesgue measurable.

The true Vitali set is not constructible, so the figure is honest about
being a schematic, and ``VitaliInfo`` returns the load-bearing numbers so
the picture can be checked rather than believed:

- the class anchors ``0``, ``sqrt(2)-1``, ``sqrt(3)-1`` whose pairwise
  differences are provably irrational (distinct cosets);
- the chosen representatives ``{((k+1) sqrt(2)) mod 1}``;
- the ``n_reps x n_shifts`` translate points, pairwise distinct (mod 1),
  with the smallest circular gap between any two.

Run with:
    python examples/demo_vitali.py
"""

import matplotlib.pyplot as plt

from realviz import vitali_illustration


def main():
    _, _, info = vitali_illustration()
    print("Vitali illustration — checkable numbers:")
    print(f"  class anchors:      {tuple(round(a, 6) for a in info.class_anchors)}")
    print(f"  reps (|V| drawn):   {info.reps.size}")
    print(f"  translates drawn:   {info.translate_points.shape}")
    print(f"  distinct points:    {info.disjoint_count} "
          f"(of {info.reps.size * info.shifts.size} — the tiling is disjoint)")
    print(f"  min circular gap:   {info.min_gap:.6f}")
    plt.show()


if __name__ == "__main__":
    main()
