#!/usr/bin/env python3
import argparse
from decimal import Decimal, ROUND_HALF_UP


def decimal_arg(value):
    return Decimal(str(value))


def mass_label(mass):
    return str(mass).replace(".", "p")


def main():
    parser = argparse.ArgumentParser(
        description="Make a refined m_pi=m_rho mass grid for lxplus Condor production."
    )
    parser.add_argument("--min-mass", type=decimal_arg, default=Decimal("1.0"))
    parser.add_argument("--max-mass", type=decimal_arg, default=Decimal("250.0"))
    parser.add_argument("--step", type=decimal_arg, default=Decimal("0.1"))
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--seed", type=int, default=250001)
    parser.add_argument("--out", default="mass_points_refined_1to250.txt")
    args = parser.parse_args()

    if args.step <= 0:
        raise ValueError("--step must be positive")
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    if args.max_mass < args.min_mass:
        raise ValueError("--max-mass must be >= --min-mass")

    scale = Decimal("0.001")
    span = args.max_mass - args.min_mass
    n_masses = int((span / args.step).to_integral_value(rounding=ROUND_HALF_UP)) + 1

    point_id = 0
    with open(args.out, "w") as handle:
        handle.write("# point_id mrho mpi seed repeat mass_index mass_label\n")
        for mass_index in range(n_masses):
            mass = (args.min_mass + args.step * mass_index).quantize(scale)
            if mass > args.max_mass:
                break
            for repeat in range(args.repeats):
                point_seed = args.seed + point_id
                label = mass_label(mass)
                handle.write(
                    f"{point_id} {mass:.3f} {mass:.3f} {point_seed} "
                    f"{repeat} {mass_index} {label}\n"
                )
                point_id += 1

    print(f"Wrote {point_id} mpi=mrho jobs to {args.out}")
    print(f"Mass range: {args.min_mass} to {args.max_mass} GeV")
    print(f"Step: {args.step} GeV")
    print(f"Repeats per mass: {args.repeats}")


if __name__ == "__main__":
    main()
