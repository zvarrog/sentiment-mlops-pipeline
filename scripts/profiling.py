"""Performance profiling utilities for training and API inference."""

import cProfile
import pstats
import time
from pathlib import Path


def profile_training():
    """Profile training pipeline to identify bottlenecks."""
    profiler = cProfile.Profile()

    profiler.enable()
    from scripts.train import run

    run()
    profiler.disable()

    # Save results
    output_dir = Path("profiling")
    output_dir.mkdir(exist_ok=True)

    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative")
    stats.dump_stats(str(output_dir / "training_profile.prof"))

    # Print top 20 slowest functions
    stats.print_stats(20)


def profile_api():
    """Profile API inference performance."""
    import joblib

    from scripts.train_modules.data_loading import load_splits

    model = joblib.load("artefacts/best_model.joblib")
    _, _, X_test, _, _, _ = load_splits()

    sample = X_test.head(100)

    # Warm-up
    model.predict(sample)

    # Measure
    times = []
    for _ in range(10):
        start = time.perf_counter()
        model.predict(sample)
        times.append(time.perf_counter() - start)

    avg_time = sum(times) / len(times)
    print(f"Average time per 100 predictions: {avg_time:.3f}s")
    print(f"Per prediction: {avg_time / 100 * 1000:.1f}ms")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "train":
        profile_training()
    else:
        profile_api()
