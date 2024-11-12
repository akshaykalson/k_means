import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import time
from multiprocessing import Pool, cpu_count

def generateData(n, c):
    logging.info(f"Generating {n} samples in {c} classes")
    X, y = make_blobs(n_samples=n, centers=c, cluster_std=1.7, shuffle=False,
                      random_state=2122)
    return X

def nearestCentroid_worker(args):
    data_point, centroids = args
    # norm(a-b) is Euclidean distance, matrix - vector computes difference
    # for all rows of matrix
    dist = np.linalg.norm(centroids - data_point, axis=1)
    return np.argmin(dist), np.min(dist)

def nearestCentroid_worker(args):
    data_point, centroids = args
    # norm(a-b) is Euclidean distance, matrix - vector computes difference
    # for all rows of matrix
    dist = np.linalg.norm(centroids - data_point, axis=1)
    return np.argmin(dist), np.min(dist)

def kmeans_worker(args):
    data_chunk, centroids, cluster_sizes = args
    N = len(data_chunk)
    k, features = centroids.shape  # Get the number of clusters and features

    # The cluster index: c[i] = j indicates that i-th datum is in j-th cluster
    c = np.zeros(N, dtype=int)
    variation = np.zeros(k)

    for i in range(N):
        cluster, dist = nearestCentroid_worker((data_chunk[i], centroids))
        c[i] = cluster
        cluster_sizes[cluster] += 1
        variation[cluster] += dist**2

    return c, variation

def kmeans(k, data, nr_iter=100, n_jobs=1):
    N, features = data.shape

    # Choose k random data points as centroids
    centroids = data[np.random.choice(np.array(range(N)), size=k, replace=False)]
    logging.debug("Initial centroids\n", centroids)

    # The cluster index: c[i] = j indicates that i-th datum is in j-th cluster
    c = np.zeros(N, dtype=int)

    logging.info("Iteration\tVariation\tDelta Variation")
    total_variation = 0.0
    for j in range(nr_iter):
        logging.debug(f"=== Iteration {j+1} ===")

        # Assign data points to nearest centroid
        variation = np.zeros(k)
        cluster_sizes = np.zeros(k, dtype=int)
        cluster_assignments = {x: [] for x in range(k)}

        for i in range(N):
            cluster, dist = nearestCentroid_worker((data[i], centroids))
            c[i] = cluster
            cluster_assignments[cluster].append(data[i])
            cluster_sizes[cluster] += 1
            variation[cluster] += dist**2

        delta_variation = -total_variation
        total_variation = np.sum(variation)
        delta_variation += total_variation

        logging.info("%3d\t\t%f\t%f" % (j, total_variation, delta_variation))

        # Recompute centroids
        centroids = np.zeros((k, features))  # This fixes the dimension to features
        for i in range(N):
            centroids[c[i]] += data[i]
        centroids = centroids / cluster_sizes.reshape(-1, 1)

        logging.debug(cluster_sizes)
        logging.debug(c)
        logging.debug(centroids)

    return total_variation, c

def computeClustering(args):
    if args.verbose:
        logging.basicConfig(format='# %(message)s', level=logging.INFO)
    if args.debug:
        logging.basicConfig(format='# %(message)s', level=logging.DEBUG)

    X = generateData(args.samples, args.classes)

    start_time = time.time()
    total_variation, assignment = kmeans(args.k_clusters, X, nr_iter=args.iterations, n_jobs=args.workers)
    end_time = time.time()
    logging.info("Clustering complete in %3.2f [s]" % (end_time - start_time))
    print(f"Total variation {total_variation}")

    if args.plot:  # Assuming 2D data
        fig, axes = plt.subplots(nrows=1, ncols=1)
        axes.scatter(X[:, 0], X[:, 1], c=assignment, alpha=0.2)
        plt.title("k-means result")
        # plt.show()
        fig.savefig(args.plot)
        plt.close(fig)

def measure_execution_time(args, X):
    start_time = time.time()
    kmeans(args.k_clusters, X, nr_iter=args.iterations, n_jobs=args.workers)
    end_time = time.time()
    return end_time - start_time

def run_experiment(workers, samples, k_clusters, iterations):
    X = generateData(samples, k_clusters)
    times = []

    for worker_count in workers:
        args = argparse.Namespace(
            workers=worker_count,
            k_clusters=k_clusters,
            iterations=iterations,
            samples=samples,
            classes=k_clusters,
            plot=None,  # Optional: provide a filename for plotting
            verbose=False,
            debug=False
        )

        print(f"Running with {worker_count} workers...")
        execution_time = measure_execution_time(args, X)
        print(f"Execution time: {execution_time} seconds")
        times.append(execution_time)

    return times

def theoretical_speedup(workers):
    parallelizable_fraction = 1.0 / (workers - 1)
    return 1.0 / ((1.0 - parallelizable_fraction) + parallelizable_fraction / workers)

def plot_speedup_graph(workers, measured_times, theoretical_speedups):
    measured_speedups = measured_times[0] / np.array(measured_times)
    
    plt.plot(workers, np.array(theoretical_speedups) * measured_times[0], label='Theoretical Execution Time', linestyle='--', color='red')
    plt.plot(workers, measured_times, label='Measured Execution Time', marker='o')

    plt.xlabel('Number of Workers')
    plt.ylabel('Execution Time (s)')
    plt.title('Parallel K-Means Execution Time')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Compute a k-means clustering.',
        epilog='Example: kmeans.py -v -k 4 --samples 10000 --classes 4 --plot result.png'
    )
    parser.add_argument('--workers', '-w',
                        default=cpu_count(),
                        type=int,
                        help='Number of parallel processes to use')
    parser.add_argument('--k_clusters', '-k',
                        default=3,
                        type=int,
                        help='Number of clusters')
    parser.add_argument('--iterations', '-i',
                        default=100,
                        type=int,
                        help='Number of iterations in k-means')
    parser.add_argument('--samples', '-s',
                        default=50000,  # Adjusted to at least 50,000 samples
                        type=int,
                        help='Number of samples to generate as input')
    parser.add_argument('--classes', '-c',
                        default=3,
                        type=int,
                        help='Number of classes to generate samples from')
    parser.add_argument('--plot', '-p',
                        type=str,
                        help='Filename to plot the final result')
    parser.add_argument('--verbose', '-v',
                        action='store_true',
                        help='Print verbose diagnostic output')
    parser.add_argument('--debug', '-d',
                        action='store_true',
                        help='Print debugging output')
    args = parser.parse_args()

    # Run experiment
    measured_times = run_experiment([args.workers], args.samples, args.k_clusters, args.iterations)

    # Calculate theoretical speedup
    theoretical_speedups = [theoretical_speedup(args.workers)]

    # Plot the speedup graph
    plot_speedup_graph([args.workers], measured_times, theoretical_speedups)